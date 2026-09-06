"""Continuous-signing inference: a whole utterance video -> a gloss sequence.

The recogniser is trained on *isolated* signs (median 54 frames). A real-world
recording is a continuous sentence: several signs run together, bracketed by
rest positions and separated by short transitions. Feeding the whole video to an
isolated-sign classifier asks it to name a sentence with one word, which it
correctly refuses to do.

This module inserts the missing stage:

    video -> landmarks -> motion energy -> segmentation -> per-segment
    recognition (with test-time augmentation) -> open-set gate -> gloss sequence

Segmentation is unsupervised and threshold-light: sign strokes are high-energy,
transitions and rests are low-energy, and the gap between the two is roughly two
orders of magnitude in these recordings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import L_SHOULDER, R_SHOULDER, FeatureConfig

EPS = 1e-6
L_HAND_WRIST, R_HAND_WRIST = 33, 54


@dataclass
class SegmentConfig:
    """Defaults selected on utterances stitched from **validation** clips, scored
    across both continuity regimes (pause-separated and coarticulated) and then
    frozen before touching held-out test utterances.

    What matters most is `gap` and `active_pct`: a within-sign dip must be
    bridged while a real inter-sign boundary survives. Tightening either
    over-segments badly - the first version of this decoder used pct=35 / gap=6
    and scored WER 1.045 against ground truth, versus 0.36 here.
    """
    smooth: int = 9            # frames, moving average on the energy curve
    active_pct: float = 18.0   # energy percentile that separates stroke from rest
    min_len: int = 26          # drop anything shorter than this many frames
    max_len: int = 100         # split segments longer than this at their energy minima
    gap: int = 18              # bridge sub-gap pauses inside one sign
    pad: int = 3               # frames of context added on each side
    tta: int = 3               # test-time crops per segment (1 = off)


def body_scale(clip: np.ndarray):
    pose = clip[:, :33]
    neck = 0.5 * (pose[:, L_SHOULDER] + pose[:, R_SHOULDER])
    sw = np.linalg.norm(pose[:, L_SHOULDER] - pose[:, R_SHOULDER], axis=-1)
    scale = float(np.median(sw[sw > EPS])) if np.any(sw > EPS) else 1.0
    return neck, max(scale, 1e-3)


def motion_energy(clip: np.ndarray, smooth: int = 9) -> np.ndarray:
    """Combined wrist speed in body units - high during a stroke, ~0 at rest."""
    neck, scale = body_scale(clip)
    lw = (clip[:, L_HAND_WRIST] - neck) / scale
    rw = (clip[:, R_HAND_WRIST] - neck) / scale
    # an undetected hand contributes no motion rather than a jump to the origin
    lok = ~(clip[:, L_HAND_WRIST] == 0).all(-1)
    rok = ~(clip[:, R_HAND_WRIST] == 0).all(-1)
    v = np.zeros(len(clip), dtype=np.float32)
    for w, ok in ((lw, lok), (rw, rok)):
        d = np.linalg.norm(np.diff(w, axis=0, prepend=w[:1]), axis=-1)
        d[~ok] = 0.0
        v += d
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / smooth
        v = np.convolve(np.pad(v, smooth // 2, mode="edge"), k, mode="valid")[:len(clip)]
    return v


def _runs(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(idx[0::2], idx[1::2]))          # [start, end)


def _split_recursive(s: int, t: int, e: np.ndarray, cfg: SegmentConfig, out: list):
    """Halve an over-long run at its lowest-energy point, preferring cuts near
    the middle, then recurse.

    Fluent signing has no rest between signs - coarticulation means the energy
    curve never returns to the floor - so for real utterances this recursion,
    not the pause detector, is what finds the boundaries. A greedy left-to-right
    scan would take the global minimum and leave one long unbalanced tail.
    """
    if t - s <= cfg.max_len:
        out.append((s, t))
        return
    lo, hi = s + cfg.min_len, t - cfg.min_len
    if hi <= lo:
        out.append((s, t))
        return
    idx = np.arange(lo, hi)
    centre = 0.5 * (s + t)
    # penalise off-centre cuts so the two halves stay comparable in length
    bias = 1.0 + 0.6 * np.abs(idx - centre) / max(1.0, 0.5 * (t - s))
    cut = int(idx[int(np.argmin(e[lo:hi] * bias))])
    _split_recursive(s, cut, e, cfg, out)
    _split_recursive(cut, t, e, cfg, out)


def segment_by_motion(clip: np.ndarray, cfg: SegmentConfig = SegmentConfig()):
    """Unsupervised split of an utterance into candidate sign segments."""
    e = motion_energy(clip, cfg.smooth)
    thr = float(np.percentile(e, cfg.active_pct))
    active = e > thr

    # bridge short dips inside a single sign
    segs = _runs(active)
    merged = []
    for s, t in segs:
        if merged and s - merged[-1][1] <= cfg.gap:
            merged[-1] = (merged[-1][0], t)
        else:
            merged.append((s, t))

    # split over-long segments at their internal energy minima
    out = []
    for s, t in merged:
        _split_recursive(int(s), int(t), e, cfg, out)

    T = len(clip)
    final = [(max(0, s - cfg.pad), min(T, t + cfg.pad))
             for s, t in out if (t - s) >= cfg.min_len]
    return final, e, thr


def _crops(clip: np.ndarray, s: int, t: int, n: int):
    """Test-time augmentation: n slightly different crops of the same segment."""
    if n <= 1:
        return [clip[s:t]]
    T = len(clip)
    span = t - s
    jit = max(1, int(round(span * 0.08)))
    out = []
    for k in range(n):
        d = 0 if k == 0 else (jit if k % 2 else -jit)
        a, b = max(0, s - d), min(T, t + d)
        if b - a >= 8:
            out.append(clip[a:b])
    return out or [clip[s:t]]


def decode(predictor, clip: np.ndarray, cfg: SegmentConfig = SegmentConfig(),
           topk: int = 5) -> dict:
    """Whole utterance -> per-segment predictions + collapsed gloss sequence.

    Each segment is classified with `cfg.tta` crops whose probabilities are
    averaged, then the open-set gate decides gloss vs. `_`.
    """
    segments, energy, thr = segment_by_motion(clip, cfg)
    if not segments:
        return {"segments": [], "sequence": [], "energy": energy, "threshold": thr}

    flat, owner = [], []
    for i, (s, t) in enumerate(segments):
        for c in _crops(clip, s, t, cfg.tta):
            flat.append(c)
            owner.append(i)
    raw = predictor.predict_batch(flat, topk=len(predictor.class_names))
    owner = np.asarray(owner)

    rows = []
    for i, (s, t) in enumerate(segments):
        members = np.flatnonzero(owner == i)
        prob = np.zeros(len(predictor.class_names))
        for m in members:
            for g, p in raw[m]["topk"]:
                prob[predictor.class_names.index(g)] += p
        prob /= len(members)
        score = float(np.mean([raw[m]["openset_score"] for m in members]))
        proto = float(np.mean([raw[m]["proto_similarity"] for m in members]))
        order = np.argsort(-prob)[:topk]
        accepted = score >= predictor.threshold
        best = predictor.class_names[int(order[0])]
        rows.append({
            "index": i, "start": int(s), "end": int(t),
            "frames": int(t - s),
            "gloss_id": best if accepted else "_",
            "best_guess": best,
            "is_known": bool(accepted),
            "confidence": float(prob[order[0]]),
            "openset_score": score,
            "proto_similarity": proto,
            "topk": [(predictor.class_names[int(j)], float(prob[j])) for j in order],
            "n_crops": len(members),
        })

    # collapse immediate repeats: one sign split across two segments should not
    # be read out twice
    seq, prev = [], None
    for r in rows:
        if r["gloss_id"] != prev or r["gloss_id"] == "_":
            seq.append(r["gloss_id"])
        prev = r["gloss_id"]
    return {"segments": rows, "sequence": seq, "energy": energy, "threshold": thr}


def sliding_windows(clip: np.ndarray, win: int = 54, stride: int = 8):
    T = len(clip)
    if T <= win:
        return [(0, T)]
    return [(s, s + win) for s in range(0, T - win + 1, stride)]


def decode_sliding(predictor, clip: np.ndarray, win: int = 54, stride: int = 6,
                   min_run: int = 3, topk: int = 5) -> dict:
    """Boundary-free alternative: classify a dense sliding window, then collapse
    consecutive identical labels into runs (a CTC-style read-out).

    No segmentation thresholds at all - the cost is that a sign shorter than the
    window can be swallowed by its neighbour.
    """
    wins = sliding_windows(clip, win, stride)
    res = predictor.predict_batch([clip[s:t] for s, t in wins], topk=topk)
    labels = [r["topk"][0][0] if r["openset_score"] >= predictor.threshold else "_"
              for r in res]

    rows, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        run = j - i + 1
        if run >= min_run:
            members = list(range(i, j + 1))
            rows.append({
                "index": len(rows),
                "start": int(wins[i][0]), "end": int(wins[j][1]),
                "frames": int(wins[j][1] - wins[i][0]),
                "gloss_id": labels[i],
                "best_guess": res[i]["topk"][0][0],
                "is_known": labels[i] != "_",
                "confidence": float(np.mean([res[m]["confidence"] for m in members])),
                "openset_score": float(np.mean([res[m]["openset_score"] for m in members])),
                "proto_similarity": float(np.mean([res[m]["proto_similarity"] for m in members])),
                "topk": res[i]["topk"],
                "n_crops": run,
            })
        i = j + 1
    seq, prev = [], None
    for r in rows:
        if r["gloss_id"] != prev or r["gloss_id"] == "_":
            seq.append(r["gloss_id"])
        prev = r["gloss_id"]
    return {"segments": rows, "sequence": seq,
            "energy": motion_energy(clip), "threshold": predictor.threshold}


def stability(predictor, clip: np.ndarray, win: int = 54, stride: int = 8) -> dict:
    """Do overlapping windows agree? Random flicker means the representation is
    not tracking anything; stable runs mean it is."""
    wins = sliding_windows(clip, win, stride)
    res = predictor.predict_batch([clip[s:t] for s, t in wins], topk=1)
    top = [r["topk"][0][0] for r in res]
    agree = float(np.mean([a == b for a, b in zip(top[:-1], top[1:])])) if len(top) > 1 else 0.0
    return {"windows": wins, "top1": top,
            "scores": np.array([r["openset_score"] for r in res]),
            "confidence": np.array([r["confidence"] for r in res]),
            "adjacent_agreement": agree,
            "n_distinct": len(set(top))}


def stitch_utterance(clips: list, rest: int = 14, blend: int = 6) -> np.ndarray:
    """Glue isolated clips into one continuous stream, for a decoder control set
    where the gloss sequence is known.

    Two regimes are reachable:
      `rest=14, blend=6` - pause-separated, the easy case;
      `rest=0,  blend=10` - coarticulated, which is what fluent signing (and the
      `data_test` recordings) actually look like.
    """
    out = [clips[0]]
    for c in clips[1:]:
        prev = out[-1][-1]
        parts = []
        if rest > 0:
            parts.append(np.repeat(prev[None], rest, axis=0))
        ramp = np.linspace(0, 1, blend)[:, None, None].astype(np.float32)
        a, b = prev[None], c[0][None]
        bl = a * (1 - ramp) + b * ramp
        bad = (a == 0).all(-1) | (b == 0).all(-1)
        bl[:, bad[0]] = b[:, bad[0]]                 # never blend through a gap
        parts += [bl.astype(np.float32), c]
        out.extend(parts)
    return np.concatenate(out).astype(np.float32)


def extraction_sanity(clip: np.ndarray) -> dict:
    """Confirm a freshly extracted video uses the corpus landmark convention.

    Slot 33 (left hand root) must sit on pose landmark 15 (left wrist) and slot
    54 on pose 16. If a video extractor swapped handedness, these flip and every
    downstream feature is mirrored.
    """
    ok_l = ~(clip[:, L_HAND_WRIST] == 0).all(-1)
    ok_r = ~(clip[:, R_HAND_WRIST] == 0).all(-1)
    d = lambda a, b, m: (float(np.linalg.norm(clip[m, a] - clip[m, b], axis=-1).mean())
                         if m.any() else float("nan"))
    sw = np.linalg.norm(clip[:, L_SHOULDER] - clip[:, R_SHOULDER], axis=-1)
    return {
        "lhand_to_left_wrist": d(L_HAND_WRIST, 15, ok_l),
        "lhand_to_right_wrist": d(L_HAND_WRIST, 16, ok_l),
        "rhand_to_right_wrist": d(R_HAND_WRIST, 16, ok_r),
        "rhand_to_left_wrist": d(R_HAND_WRIST, 15, ok_r),
        "shoulder_width": float(np.median(sw)),
        "left_shoulder_x_gt_right": bool(np.median(clip[:, L_SHOULDER, 0]) >
                                         np.median(clip[:, R_SHOULDER, 0])),
        "pose_detected": float((~(clip[:, :33] == 0).all(-1).all(-1)).mean()),
        "lhand_detected": float(ok_l.mean()),
        "rhand_detected": float(ok_r.mean()),
    }


def distribution_shift(clip: np.ndarray, stats: dict, fcfg: FeatureConfig) -> dict:
    """Mean |z| of each stream against the training standardiser - a cheap check
    that a newly extracted video landed in the same feature space as the corpus."""
    from .feature_engineering import clip_to_streams
    s = clip_to_streams(clip, fcfg)
    return {k: float(np.abs((v - stats[k][0]) / stats[k][1]).mean()) for k, v in s.items()}
