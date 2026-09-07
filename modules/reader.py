"""Deployment path: a real video in, a checked gloss sequence out.

This is the object the whole v2 exists to produce. Given any recording - a
`data_test/` mp4, a webcam capture, a `.npy` in the corpus layout - it returns

    * how many signs it saw and **where each one is in the video**,
    * for each: the gloss, or `_` when the reader declines to name it,
    * for each: *why* it declined - low sequence posterior, or a shape the word
      model's metric space does not recognise as any of the 184 glosses,
    * the Thai lemma, when a lexicon supplies one.

Two models cooperate, each doing what it is good at:

    CSR-Net (v2)   segmentation + sequence: what signs, in what order, when.
                   Trained on coarticulated utterances, so it is the only part
                   that has ever seen signs run together.
    MSSF-Net (v1)  the open-set judge. Its ArcFace metric space and class
                   prototypes were built for exactly this question - "is this
                   any of the 184?" - and were calibrated on validation in v1.
                   Reusing them means the rejection behaviour is inherited, not
                   re-invented, and the two systems' verdicts can disagree
                   visibly rather than silently.

A sign is emitted only if **both** agree. That is deliberately conservative: on
a vocabulary this small, a confident wrong Thai word is worse than a gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import CFG
from .feature_engineering import apply_standardiser, clip_to_streams
from .lexicon import Lexicon
from .seq_model import collapse_runs, greedy_ctc

UNKNOWN = "_"
DOWNSAMPLE = 4


@dataclass
class ContinuousReader:
    csr: object                       # CSRNet
    word: object                      # MSSFNet, for the open-set judgement
    stats: dict
    class_names: list
    prototypes: np.ndarray
    fuse_stats: dict
    device: object
    word_threshold: float = -1.47     # v1's validation-calibrated open-set gate
    seq_threshold: float = 0.0        # CSR-Net posterior gate, calibrated in v2
    lexicon: Lexicon | None = None
    use_word_gate: bool = True
    min_run: int = 3                  # frame-head read-out: minimum steps per sign

    # ------------------------------------------------------------------
    def _forward(self, clip: np.ndarray) -> dict:
        s = clip_to_streams(np.asarray(clip, np.float32), CFG.feature, resample=False)
        s = apply_standardiser({k: v[None] for k, v in s.items()}, self.stats)
        xb = {k: torch.from_numpy(v).to(self.device) for k, v in s.items()}
        mask = torch.ones(1, xb["lh"].shape[1], dtype=torch.bool, device=self.device)
        self.csr.eval()
        with torch.no_grad():
            out = self.csr(xb, mask)
        return {
            "logp": F.log_softmax(out["ctc"].float(), -1)[0].cpu().numpy(),
            "frame": F.softmax(out["frame"].float(), -1)[0].cpu().numpy(),
            "bnd": torch.sigmoid(out["bnd"].float())[0].cpu().numpy(),
            "T": int(xb["lh"].shape[1]),
        }

    # ------------------------------------------------------------------
    def _word_scores(self, clip: np.ndarray, spans: list) -> list:
        """Score every decoded span with the v1 word model + prototypes.

        The crop handed over is the span padded a little, because CTC peaks sit
        near the middle of a sign and the word model wants the whole stroke.
        """
        from .openset import fuse, scores
        if not spans or not self.use_word_gate:
            return [{"openset": float("nan"), "proto": float("nan"),
                     "word_gloss": None, "word_p": float("nan")} for _ in spans]
        crops = []
        for s, e in spans:
            pad = max(4, (e - s) // 4)
            a, b = max(0, s - pad), min(len(clip), e + pad)
            crops.append(clip[a:b] if b - a >= 8 else clip[max(0, b - 8):b])
        banks = [clip_to_streams(c, CFG.feature) for c in crops]
        banks = [apply_standardiser({k: v[None] for k, v in b.items()}, self.stats)
                 for b in banks]
        xb = {k: torch.from_numpy(np.concatenate([b[k] for b in banks])).to(self.device)
              for k in ("lh", "rh", "pose", "face")}
        self.word.eval()
        with torch.no_grad():
            out = self.word(xb)
        logits = out["logits"].float().cpu().numpy()
        emb = out["embed"].float().cpu().numpy()
        sc = scores(logits, emb, self.prototypes)
        fused, _ = fuse(sc, self.fuse_stats)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return [{"openset": float(fused[i]), "proto": float(sc["proto"][i]),
                 "word_gloss": self.class_names[int(p[i].argmax())],
                 "word_p": float(p[i].max())} for i in range(len(spans))]

    # ------------------------------------------------------------------
    def _snap(self, tok: dict, runs: list, T: int, min_frames: int = 32) -> tuple:
        """Turn a CTC peak into a usable frame span."""
        peak = tok.get("peak", tok["start"])
        best = None
        for r in runs:
            if r["start"] <= peak < r["end"]:
                best = r
                break
            if best is None and r["token"] == tok["token"]:
                best = r                      # same label nearby, if no overlap
        s4, e4 = (best["start"], best["end"]) if best else (tok["start"], tok["end"])
        s, e = s4 * DOWNSAMPLE, e4 * DOWNSAMPLE
        if e - s < min_frames:                # widen a sliver around its centre
            c = (s + e) // 2
            s, e = c - min_frames // 2, c + min_frames // 2
        return int(np.clip(s, 0, max(T - 1, 0))), int(np.clip(e, 1, T))

    def _energy_tokens(self, clip: np.ndarray, f: dict) -> list:
        """Segment with the v1 motion-energy detector, label with CSR-Net.

        Why this exists. On the corpus-like utterances CSR-Net's own read-outs win
        comfortably. On the two real recordings they do not: the sequence head is
        trained by a sequence loss, so on genuinely out-of-distribution input it
        does the safe thing and emits blank almost everywhere - it found one sign
        in a four-sign clause. v1's motion-energy segmenter has no learned
        confidence to collapse and cut both recordings at the right number of
        signs.

        So the two are combined along the axis each is good at: **boundaries from
        the unsupervised heuristic, labels from the model that has seen
        coarticulation**. It degrades gracefully where the learned segmentation
        goes quiet, at the cost of the heuristic's own thresholds.
        """
        from .continuous import SegmentConfig, segment_by_motion
        spans, _, _ = segment_by_motion(clip, SegmentConfig())
        prob, blank = f["frame"], self.csr.blank_id
        toks = []
        for s, e in spans:
            a, b = int(np.floor(s / DOWNSAMPLE)), int(np.ceil(e / DOWNSAMPLE))
            a, b = max(0, a), min(len(prob), max(b, a + 1))
            p = prob[a:b].mean(0).copy()
            p[blank] = 0.0                       # a segment is a sign by assumption
            k = int(p.argmax())
            toks.append({"token": k, "start": a, "end": b,
                         "confidence": float(p[k]),
                         "peak": int(a + prob[a:b, k].argmax())})
        return toks

    def read(self, clip: np.ndarray, head: str = "ctc") -> dict:
        """Whole recording -> segments, sequence and every diagnostic behind it.

        `head` selects the read-out: ``ctc`` (sequence-trained, sharpest on
        corpus-like material), ``frame`` (contiguous spans from the frame
        mapping), or ``energy`` (motion-energy boundaries + model labels, the
        most robust on unseen recordings).
        """
        f = self._forward(clip)
        logp, T = f["logp"], f["T"]
        if head == "ctc":
            toks = greedy_ctc(logp, self.csr.blank_id)
        elif head == "energy":
            toks = self._energy_tokens(np.asarray(clip, np.float32), f)
        else:
            toks = collapse_runs(f["frame"].argmax(-1), self.csr.blank_id, self.min_run)
            for t in toks:
                seg = logp[t["start"]:t["end"]]
                t["confidence"] = float(np.exp(seg.max(-1)).mean())
                t["peak"] = int(t["start"] + np.argmax(seg.max(-1)))

        if head == "energy":
            # the boundaries are the whole point of this read-out; do not move them
            spans = [(min(t["start"] * DOWNSAMPLE, T - 1),
                      min(t["end"] * DOWNSAMPLE, T)) for t in toks]
        else:
            # CTC decides *what* and in *what order*; its peaks are one or two
            # steps wide, which is useless as a location. The frame head's runs
            # are contiguous by construction, so each peak is snapped to the run
            # that contains it. Best of both: the sequence comes from the
            # sequence loss, the extent from the head trained on the frame mapping.
            runs = collapse_runs(f["frame"].argmax(-1), self.csr.blank_id, 1)
            spans = [self._snap(t, runs, T) for t in toks]
        wsc = self._word_scores(np.asarray(clip, np.float32), spans)

        # The sequence gate is calibrated on CTC peak posteriors. The energy
        # read-out's confidence is a mean frame posterior over a whole segment -
        # a different quantity on a different scale - so that gate does not
        # transfer, and this read-out is gated by the word model alone, which is
        # exactly the gate v1 calibrated for whole-segment crops.
        use_seq_gate = head != "energy"

        segs = []
        for t, (s, e), w in zip(toks, spans, wsc):
            tok = t["token"]
            model_says_unk = bool(self.csr.use_unk and tok == self.csr.unk_id)
            gid = None if model_says_unk else self.class_names[tok]
            seq_ok = (not use_seq_gate) or t["confidence"] >= self.seq_threshold
            word_ok = (not self.use_word_gate) or (not np.isfinite(w["openset"])) \
                or (w["openset"] >= self.word_threshold)
            accepted = (not model_says_unk) and seq_ok and word_ok
            reason = ("accepted" if accepted else
                      "model emitted <unk>" if model_says_unk else
                      "sequence posterior below gate" if not seq_ok else
                      "outside the word model's metric space")
            segs.append({
                "start": int(s), "end": int(e), "frames": int(e - s),
                "step_start": int(t["start"]), "step_end": int(t["end"]),
                "gloss_id": gid if accepted else UNKNOWN,
                "best_guess": gid,
                "thai": (self.lexicon.thai(gid) if (self.lexicon and gid and accepted)
                         else (UNKNOWN if not accepted else f"GLOSS_{gid}")),
                "is_known": accepted, "reason": reason,
                "confidence": float(t["confidence"]),
                "openset_score": w["openset"], "proto_similarity": w["proto"],
                "word_model_says": w["word_gloss"], "word_model_p": w["word_p"],
            })

        seq, prev = [], None
        for r in segs:
            if r["gloss_id"] != prev or r["gloss_id"] == UNKNOWN:
                seq.append(r["gloss_id"])
            prev = r["gloss_id"]

        return {"segments": segs, "sequence": seq, "n_signs_seen": len(segs),
                "posteriorgram": np.exp(logp), "frame_posterior": f["frame"],
                "activity": f["bnd"], "n_frames": T, "downsample": DOWNSAMPLE,
                "head": head}

    # ------------------------------------------------------------------
    def read_video(self, path: Path, max_frames: int = 900,
                   square_pixels: bool = True) -> dict:
        """The deployment entry point. `square_pixels` defaults on here because a
        recording whose frame aspect differs from the corpus arrives
        geometrically distorted (see `inference.aspect_correct`); the notebook
        turns it off in one place only, to show the uncorrected baseline."""
        from .inference import extract_from_video
        clip = extract_from_video(Path(path), max_frames=max_frames,
                                  square_pixels=square_pixels)
        out = self.read(clip)
        out["clip"] = clip
        out["path"] = str(path)
        return out

    # ------------------------------------------------------------------
    def calibrate(self, pool, target_accept: float = 0.90,
                  batch_size: int = 8) -> float:
        """Set `seq_threshold` from **validation** utterances only.

        The threshold is the posterior below which a CTC peak is not worth
        emitting. It is picked so that `target_accept` of peaks that are
        *correct* survive - i.e. the gate is tuned against known-good material,
        never against the test set, exactly as v1 calibrated its own gate.
        """
        confs = []
        rng = np.random.default_rng(0)
        self.csr.eval()
        for ids in pool._order(rng, False, batch_size):
            xb, m, _, _, _ = pool.batch(ids, self.device, self.csr)
            with torch.no_grad():
                out = self.csr(xb, m)
            lp = F.log_softmax(out["ctc"].float(), -1).cpu().numpy()
            m4 = out["mask4"].cpu().numpy()
            for r, i in enumerate(ids):
                ref = [int(x) for x in pool.utts[i].labels]
                for t in greedy_ctc(lp[r], self.csr.blank_id, m4[r]):
                    if t["token"] in ref:              # a peak that names a true sign
                        confs.append(t["confidence"])
        if not confs:
            return self.seq_threshold
        self.seq_threshold = float(np.quantile(confs, 1.0 - target_accept))
        return self.seq_threshold


# ------------------------------------------------------- attribution -------
@dataclass
class Attribution:
    """Why did the reader say `_`? The question v1.2 could not answer."""
    sentence: str
    tokens: list
    content: list
    in_vocab: list
    out_of_vocab: list
    n_signs_seen: int
    n_named: int
    n_rejected: int
    reasons: dict = field(default_factory=dict)
    decidable: bool = False

    @property
    def count_error(self) -> int:
        """Signs localised minus content words expected. Measurable with no
        lexicon at all, which is the point: the filename gives a sentence, and a
        sentence gives an expected sign count even when it gives no glosses."""
        return self.n_signs_seen - len(self.content)

    def verdict(self) -> str:
        count = (f"{self.n_signs_seen} signs localised against "
                 f"{len(self.content)} content word(s) "
                 f"[{', '.join(self.content)}] -> count error {self.count_error:+d}; "
                 f"{self.n_named} named, {self.n_rejected} rejected. ")
        if not self.decidable:
            return count + ("Attribution is UNDECIDABLE: with no lemma table a "
                            "rejection cannot be assigned to the vocabulary or to "
                            "the model.")
        n_out = len(self.out_of_vocab)
        return count + (f"The sentence contains {n_out} word(s) the vocabulary "
                        f"cannot express ({', '.join(self.out_of_vocab) or '-'}), "
                        f"so at least {n_out} rejection(s) are correct behaviour "
                        f"rather than model failure.")


def attribute(result: dict, sentence: str, lex: Lexicon) -> Attribution:
    """Combine a decode with the vocabulary-coverage check.

    A rejection is only a *model* failure when the word was inside the
    vocabulary to begin with. Without this split, every v1.2 number blamed the
    model for signs it was never taught.
    """
    from .lexicon import content_tokens, coverage
    cov = coverage(sentence, lex)
    reasons = {}
    for s in result["segments"]:
        reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
    named = sum(1 for s in result["segments"] if s["is_known"])
    return Attribution(sentence=str(sentence), tokens=cov.tokens,
                       content=content_tokens(sentence),
                       in_vocab=cov.in_vocab, out_of_vocab=cov.out_of_vocab,
                       n_signs_seen=len(result["segments"]), n_named=named,
                       n_rejected=len(result["segments"]) - named, reasons=reasons,
                       decidable=cov.lexicon_filled > 0)
