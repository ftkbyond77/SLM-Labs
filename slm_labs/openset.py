"""Open-set extraction: motion-based segmentation + CTC + prototype/embedding → known word | '_' (unknown) ต่อ segment

แนวคิด
  1. Segment ด้วย hand-activity (ไม่พึ่ง vocab)  → รู้ว่า "มีกี่คำ" แม้ไม่รู้จักคำ
  2. Global view : forward ทั้งคลิป → CTC tokens + frame embeddings (บริบททั้งประโยค)
  3. Local view  : forward เฉพาะ segment → classifier + embedding (เหมือน isolated sign)
  4. ตัดสินใจต่อ segment ด้วย (CTC conf) ∧ (cosine กับ class prototype) ; ไม่ผ่าน → '_' + เก็บ embedding/metadata
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import cfg, DEVICE, to_dev
from .vocab import SIGN_CLASSES, CLASS2WORD, NULL_CLASS, CLS2ID, N_CLASSES, UNK
from .features import hand_activity, crop, temporal_resample
from .data import pad_feats
from .metrics import ctc_decode_with_confidence

NULL_ID = CLS2ID[NULL_CLASS]


# ---------------- prototypes ----------------

class Prototypes:
    def __init__(self, protos: np.ndarray, sim_thr: float, stats: dict | None = None):
        self.P = protos.astype(np.float32)                 # (C,E) L2-normalised
        self.sim_thr = float(sim_thr)
        self.stats = stats or {}

    @torch.no_grad()
    def _embed_items(model, items, max_frames=cfg.MAX_FRAMES_ISO, bs=64):
        model.eval(); embs, probs = [], []
        for i in range(0, len(items), bs):
            feats = [it["feat"] if len(it["feat"]["hand"]) <= max_frames else temporal_resample(it["feat"], max_frames) for it in items[i:i + bs]]
            o = model.forward_batch(to_dev(pad_feats(feats)))
            embs.append(o["emb"].cpu().numpy()); probs.append(o["logits_cls"].softmax(-1).cpu().numpy())
        return np.concatenate(embs), np.concatenate(probs)

    @classmethod
    def build(cls, model, iso_train, iso_val, quantile=cfg.PROTO_SIM_QUANTILE):
        E_tr, _ = cls._embed_items(model, iso_train); y_tr = np.array([it["label"] for it in iso_train])
        P = np.zeros((N_CLASSES, E_tr.shape[1]), np.float32)
        for c in range(N_CLASSES):
            if (y_tr == c).any():
                v = E_tr[y_tr == c].mean(0); P[c] = v / (np.linalg.norm(v) + 1e-8)
        E_va, pr_va = cls._embed_items(model, iso_val); y_va = np.array([it["label"] for it in iso_val])
        sims = E_va @ P.T
        correct = sims[np.arange(len(y_va)), y_va]
        sim_thr = float(np.quantile(correct, quantile))
        stats = dict(val_correct_sim_mean=float(correct.mean()), val_correct_sim_std=float(correct.std()),
                     val_nn_acc=float((sims.argmax(1) == y_va).mean()), val_max_sim_mean=float(sims.max(1).mean()),
                     val_cls_conf_mean=float(pr_va.max(1).mean()), n_train=int(len(y_tr)), n_val=int(len(y_va)))
        return cls(P, sim_thr, stats)

    def sims(self, emb: np.ndarray) -> np.ndarray:
        return self.P @ emb

    def nearest(self, emb: np.ndarray, exclude_null=True, k=3):
        s = self.sims(emb).copy()
        if exclude_null:
            s[NULL_ID] = -1
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

    def save(self, path):
        np.savez(path, P=self.P, sim_thr=self.sim_thr, stats=str(self.stats))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        import ast
        return cls(z["P"], float(z["sim_thr"]), ast.literal_eval(str(z["stats"])))


# ---------------- segmentation ----------------

SEG_PARAMS = dict(min_frames=cfg.SEG_MIN_FRAMES, min_dist=cfg.SEG_MIN_DIST, prominence=cfg.SEG_PROMINENCE, energy_frac=cfg.SEG_ENERGY_FRAC,
                  wrist_y=cfg.SEG_ACTIVE_WRIST_Y, smooth=3)


def segment_timeline(feat: dict, act: dict | None = None, **over) -> list[tuple[int, int]]:
    """→ list of (start, end) frame spans ของช่วงที่ "กำลังทำท่า" แต่ละคำ (ตัดที่จุดที่ความเร็วมือต่ำสุด)
    params (SEG_PARAMS / override): min_frames, min_dist, prominence, energy_frac, wrist_y, smooth"""
    from scipy.signal import find_peaks
    P = {**SEG_PARAMS, **over}
    for k in ("min_frames", "min_dist", "smooth"):
        P[k] = int(round(float(P[k])))
    act = act or hand_activity(feat, smooth=P["smooth"])
    T = len(act["energy"]); energy = act["energy"]
    if P["smooth"] != 3 and "energy_raw" not in act:      # re-smooth ถ้าขอ smooth ต่างจาก default
        energy = hand_activity(feat, smooth=P["smooth"])["energy"]
    wrist_y = np.nanmin(act["wrist_y"], axis=1)
    raised = wrist_y < P["wrist_y"]
    ref = np.percentile(energy[raised], 90) if raised.any() else np.percentile(energy, 90)
    e_thr = P["energy_frac"] * max(ref, 1e-4)
    active = raised | ((act["present"] > 0) & (energy > e_thr))
    runs, i = [], 0
    while i < T:
        if active[i]:
            j = i
            while j < T and active[j]:
                j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 3:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= P["min_frames"]]
    segs = []
    for s, e in runs:
        seg_e = energy[s:e]
        if e - s < 2 * P["min_frames"]:
            segs.append((s, e)); continue
        valleys, _ = find_peaks(-seg_e, distance=P["min_dist"], prominence=P["prominence"] * (seg_e.max() - seg_e.min() + 1e-6))
        cuts = [s] + [s + int(v) for v in valleys if P["min_frames"] <= v <= (e - s) - P["min_frames"]] + [e]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if segs and a == segs[-1][1] and (b - a) < P["min_frames"]:
                segs[-1] = (segs[-1][0], b)
            else:
                segs.append((a, b))
    return segs


def tune_segmentation(items, grid=None, set_global=True):
    """grid-search พารามิเตอร์ segmentation ให้ #segments ≈ #glosses บน sentence items (ไม่ใช้ vocab/model)
    → DataFrame ผลทุก config (เรียงตาม mae) ; set_global=True → เขียนค่าที่ดีที่สุดลง SEG_PARAMS"""
    import itertools, pandas as pd
    grid = grid or dict(min_dist=[5, 8, 10, 12], prominence=[0.15, 0.3, 0.45], energy_frac=[0.25, 0.4], min_frames=[4, 6], smooth=[3, 5])
    n_true = np.array([len([g for g in it["glosses"] if g != NULL_CLASS]) for it in items])
    acts = {sm: [hand_activity(it["feat"], smooth=sm) for it in items] for sm in grid.get("smooth", [3])}
    rows = []
    for vals in itertools.product(*grid.values()):
        P = dict(zip(grid.keys(), vals))
        n_pred = np.array([len(segment_timeline(it["feat"], act=a, **P)) for it, a in zip(items, acts[P.get("smooth", 3)])])
        d = n_pred - n_true
        rows.append(dict(**P, mae=float(np.abs(d).mean()), bias=float(d.mean()), exact=float((d == 0).mean()), within1=float((np.abs(d) <= 1).mean())))
    df = pd.DataFrame(rows).sort_values(["mae", "within1"], ascending=[True, False]).reset_index(drop=True)
    if set_global:
        SEG_PARAMS.update({k: (int(v) if k in ("min_dist", "min_frames", "smooth") else float(v)) for k, v in df.iloc[0][list(grid.keys())].items()})
    return df


def _assign_tokens(segs, toks, T):
    """map CTC tokens → segments (ตาม frame กึ่งกลางของ emission) ; token นอก segment → สร้าง segment ใหม่ ; หลาย token ใน segment → แบ่ง"""
    segs = [list(s) for s in segs]
    owner = {}
    extra = []
    for ti, tk in enumerate(toks):
        c = (tk["start"] + tk["end"]) / 2
        best, bd = None, 1e9
        for si, (s, e) in enumerate(segs):
            d = 0 if s <= c < e else min(abs(c - s), abs(c - e))
            if d < bd:
                best, bd = si, d
        if best is not None and bd <= cfg.SEG_MIN_DIST:
            owner.setdefault(best, []).append(ti)
        else:
            s0 = max(0, int(c) - cfg.SEG_MIN_FRAMES); e0 = min(T, int(c) + cfg.SEG_MIN_FRAMES)
            extra.append(([s0, e0], [ti]))
    out = []
    for si, seg in enumerate(segs):
        tis = owner.get(si, [])
        if len({toks[t]["cls"] for t in tis}) <= 1:
            out.append((seg, tis)); continue
        # split at midpoints between consecutive distinct-class token centres
        groups, prev_cls = [], None
        for t in tis:
            if toks[t]["cls"] == prev_cls:
                groups[-1].append(t)
            else:
                groups.append([t]); prev_cls = toks[t]["cls"]
        centres = [np.mean([(toks[t]["start"] + toks[t]["end"]) / 2 for t in g]) for g in groups]
        cuts = [seg[0]] + [int((a + b) / 2) for a, b in zip(centres[:-1], centres[1:])] + [seg[1]]
        for g, a, b in zip(groups, cuts[:-1], cuts[1:]):
            out.append(([a, max(b, a + 1)], g))
    out += extra
    out.sort(key=lambda z: z[0][0])
    return out


# ---------------- analysis ----------------

@torch.no_grad()
def analyze_clip(model, feat: dict, protos: Prototypes, memory=None, t_ms: np.ndarray | None = None,
                 merge_sim: float = 0.97, max_merge_frames: int = 25) -> dict:
    """feat (ทั้งคลิป) → slots: [{start,end,word|'_',status,conf,sim,emb,...}]"""
    model.eval()
    T0 = len(feat["hand"])
    if T0 > cfg.MAX_FRAMES_SEQ:
        feat = temporal_resample(feat, cfg.MAX_FRAMES_SEQ)
    T = len(feat["hand"])
    idx_map = np.linspace(0, T0 - 1, T).round().astype(int)
    if t_ms is None:
        t_ms = np.arange(T0) * (1000.0 / cfg.TARGET_FPS)

    x = to_dev(pad_feats([feat])); o = model.forward_batch(x)
    toks = ctc_decode_with_confidence(o["logits_ctc"], [T])[0]
    act = hand_activity(feat); segs = segment_timeline(feat, act)
    if not segs:
        segs = [(0, T)]
    assigned = _assign_tokens(segs, toks, T)

    # local views (batch)
    def _local(assigned_):
        crops = [crop(feat, s, e) for (s, e), _ in assigned_]
        ol = model.forward_batch(to_dev(pad_feats(crops)))
        return ol["emb"].cpu().numpy(), ol["logits_cls"].softmax(-1).cpu().numpy()

    loc_emb, loc_prob = _local(assigned)
    # merge: segment ติดกันที่ไม่มี CTC token ทั้งคู่ + local embedding เกือบเหมือนกัน (ท่าเดียวกันที่มี "hold" ตรงกลาง) → รวม
    merged, i = [], 0
    while i < len(assigned):
        (s, e), tis = assigned[i]; j = i
        while (j + 1 < len(assigned) and not tis and not assigned[j + 1][1] and assigned[j + 1][0][0] - e <= 1
               and float(loc_emb[i] @ loc_emb[j + 1]) >= merge_sim and (assigned[j + 1][0][1] - s) <= max_merge_frames):
            j += 1; e = assigned[j][0][1]
        merged.append(([s, e], tis)); i = j + 1
    if len(merged) != len(assigned):
        assigned = merged; loc_emb, loc_prob = _local(assigned)
    frame_emb = o["frame_emb"]

    slots = []
    for k, ((s, e), tis) in enumerate(assigned):
        g_emb = model.segment_embed(frame_emb, s, e)[0].cpu().numpy()
        emb = loc_emb[k] + g_emb; emb = emb / (np.linalg.norm(emb) + 1e-8)
        nn3 = protos.nearest(emb, k=3)
        tok = max((toks[t] for t in tis), key=lambda z: z["conf"]) if tis else None
        loc_cls, loc_conf = int(loc_prob[k].argmax()), float(loc_prob[k].max())
        null_sim = float(protos.sims(emb)[NULL_ID])
        slot = dict(idx=k, start=int(idx_map[s]), end=int(idx_map[min(e, T - 1)]) + 1,
                    t_start_ms=float(t_ms[idx_map[s]]), t_end_ms=float(t_ms[idx_map[min(e, T - 1)]]),
                    n_frames=int(e - s), emb=emb.astype(np.float32), word=UNK, status="unknown", source="none",
                    conf=0.0, sim=float(nn3[0][1]), nearest=[(CLASS2WORD[SIGN_CLASSES[i]], round(sm, 3)) for i, sm in nn3],
                    ctc=(CLASS2WORD[SIGN_CLASSES[tok["cls"]]], round(tok["conf"], 3)) if tok else None,
                    local=(CLASS2WORD[SIGN_CLASSES[loc_cls]], round(loc_conf, 3)), null_sim=round(null_sim, 3), memory=None)
        # --- decision
        if tok is not None and tok["conf"] >= cfg.MIN_TOKEN_CONF and float(protos.sims(emb)[tok["cls"]]) >= protos.sim_thr:
            slot.update(word=CLASS2WORD[SIGN_CLASSES[tok["cls"]]], status="known", source="ctc", conf=tok["conf"],
                        sim=float(protos.sims(emb)[tok["cls"]]), cls=tok["cls"])
        elif loc_cls != NULL_ID and loc_conf >= cfg.LOCAL_RESCUE_CLS_CONF and nn3[0][0] == loc_cls and nn3[0][1] >= protos.sim_thr:
            slot.update(word=CLASS2WORD[SIGN_CLASSES[loc_cls]], status="known", source="local", conf=loc_conf, sim=nn3[0][1], cls=loc_cls)
        elif null_sim >= protos.sim_thr and null_sim >= nn3[0][1] and (tok is None):
            slot.update(status="null", source="proto")            # ท่าพัก / transition → ไม่ใช่คำ
        else:
            if memory is not None:
                m = memory.nearest_learned(emb)
                slot["memory"] = m
                if m and m["sim"] >= cfg.MEMORY_SIM_THRESHOLD:
                    slot.update(word=m["word"], status="learned", source="memory", conf=m["sim"], sim=m["sim"])
        slots.append(slot)

    return dict(slots=slots, segments=[(int(idx_map[s]), int(idx_map[min(e, T - 1)]) + 1) for (s, e), _ in assigned],
                tokens=toks, act=act, T=T0, idx_map=idx_map, face_emb=o["face_emb"][0].cpu().numpy(),
                clip_emb=o["emb"][0].cpu().numpy(), energy=act["energy"])


def slots_to_words(slots, include_null=False):
    return [s["word"] for s in slots if s["status"] != "null" or include_null]


# ---------------- evaluation helpers (open-set) ----------------

@torch.no_grad()
def known_unknown_scores(model, protos, known_items, unknown_feats):
    """score = max cosine กับ prototype (ยกเว้น null) ; known_items = isolated test ; unknown_feats = list of feat (คำนอก vocab)"""
    E_k, _ = Prototypes._embed_items(model, known_items)
    ks = [protos.nearest(e, k=1)[0][1] for e in E_k]
    us = []
    if unknown_feats:
        E_u, _ = Prototypes._embed_items(model, [dict(feat=f) for f in unknown_feats])
        us = [protos.nearest(e, k=1)[0][1] for e in E_u]
    return np.array(ks), np.array(us)


@torch.no_grad()
def eval_segment_count(model, protos, items, use_model=True):
    """วัดว่า segmentation นับ "จำนวนคำ" ได้ตรงกับจำนวน gloss จริงแค่ไหน (sentence items) → dict(mae, exact, rows)
    use_model=False → motion-only segments ; True → หลัง analyze_clip (CTC assignment + merge + null filter)"""
    rows = []
    for it in items:
        n_true = len([g for g in it["glosses"] if g != NULL_CLASS])
        if use_model:
            a = analyze_clip(model, it["feat"], protos)
            n_pred = len([s for s in a["slots"] if s["status"] != "null"])
        else:
            n_pred = len(segment_timeline(it["feat"]))
        rows.append(dict(id=it["id"], n_true=n_true, n_pred=n_pred))
    d = np.array([r["n_pred"] - r["n_true"] for r in rows])
    return dict(mae=float(np.abs(d).mean()), bias=float(d.mean()), exact=float((d == 0).mean()), within1=float((np.abs(d) <= 1).mean()), rows=rows)
