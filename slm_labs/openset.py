"""Open-vocabulary extraction: motion segmentation → per-segment classify + prototype → known word | '_' (unknown)

v3 ทำให้เรียบง่ายลง (v1/v2 เอา CTC token มา assign เข้า segment แล้วแตก/รวม segment ซ้ำอีกชั้น ซึ่งพังทั้งคู่):

  1. `segment_timeline()`  ตัดคลิปเป็นช่วง "กำลังทำท่า" จาก hand-activity (ไม่ใช้ vocab เลย) → "มีกี่คำ"
  2. แต่ละ segment ถูก forward เหมือน isolated clip → cls prob + emb (128-d, L2, ฝึกด้วย ArcFace)
  3. ตัดสิน: known / learned (จาก memory) / null (ท่าพัก) / '_' (unknown → เก็บ embedding + metadata)

Stage B แบบ CTC ยังอยู่ (metrics.eval_sequence) แต่เป็น "ทางเลือกเชิงเปรียบเทียบ" ไม่ใช่ทางหลักของ pipeline
"""
from __future__ import annotations

import numpy as np
import torch

from .config import cfg, DEVICE, to_dev
from .vocab import SIGN_CLASSES, CLASS2WORD, NULL_CLASS, CLS2ID, N_CLASSES, UNK
from .features import hand_activity, crop, temporal_resample
from .data import pad_feats

NULL_ID = CLS2ID[NULL_CLASS]


# ---------------- prototypes ----------------

class Prototypes:
    """prototype ต่อ class = mean ของ embedding บน train + threshold cosine ที่ calibrate บน val"""

    def __init__(self, protos: np.ndarray, sim_thr: float, stats: dict | None = None):
        self.P = protos.astype(np.float32)                 # (C,E) L2-normalised
        self.sim_thr = float(sim_thr)
        self.stats = stats or {}

    @staticmethod
    @torch.no_grad()
    def embed_items(model, items, max_frames=cfg.MAX_FRAMES_ISO, bs=64):
        model.eval(); embs, probs = [], []
        for i in range(0, len(items), bs):
            feats = [it["feat"] if len(it["feat"]["hand"]) <= max_frames else temporal_resample(it["feat"], max_frames)
                     for it in items[i:i + bs]]
            o = model.forward_batch(to_dev(pad_feats(feats)))
            embs.append(o["emb"].cpu().numpy()); probs.append(o["logits_cls"].softmax(-1).cpu().numpy())
        return np.concatenate(embs), np.concatenate(probs)

    @classmethod
    def build(cls, model, iso_train, iso_val, quantile=cfg.PROTO_SIM_QUANTILE):
        E_tr, _ = cls.embed_items(model, iso_train); y_tr = np.array([it["label"] for it in iso_train])
        P = np.zeros((N_CLASSES, E_tr.shape[1]), np.float32)
        for c in range(N_CLASSES):
            if (y_tr == c).any():
                v = E_tr[y_tr == c].mean(0); P[c] = v / (np.linalg.norm(v) + 1e-8)
        E_va, pr_va = cls.embed_items(model, iso_val); y_va = np.array([it["label"] for it in iso_val])
        sims = E_va @ P.T
        correct = sims[np.arange(len(y_va)), y_va]
        sim_thr = float(np.quantile(correct, quantile))
        stats = dict(val_correct_sim_mean=float(correct.mean()), val_correct_sim_std=float(correct.std()),
                     val_nn_acc=float((sims.argmax(1) == y_va).mean()), val_max_sim_mean=float(sims.max(1).mean()),
                     val_wrong_sim_mean=float(np.mean([np.delete(s, y).max() for s, y in zip(sims, y_va)])),
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
        import ast
        z = np.load(path, allow_pickle=True)
        return cls(z["P"], float(z["sim_thr"]), ast.literal_eval(str(z["stats"])))


# ---------------- segmentation ----------------

SEG_PARAMS = dict(min_frames=cfg.SEG_MIN_FRAMES, min_dist=cfg.SEG_MIN_DIST, prominence=cfg.SEG_PROMINENCE,
                  energy_frac=cfg.SEG_ENERGY_FRAC, wrist_y=cfg.SEG_ACTIVE_WRIST_Y, smooth=3)


def segment_timeline(feat: dict, act: dict | None = None, **over) -> list[tuple[int, int]]:
    """→ list of (start, end) frame spans ของช่วงที่ "กำลังทำท่า" (ตัดที่จุดที่ความเร็วมือต่ำสุด)"""
    from scipy.signal import find_peaks
    P = {**SEG_PARAMS, **over}
    for k in ("min_frames", "min_dist", "smooth"):
        P[k] = int(round(float(P[k])))
    act = act or hand_activity(feat, smooth=P["smooth"])
    T = len(act["energy"]); energy = act["energy"]
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
    """grid-search ให้ #segments ≈ #glosses บน **sentence-train** (ใช้แค่จำนวนคำ ไม่ใช้ว่าคำอะไร)"""
    import itertools, pandas as pd
    grid = grid or dict(min_dist=[8, 12, 16, 20], prominence=[0.15, 0.3, 0.45], energy_frac=[0.25, 0.4, 0.55],
                        min_frames=[6, 8, 10], smooth=[3, 5])
    n_true = np.array([len([g for g in it["glosses"] if g != NULL_CLASS]) for it in items])
    acts = {sm: [hand_activity(it["feat"], smooth=sm) for it in items] for sm in grid.get("smooth", [3])}
    rows = []
    for vals in itertools.product(*grid.values()):
        P = dict(zip(grid.keys(), vals))
        n_pred = np.array([len(segment_timeline(it["feat"], act=a, **P)) for it, a in zip(items, acts[P.get("smooth", 3)])])
        d = n_pred - n_true
        rows.append(dict(**P, mae=float(np.abs(d).mean()), bias=float(d.mean()), exact=float((d == 0).mean()),
                         within1=float((np.abs(d) <= 1).mean())))
    df = pd.DataFrame(rows).sort_values(["mae", "within1"], ascending=[True, False]).reset_index(drop=True)
    if set_global:
        SEG_PARAMS.update({k: (int(v) if k in ("min_dist", "min_frames", "smooth") else float(v))
                           for k, v in df.iloc[0][list(grid.keys())].items()})
    return df


@torch.no_grad()
def ctc_boundaries(model, feat: dict, min_run: int = 1) -> list[int]:
    """ใช้ CTC frame-posterior เป็น "ตัวเสนอขอบเขตคำ" (ไม่ใช่ตัว decode)

    diagnostic ของ v1/v2 แสดงว่า CTC head จัดลำดับคำได้ *ถูกต้องตามเวลา* แม้ P(blank)≈0.97 จะทำให้
    greedy decode ออกมาว่าง — คือ alignment ดีแต่ calibration พัง เลยเอาเฉพาะส่วนที่มันเก่ง (ตำแหน่ง)
    มาช่วยตัด segment ที่ motion-energy รวมกันเกินไป (คลิปจริงเร็วกว่า dataset ~2 เท่า
    พารามิเตอร์ min_dist ที่ tune จาก dataset จึงตัดไม่พอ)

    → คืน index ของเฟรม (สเกลเดิมของ feat) ที่คลาสเด่นที่สุดของ CTC เปลี่ยน
    """
    T0 = len(feat["hand"])
    f = feat if T0 <= cfg.MAX_FRAMES_SEQ else temporal_resample(feat, cfg.MAX_FRAMES_SEQ)
    T = len(f["hand"])
    x = to_dev(pad_feats([f]))
    o = model.forward_batch(x)
    p = o["logits_ctc"][0].float().softmax(-1).cpu().numpy()
    best = p[:, 1:].argmax(1)                                    # คลาส (ไม่นับ blank) ที่เด่นที่สุดต่อเฟรม-หลัง-stride
    cuts, run, prev = [], 1, best[0]
    for i in range(1, len(best)):
        if best[i] != prev:
            if run >= min_run:
                cuts.append(i)
            prev, run = best[i], 1
        else:
            run += 1
    scale = (T0 - 1) / max(T - 1, 1)
    return sorted({int(round(c * model.ctc_stride * scale)) for c in cuts if 0 < c * model.ctc_stride < T})


def split_segments(segs, cuts, min_frames: int, max_frames: int):
    """แตก segment ที่ยาวเกิน max_frames ที่จุด cut ที่อยู่ข้างใน (เว้นระยะ min_frames จากขอบ)"""
    out = []
    for s, e in segs:
        if e - s <= max_frames:
            out.append((s, e)); continue
        inner = [c for c in cuts if s + min_frames <= c <= e - min_frames]
        pts = [s] + inner + [e]
        for a, b in zip(pts[:-1], pts[1:]):
            if out and a == out[-1][1] and b - a < min_frames:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


# ---------------- analysis ----------------

@torch.no_grad()
def analyze_clip(model, feat: dict, protos: Prototypes, memory=None, t_ms: np.ndarray | None = None,
                 merge_sim: float = 0.95, use_ctc_cuts: bool | None = None, max_seg_frames: int | None = None) -> dict:
    """feat (ทั้งคลิป) → slots: [{start,end,word|'_',status,conf,sim,emb,...}]

    หมายเหตุสำคัญ: segmentation ทำบน **ความละเอียดเวลาเดิม** ไม่ resample ทั้งคลิปก่อน
    (v1/v2 บีบทั้งคลิปลงเหลือ MAX_FRAMES_SEQ ก่อน → คลิปยาว 24 s ถูกเร่งเป็น 1.9 เท่า แล้วค่อยตัด segment
    ทำให้ segment ที่ป้อนเข้า classifier เร็วกว่าคลิป isolated ที่ใช้ train)  แต่ละ segment ถูก cap
    ที่ MAX_FRAMES_ISO ทีละอันแทน
    """
    model.eval()
    T = T0 = len(feat["hand"])
    idx_map = np.arange(T)
    if t_ms is None:
        t_ms = np.arange(T0) * (1000.0 / cfg.TARGET_FPS)

    act = hand_activity(feat)
    segs = segment_timeline(feat, act) or [(0, T)]
    cuts = []
    if cfg.USE_CTC_CUTS if use_ctc_cuts is None else use_ctc_cuts:
        cuts = ctc_boundaries(model, feat)
        segs = split_segments(segs, cuts, int(SEG_PARAMS["min_frames"]), max_seg_frames or cfg.SEG_MAX_FRAMES)

    def _forward(spans):
        crops = [crop(feat, s, e) for s, e in spans]
        crops = [c if len(c["hand"]) <= cfg.MAX_FRAMES_ISO else temporal_resample(c, cfg.MAX_FRAMES_ISO) for c in crops]
        o = model.forward_batch(to_dev(pad_feats(crops)))
        return o["emb"].cpu().numpy(), o["logits_cls"].softmax(-1).cpu().numpy()

    emb_s, prob_s = _forward(segs)
    # รวม segment ที่ติดกันและ "เป็นท่าเดียวกัน" (embedding เกือบเท่ากัน) — แก้ over-segmentation จาก hold กลางคำ
    merged, i = [], 0
    while i < len(segs):
        s, e = segs[i]; j = i
        while (j + 1 < len(segs) and segs[j + 1][0] - e <= 2 and float(emb_s[i] @ emb_s[j + 1]) >= merge_sim):
            j += 1; e = segs[j][1]
        merged.append((s, e)); i = j + 1
    if len(merged) != len(segs):
        segs = merged; emb_s, prob_s = _forward(segs)

    learned = memory.learned_prototypes() if memory is not None else {}
    slots = []
    for k, (s, e) in enumerate(segs):
        emb = emb_s[k] / (np.linalg.norm(emb_s[k]) + 1e-8)
        p = prob_s[k]
        cls_id, cls_conf = int(p.argmax()), float(p.max())
        sims = protos.sims(emb)
        nn3 = protos.nearest(emb, k=3)
        null_sim = float(sims[NULL_ID])
        # คำที่จะเสนอ = คลาสของ cls head เว้นแต่ prototype-NN มั่นใจกว่าชัดเจน (protoNN แม่นพอ ๆ กับ cls head
        # แต่เป็นคนละ view ของโมเดล → รวมกันแล้วดีกว่าใช้ตัวใดตัวหนึ่ง)
        cand = cls_id if float(sims[cls_id]) >= nn3[0][1] - 0.05 else nn3[0][0]
        cand_sim = float(sims[cand]); cand_conf = float(p[cand])
        slot = dict(idx=k, start=int(idx_map[s]), end=int(idx_map[min(e, T - 1)]) + 1,
                    t_start_ms=float(t_ms[idx_map[s]]), t_end_ms=float(t_ms[idx_map[min(e, T - 1)]]),
                    n_frames=int(e - s), emb=emb.astype(np.float32), word=UNK, status="unknown", source="none",
                    conf=round(cand_conf, 4), sim=float(nn3[0][1]),
                    nearest=[(CLASS2WORD[SIGN_CLASSES[i]], round(sm, 3)) for i, sm in nn3],
                    cls=(CLASS2WORD[SIGN_CLASSES[cls_id]], round(cls_conf, 3)),
                    null_sim=round(null_sim, 3), memory=None)

        if cand == NULL_ID or (null_sim >= protos.sim_thr and null_sim >= nn3[0][1]):
            slot.update(status="null", source="proto")                 # ท่าพัก / transition → ไม่ใช่คำ
        elif cand_sim >= protos.sim_thr and cand_conf >= cfg.SEG_CLS_CONF:
            slot.update(word=CLASS2WORD[SIGN_CLASSES[cand]], status="known", source="cls+proto",
                        sim=cand_sim, cls_id=int(cand))
        else:
            m = None
            if learned:
                best = max(learned.items(), key=lambda kv: float(kv[1]["vec"] @ emb))
                m = dict(word=best[0], sim=float(best[1]["vec"] @ emb), n=best[1]["n"])
                slot["memory"] = m
            if m and m["sim"] >= cfg.MEMORY_SIM_THRESHOLD:
                slot.update(word=m["word"], status="learned", source="memory", conf=round(m["sim"], 4), sim=m["sim"])
        slots.append(slot)

    return dict(slots=slots, segments=[(int(idx_map[s]), int(idx_map[min(e, T - 1)]) + 1) for s, e in segs],
                tokens=[], ctc_cuts=cuts, act=act, T=T0, idx_map=idx_map, energy=act["energy"])


def slots_to_words(slots, include_null=False):
    return [s["word"] for s in slots if s["status"] != "null" or include_null]


# ---------------- evaluation ----------------

@torch.no_grad()
def eval_sequence_segments(model, protos, items, memory=None, keep_unknown=False, use_ctc_cuts=None):
    """**Stage B ทางหลักของ v3**: segment → classify → gloss sequence → WER/CER เทียบ gloss จริง
    (keep_unknown=False → ตัด '_' ออกก่อนวัด เพื่อเทียบกับ CTC ที่ไม่มีแนวคิด unknown อย่างยุติธรรม)"""
    from .metrics import wer_cer
    from .vocab import CLASS2WORD as C2W
    refs, hyps = [], []
    for it in items:
        a = analyze_clip(model, it["feat"], protos, memory=memory, use_ctc_cuts=use_ctc_cuts)
        w = [s["word"] for s in a["slots"] if s["status"] != "null"]
        if not keep_unknown:
            w = [x for x in w if x != UNK]
        hyps.append(w)
        refs.append([C2W[g] for g in it["glosses"] if g != NULL_CLASS])
    wer, cer = wer_cer(refs, hyps)
    return dict(wer=wer, cer=cer, sent_acc=float(np.mean([r == h for r, h in zip(refs, hyps)])), refs=refs, hyps=hyps)


def tune_openset_thresholds(model, protos, sent_val, conf_grid=(0.0, 0.1, 0.2, 0.3, 0.4),
                            sim_grid=(0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75), set_global=True, use_ctc_cuts=None):
    """เลือก (SEG_CLS_CONF, sim_thr) ที่ให้ WER ต่ำสุดบน **sentence-val** (ไม่แตะ test)

    threshold ที่ calibrate จาก isolated-val อย่างเดียว (PROTO_SIM_QUANTILE) เข้มเกินไปสำหรับ segment
    ที่ตัดมาจากประโยคจริง — segment มี co-articulation และขอบเขตไม่คม
    """
    import pandas as pd
    base_conf, base_thr = cfg.SEG_CLS_CONF, protos.sim_thr
    rows = []
    for c in conf_grid:
        for s in sim_grid:
            cfg.SEG_CLS_CONF, protos.sim_thr = float(c), float(s)
            r = eval_sequence_segments(model, protos, sent_val, use_ctc_cuts=use_ctc_cuts)
            rk = eval_sequence_segments(model, protos, sent_val, keep_unknown=True, use_ctc_cuts=use_ctc_cuts)
            n_unk = sum(w == UNK for h in rk["hyps"] for w in h)
            rows.append(dict(seg_cls_conf=c, sim_thr=s, wer=r["wer"], cer=r["cer"], sent_acc=r["sent_acc"], n_unknown=n_unk))
    df = pd.DataFrame(rows).sort_values(["wer", "cer"]).reset_index(drop=True)
    if set_global:
        cfg.SEG_CLS_CONF = float(df.iloc[0]["seg_cls_conf"]); protos.sim_thr = float(df.iloc[0]["sim_thr"])
    else:
        cfg.SEG_CLS_CONF, protos.sim_thr = base_conf, base_thr
    return df


@torch.no_grad()
def known_unknown_scores(model, protos, known_items, unknown_items):
    """score = max cosine กับ prototype (ยกเว้น null) — สูง = น่าจะเป็นคำที่รู้จัก"""
    E_k, _ = Prototypes.embed_items(model, known_items)
    ks = [protos.nearest(e, k=1)[0][1] for e in E_k]
    us = []
    if unknown_items:
        items = unknown_items if isinstance(unknown_items[0], dict) and "feat" in unknown_items[0] else [dict(feat=f) for f in unknown_items]
        E_u, _ = Prototypes.embed_items(model, items)
        us = [protos.nearest(e, k=1)[0][1] for e in E_u]
    return np.array(ks), np.array(us)


@torch.no_grad()
def eval_segment_count(model, protos, items, use_model=True, use_ctc_cuts=None):
    """segmentation นับ "จำนวนคำ" ตรงกับจำนวน gloss จริงแค่ไหน"""
    rows = []
    for it in items:
        n_true = len([g for g in it["glosses"] if g != NULL_CLASS])
        if use_model:
            a = analyze_clip(model, it["feat"], protos, use_ctc_cuts=use_ctc_cuts)
            n_pred = len([s for s in a["slots"] if s["status"] != "null"])
        else:
            n_pred = len(segment_timeline(it["feat"]))
        rows.append(dict(id=it["id"], n_true=n_true, n_pred=n_pred))
    d = np.array([r["n_pred"] - r["n_true"] for r in rows])
    return dict(mae=float(np.abs(d).mean()), bias=float(d.mean()), exact=float((d == 0).mean()),
                within1=float((np.abs(d) <= 1).mean()), rows=rows)
