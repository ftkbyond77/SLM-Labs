"""Metrics: CTC decode (+confidence, +emission frames), WER/CER, BLEU/chrF, eval loops, open-set AUROC"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import DEVICE
from .vocab import SIGN_CLASSES, NULL_CLASS, BLANK, ids_to_words


from .config import cfg


def _penalise(logits_ctc, blank_penalty):
    bp = cfg.BLANK_PENALTY if blank_penalty is None else blank_penalty
    if bp:
        logits_ctc = logits_ctc.clone(); logits_ctc[..., BLANK] -= bp
    return logits_ctc


def ctc_greedy_decode(logits_ctc, lens, drop_null=True, blank_penalty=None):
    """lens = ความยาวหลัง stride (model.ctc_lens) ; blank_penalty ลบจาก blank logit (None = cfg.BLANK_PENALTY)"""
    pred = _penalise(logits_ctc, blank_penalty).argmax(-1).cpu().numpy(); out = []
    for p, L in zip(pred, lens):
        seq, prev = [], BLANK
        for t in p[:L]:
            if t != BLANK and t != prev:
                seq.append(int(t) - 1)
            prev = t
        if drop_null:
            seq = [c for c in seq if SIGN_CLASSES[c] != NULL_CLASS]
        out.append(seq)
    return out


@torch.no_grad()
def ctc_decode_with_confidence(logits_ctc, lens, drop_null=True, blank_penalty=None, stride=1):
    """greedy CTC → list[dict(cls, conf, start, end)] ต่อคลิป ; conf = mean prob (หลัง penalty) ของเฟรมที่ emit token ; start/end คูณ stride กลับเป็นเฟรมจริง"""
    probs = _penalise(logits_ctc, blank_penalty).float().softmax(-1).cpu().numpy(); out = []
    for p, L in zip(probs, lens):
        am = p[:L].argmax(-1); toks, prev, cur = [], BLANK, None
        for t, tok in enumerate(am):
            if tok != BLANK and tok != prev:
                if cur: toks.append(cur)
                cur = dict(cls=int(tok) - 1, probs=[], start=t, end=t + 1)
            if tok != BLANK and cur is not None:
                cur["probs"].append(float(p[t, tok])); cur["end"] = t + 1
            prev = tok
        if cur: toks.append(cur)
        res = []
        for tk in toks:
            if drop_null and SIGN_CLASSES[tk["cls"]] == NULL_CLASS:
                continue
            res.append(dict(cls=tk["cls"], conf=float(np.mean(tk["probs"])), start=tk["start"] * stride, end=tk["end"] * stride))
        out.append(res)
    return out


@torch.no_grad()
def tune_blank_penalty(model, dl, grid=(0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6), set_global=True):
    """เลือก blank penalty ที่ให้ val WER ต่ำสุด → เขียนลง cfg.BLANK_PENALTY"""
    import pandas as pd
    rows = [dict(blank_penalty=bp, **{k: v for k, v in eval_sequence(model, dl, blank_penalty=bp).items() if k in ("wer", "cer", "sent_acc")}) for bp in grid]
    df = pd.DataFrame(rows).sort_values(["wer", "cer"]).reset_index(drop=True)
    if set_global:
        cfg.BLANK_PENALTY = float(df.iloc[0]["blank_penalty"])
    return df


def wer_cer(refs_words, hyps_words):
    import jiwer
    r = [" ".join(x) for x in refs_words]; h = [" ".join(x) if x else "<empty>" for x in hyps_words]
    wer = jiwer.wer(r, h)
    cer = jiwer.cer(["".join(x) for x in refs_words], ["".join(x) if x else "?" for x in hyps_words])
    return wer, cer


def bleu_chrf(refs, hyps):
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return bleu, chrf


@torch.no_grad()
def eval_isolated(model, dl, use_face=True):
    model.eval(); ys, ps, embs = [], [], []
    for x in dl:
        o = model(x["hand"].to(DEVICE), x["body"].to(DEVICE), x["face"].to(DEVICE), x["mask"].to(DEVICE), use_face)
        ps += o["logits_cls"].argmax(-1).cpu().tolist(); ys += x["targets"].tolist(); embs.append(o["emb"].cpu())
    return dict(acc=accuracy_score(ys, ps), macro_f1=f1_score(ys, ps, average="macro"), y=ys, p=ps, emb=torch.cat(embs).numpy())


@torch.no_grad()
def eval_sequence(model, dl, use_face=True, blank_penalty=None):
    model.eval(); refs, hyps = [], []
    for x in dl:
        o = model(x["hand"].to(DEVICE), x["body"].to(DEVICE), x["face"].to(DEVICE), x["mask"].to(DEVICE), use_face)
        hyps += [ids_to_words(s) for s in ctc_greedy_decode(o["logits_ctc"], model.ctc_lens(x["lens"]).tolist(), blank_penalty=blank_penalty)]
        i = 0
        for L in x["target_lens"].tolist():
            refs.append(ids_to_words([t - 1 for t in x["targets"][i:i + L].tolist()])); i += L
    wer, cer = wer_cer(refs, hyps)
    exact = float(np.mean([r == h for r, h in zip(refs, hyps)]))
    return dict(wer=wer, cer=cer, sent_acc=exact, refs=refs, hyps=hyps)


def openset_auroc(known_scores, unknown_scores):
    """score สูง = known ; → AUROC + TNR@95%TPR"""
    y = np.r_[np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    s = np.r_[known_scores, unknown_scores]
    auroc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    thr = np.percentile(known_scores, 5)                     # 95% ของ known ผ่าน
    tnr = float(np.mean(np.asarray(unknown_scores) < thr)) if len(unknown_scores) else float("nan")
    return dict(auroc=float(auroc), tnr_at_95tpr=tnr, thr=float(thr))


@torch.no_grad()
def eval_isolated_shifted(model, items, strength: float, seed: int = 0, max_frames=None, bs=64):
    """จำลอง "วิดีโอถ่ายคนละที่/คนละกล้อง": ใส่ augmentation (หมุน/ย่อขยาย/เลื่อน/mirror/speed) แรง = strength ลงบน test items แล้ววัด accuracy"""
    import random
    from .config import cfg, to_dev
    from .features import augment, temporal_resample
    from .data import pad_feats
    random.seed(seed); np.random.seed(seed); model.eval()
    max_frames = max_frames or cfg.MAX_FRAMES_ISO; ys, ps = [], []
    for i in range(0, len(items), bs):
        feats = []
        for it in items[i:i + bs]:
            f = augment(it["feat"], strength=strength) if strength > 0 else it["feat"]
            feats.append(temporal_resample(f, max_frames) if len(f["hand"]) > max_frames else f); ys.append(it["label"])
        o = model.forward_batch(to_dev(pad_feats(feats))); ps += o["logits_cls"].argmax(-1).cpu().tolist()
    return dict(acc=accuracy_score(ys, ps), macro_f1=f1_score(ys, ps, average="macro"))
