"""Metrics for every stage of the pipeline.

    Sign recognition  -> Accuracy, Top-5, Macro-F1, balanced accuracy, ECE
    Sign sequence     -> WER / CER over gloss sequences
    Translation       -> BLEU / chrF over the generated Thai sentence
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ------------------------------------------------- stage 1: recognition ----
def topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    idx = np.argpartition(-logits, kth=min(k, logits.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean([yi in row for yi, row in zip(y, idx)]))


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def recognition_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    prob = softmax(logits)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {
        "accuracy": float((pred == y).mean()),
        "top5_accuracy": topk_accuracy(logits, y, 5),
        "top10_accuracy": topk_accuracy(logits, y, 10),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mean_confidence": float(prob.max(1).mean()),
        "ece": expected_calibration_error(prob, y),
        "n": int(len(y)),
    }


def per_class_table(logits, y, class_names):
    pred = logits.argmax(1)
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    rows = []
    for i, name in enumerate(class_names):
        d = rep.get(str(i))
        if d is None:
            continue
        rows.append({"gloss": name, "support": int(d["support"]),
                     "precision": d["precision"], "recall": d["recall"],
                     "f1": d["f1-score"]})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("f1")


def confusion(logits, y, n_classes):
    return confusion_matrix(y, logits.argmax(1), labels=np.arange(n_classes))


def top_confusions(cm, class_names, k: int = 20):
    import pandas as pd
    c = cm.copy()
    np.fill_diagonal(c, 0)
    flat = np.dstack(np.unravel_index(np.argsort(-c, axis=None), c.shape))[0][:k]
    return pd.DataFrame([{"true": class_names[i], "pred": class_names[j],
                          "count": int(c[i, j])} for i, j in flat if c[i, j] > 0])


# ---------------------------------------------------- stage 2: sequence ----
def _levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(refs: list, hyps: list) -> float:
    """Word (gloss) error rate over a list of token sequences."""
    num = sum(_levenshtein(list(r), list(h)) for r, h in zip(refs, hyps))
    den = sum(len(r) for r in refs)
    return float(num / max(den, 1))


def cer(refs: list, hyps: list) -> float:
    """Character error rate over the surface strings."""
    num = sum(_levenshtein(list("".join(r)), list("".join(h))) for r, h in zip(refs, hyps))
    den = sum(len("".join(r)) for r in refs)
    return float(num / max(den, 1))


def sequence_metrics(refs: list, hyps: list) -> dict:
    exact = float(np.mean([r == h for r, h in zip(refs, hyps)]))
    return {"wer": wer(refs, hyps), "cer": cer(refs, hyps),
            "sentence_exact_match": exact, "n_sequences": len(refs)}


# ------------------------------------------------- stage 3: translation ----
def translation_metrics(refs: list, hyps: list) -> dict:
    """BLEU / chrF with character tokenisation, which is the correct setting for
    Thai (no orthographic word boundaries)."""
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        return {"bleu_char": float(bleu.score), "chrf": float(chrf.score),
                "n": len(refs)}
    except Exception as e:                       # pragma: no cover
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------ open-set ----
def openset_metrics(score_known: np.ndarray, score_unknown: np.ndarray,
                    thr: float | None = None) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.r_[np.ones(len(score_known)), np.zeros(len(score_unknown))]
    s = np.r_[score_known, score_unknown]
    out = {"auroc": float(roc_auc_score(y, s)),
           "aupr_known": float(average_precision_score(y, s)),
           "aupr_unknown": float(average_precision_score(1 - y, -s))}
    if thr is not None:
        out["threshold"] = float(thr)
        out["tpr_known_accepted"] = float((score_known >= thr).mean())
        out["fpr_unknown_accepted"] = float((score_unknown >= thr).mean())
        out["unknown_recall"] = float((score_unknown < thr).mean())
    return out
