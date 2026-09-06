"""Open-set recognition: class prototypes, rejection scores, threshold calibration.

Any deployed sign-language reader is open-set by construction: a real signer will
produce signs outside the 184-gloss vocabulary.  Rather than forcing a wrong
gloss, the model emits ``_`` (unknown) whenever its evidence is weak.
"""
from __future__ import annotations

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def build_prototypes(embeds: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Class-mean of L2-normalised train embeddings, re-normalised."""
    e = l2n(embeds)
    proto = np.zeros((n_classes, e.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            proto[c] = e[m].mean(0)
    return l2n(proto)


def prototype_similarity(embeds: np.ndarray, proto: np.ndarray) -> np.ndarray:
    return l2n(embeds) @ proto.T                       # (N, C) cosine in [-1, 1]


def scores(logits: np.ndarray, embeds: np.ndarray, proto: np.ndarray) -> dict:
    """Three complementary rejection signals + their calibrated fusion.

    msp     - max softmax probability (classifier confidence)
    energy  - negative free energy, less over-confident than MSP on novel input
    proto   - max cosine similarity to a class prototype in the metric space
    """
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    energy = np.log(np.exp(z).sum(1)) + logits.max(1)      # logsumexp(logits)
    sim = prototype_similarity(embeds, proto)
    pmax = sim.max(1)
    # margin between the best and the runner-up prototype: a confusable sample
    # sits between two clusters and should also be rejected
    part = np.partition(sim, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    return {"msp": msp, "energy": energy, "proto": pmax, "proto_margin": margin,
            "pred_proto": sim.argmax(1), "sim": sim}


def fuse(sc: dict, stats: dict | None = None) -> tuple:
    """z-normalise each signal on a reference set, then average."""
    keys = ("msp", "proto", "proto_margin")
    if stats is None:
        stats = {k: (float(sc[k].mean()), float(sc[k].std() + 1e-8)) for k in keys}
    fused = np.mean([(sc[k] - stats[k][0]) / stats[k][1] for k in keys], axis=0)
    return fused, stats


def calibrate_threshold(score_known: np.ndarray, target_tpr: float = 0.90) -> float:
    """Threshold that keeps `target_tpr` of *known* validation samples accepted.

    Calibrated on validation only - the test set never influences the operating
    point, otherwise the open-set numbers would be optimistically biased.
    """
    return float(np.quantile(score_known, 1.0 - target_tpr))


def decide(pred: np.ndarray, score: np.ndarray, thr: float,
           class_names: list, unknown_token: str = "_") -> list:
    return [class_names[p] if s >= thr else unknown_token
            for p, s in zip(pred, score)]


def oscr(score_known, correct_known, score_unknown, n_points: int = 200) -> float:
    """Open-Set Classification Rate AUC: correct-classification rate of known
    samples vs. false-acceptance rate of unknown samples, integrated."""
    lo = float(min(score_known.min(), score_unknown.min()))
    hi = float(max(score_known.max(), score_unknown.max()))
    ths = np.linspace(lo, hi, n_points)
    ccr = np.array([np.mean(correct_known & (score_known >= t)) for t in ths])
    fpr = np.array([np.mean(score_unknown >= t) for t in ths])
    o = np.argsort(fpr)
    return float(np.trapezoid(ccr[o], fpr[o])) if hasattr(np, "trapezoid") \
        else float(np.trapz(ccr[o], fpr[o]))
