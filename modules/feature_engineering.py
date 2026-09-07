"""Vectorised feature engineering: raw (T,75,2) MediaPipe clips -> stream tensors.

Design goals
------------
* invariance  : translation / scale / small-rotation invariance via a body frame
* locality    : hand *shape* is encoded wrist-locally, hand *location* is kept
                separately (in TSL, where a sign is made carries meaning)
* dynamics    : first (dX) and second (d2X) temporal derivatives
* robustness  : MediaPipe drop-outs (exact zeros) are interpolated, and a
                per-frame presence mask is fed to the model as a feature
* speed       : everything is numpy-vectorised over the time axis, one pass
                per clip, no Python loops over frames or landmarks
"""
from __future__ import annotations

import numpy as np

from .config import (BODY_IDX, FACE_IDX, LHAND_SLICE, L_EAR, L_EYE, L_SHOULDER,
                     L_WRIST, MOUTH_L, MOUTH_R, NOSE, POSE_SLICE, RHAND_SLICE,
                     R_EAR, R_EYE, R_SHOULDER, R_WRIST, FeatureConfig)

FINGERTIPS = np.array([4, 8, 12, 16, 20])
EPS = 1e-6


# ------------------------------------------------------------- utilities ---
def _nan_from_zeros(x: np.ndarray) -> np.ndarray:
    """Exact (0,0) is the MediaPipe sentinel for a landmark that was not
    detected; turn it into NaN so it can be interpolated instead of learned."""
    x = x.copy()
    miss = (x == 0).all(axis=-1)
    x[miss] = np.nan
    return x


def _interp_nan_time(x: np.ndarray) -> np.ndarray:
    """Linear interpolation along time for every (landmark, coord) column."""
    T = x.shape[0]
    flat = x.reshape(T, -1)
    idx = np.arange(T)
    good = ~np.isnan(flat)
    allbad = ~good.any(axis=0)
    out = flat.copy()
    for c in np.where(~allbad)[0]:                 # loops over <=150 columns only
        g = good[:, c]
        if g.all():
            continue
        out[:, c] = np.interp(idx, idx[g], flat[g, c])
    out[:, allbad] = 0.0
    return out.reshape(x.shape)


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Cheap jitter removal - O(T*D) sliding mean via a running cumulative sum
    (no per-column convolution loop)."""
    if w is None or w < 2 or x.shape[0] < w:
        return x
    T = x.shape[0]
    pad = w // 2
    flat = np.pad(x.reshape(T, -1), ((pad, pad), (0, 0)), mode="edge")
    cs = np.cumsum(flat, axis=0, dtype=np.float64)
    cs = np.vstack([np.zeros((1, cs.shape[1])), cs])
    out = (cs[w:] - cs[:-w]) / w
    return out[:T].reshape(x.shape).astype(np.float32)


def resample_time(x: np.ndarray, n: int) -> np.ndarray:
    """Uniformly resample a (T, ...) array to exactly n frames.

    Fully vectorised linear interpolation: gather the two bracketing frames for
    all n output positions at once and blend, instead of np.interp per column.
    """
    T = x.shape[0]
    if T == n:
        return np.ascontiguousarray(x, dtype=np.float32)
    flat = x.reshape(T, -1)
    dst = np.linspace(0.0, T - 1.0, n, dtype=np.float32)
    i0 = np.floor(dst).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (dst - i0).astype(np.float32)[:, None]
    out = flat[i0] * (1.0 - w) + flat[i1] * w
    return out.reshape((n,) + x.shape[1:]).astype(np.float32)


def _derivatives(x: np.ndarray, order: int = 2) -> list:
    """[dX, d2X] with zero-padded first frames, same shape as x."""
    outs, cur = [], x
    for _ in range(order):
        d = np.diff(cur, axis=0, prepend=cur[:1])
        outs.append(d.astype(np.float32))
        cur = d
    return outs


# --------------------------------------------------------- normalisation ---
def body_frame(pose: np.ndarray):
    """Neck origin + shoulder-width scale, both robust over the whole clip."""
    neck = 0.5 * (pose[:, L_SHOULDER] + pose[:, R_SHOULDER])          # (T,2)
    sw = np.linalg.norm(pose[:, L_SHOULDER] - pose[:, R_SHOULDER], axis=-1)
    scale = float(np.median(sw[sw > EPS])) if np.any(sw > EPS) else 1.0
    return neck, max(scale, 1e-3)


# ------------------------------------------------------------- streams -----
def _hand_stream(hand: np.ndarray, neck: np.ndarray, scale: float,
                 present: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(T, D) = wrist-local shape + global location + dynamics + presence."""
    wrist = hand[:, :1]                                   # (T,1,2)
    local = hand - wrist
    hs_t = np.linalg.norm(local, axis=-1).max(axis=1)     # (T,)
    hs = float(np.median(hs_t[hs_t > EPS])) if np.any(hs_t > EPS) else scale
    hs = max(hs, 1e-3)
    local = local / hs                                    # scale-free hand shape
    loc = (wrist[:, 0] - neck) / scale                    # where the hand is (T,2)

    tip_d = np.linalg.norm(hand[:, FINGERTIPS] - wrist, axis=-1) / hs   # (T,5)
    span = np.linalg.norm(hand[:, FINGERTIPS, None, :] -
                          hand[:, None, FINGERTIPS, :], axis=-1).max((1, 2)) / hs

    base = np.concatenate([local.reshape(len(hand), -1), loc,
                           tip_d, span[:, None]], axis=1).astype(np.float32)
    feats = [base]
    d1, d2 = _derivatives(base, 2)
    if cfg.use_velocity:
        feats.append(d1)
    if cfg.use_acceleration:
        feats.append(d2)
    feats.append(present[:, None].astype(np.float32))
    return np.concatenate(feats, axis=1)


def _pose_stream(pose: np.ndarray, lh: np.ndarray, rh: np.ndarray,
                 neck: np.ndarray, scale: float, cfg: FeatureConfig) -> np.ndarray:
    body = (pose[:, BODY_IDX] - neck[:, None]) / scale
    base = body.reshape(len(pose), -1)

    # relational cues: sign location is defined w.r.t. the body and the other hand
    lw = (lh[:, 0] - neck) / scale
    rw = (rh[:, 0] - neck) / scale
    nose = (pose[:, NOSE] - neck) / scale
    rel = np.stack([
        np.linalg.norm(lw - rw, axis=-1),          # inter-hand distance
        np.linalg.norm(lw - nose, axis=-1),        # left hand to face
        np.linalg.norm(rw - nose, axis=-1),        # right hand to face
        np.linalg.norm(lw, axis=-1),               # left hand to neck
        np.linalg.norm(rw, axis=-1),               # right hand to neck
    ], axis=1)
    base = np.concatenate([base, rel], axis=1).astype(np.float32)

    feats = [base]
    d1, d2 = _derivatives(base, 2)
    if cfg.use_velocity:
        feats.append(d1)
    if cfg.use_acceleration:
        feats.append(d2)
    return np.concatenate(feats, axis=1)


def _face_stream(pose: np.ndarray, neck: np.ndarray, scale: float,
                 cfg: FeatureConfig) -> np.ndarray:
    """Coarse *face impression* stream.

    TSL-ONE-S ships MediaPipe **Pose**, not the 468-point Face Mesh, so the only
    facial evidence in the .npy files is pose landmarks 0..10 (nose, 6 eye
    points, 2 ears, 2 mouth corners).  That supports head pose, head motion,
    gaze line and mouth-width dynamics - a coarse facial-expression proxy - but
    not brow / eyelid / cheek articulation.  The stream is built and fused
    exactly as a dense face stream would be, so Face Mesh is a drop-in upgrade.
    """
    face = pose[:, FACE_IDX]
    nose = face[:, NOSE]
    iod_t = np.linalg.norm(face[:, L_EYE] - face[:, R_EYE], axis=-1)
    iod = float(np.median(iod_t[iod_t > EPS])) if np.any(iod_t > EPS) else scale * 0.2
    iod = max(iod, 1e-3)
    local = ((face - nose[:, None]) / iod).reshape(len(pose), -1)

    eye_vec = face[:, L_EYE] - face[:, R_EYE]
    roll = np.arctan2(eye_vec[:, 1], eye_vec[:, 0])
    mouth_c = 0.5 * (face[:, MOUTH_L] + face[:, MOUTH_R])
    eye_c = 0.5 * (face[:, L_EYE] + face[:, R_EYE])
    geo = np.stack([
        np.linalg.norm(face[:, MOUTH_L] - face[:, MOUTH_R], axis=-1) / iod,  # mouth width
        np.linalg.norm(mouth_c - nose, axis=-1) / iod,                       # jaw drop proxy
        np.linalg.norm(face[:, L_EAR] - face[:, R_EAR], axis=-1) / iod,      # yaw proxy
        np.linalg.norm(mouth_c - eye_c, axis=-1) / iod,                      # pitch proxy
        np.cos(roll), np.sin(roll),                                          # head roll
        (nose[:, 0] - neck[:, 0]) / scale,                                   # head shift x
        (nose[:, 1] - neck[:, 1]) / scale,                                   # head shift y
    ], axis=1)

    base = np.concatenate([local, geo], axis=1).astype(np.float32)
    d1, _ = _derivatives(base, 2)
    return np.concatenate([base, d1], axis=1)


# -------------------------------------------------------------- public -----
FACE_GEO_NAMES = ["mouth_width", "jaw_drop", "ear_span", "eye_mouth",
                  "roll_cos", "roll_sin", "head_dx", "head_dy"]


def clip_to_streams(clip: np.ndarray, cfg: FeatureConfig,
                    resample: bool = True) -> dict:
    """raw (T,75,2) -> {lh, rh, pose, face}, each (cfg.n_frames, D) float32.

    `resample=False` keeps the clip's native length. That is what the continuous
    (utterance-level) model needs: a sentence must stay frame-synchronous so the
    sequence head can say *when* each sign happened, and squashing a 300-frame
    utterance into 64 frames would destroy exactly that information. The feature
    definitions are otherwise identical, so the same standardiser applies to both.
    """
    x = _nan_from_zeros(np.asarray(clip, dtype=np.float32))
    lh_present = (~np.isnan(x[:, LHAND_SLICE]).all(axis=(1, 2))).astype(np.float32)
    rh_present = (~np.isnan(x[:, RHAND_SLICE]).all(axis=(1, 2))).astype(np.float32)
    x = _interp_nan_time(x)
    x = _moving_average(x, cfg.smooth_window)

    pose, lh, rh = x[:, POSE_SLICE], x[:, LHAND_SLICE], x[:, RHAND_SLICE]
    # a wholly-undetected hand is parked on its pose wrist stub and flagged absent
    if lh_present.sum() == 0:
        lh = np.repeat(pose[:, L_WRIST][:, None], 21, axis=1)
    if rh_present.sum() == 0:
        rh = np.repeat(pose[:, R_WRIST][:, None], 21, axis=1)
    neck, scale = body_frame(pose)

    raw = {
        "lh": _hand_stream(lh, neck, scale, lh_present, cfg),
        "rh": _hand_stream(rh, neck, scale, rh_present, cfg),
        "pose": _pose_stream(pose, lh, rh, neck, scale, cfg),
        "face": _face_stream(pose, neck, scale, cfg),
    }
    if not resample:
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in raw.items()}
    return {k: resample_time(v, cfg.n_frames) for k, v in raw.items()}


def build_feature_bank(clips, cfg: FeatureConfig, progress: bool = True) -> dict:
    """Materialise the whole corpus once as contiguous (N, T, D) float32 arrays."""
    it = clips
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(clips, desc="feature engineering", unit="clip")
        except Exception:
            pass
    per = [clip_to_streams(c, cfg) for c in it]
    return {k: np.ascontiguousarray(np.stack([p[k] for p in per]), dtype=np.float32)
            for k in ("lh", "rh", "pose", "face")}


def stream_dims(bank: dict) -> dict:
    return {k: int(v.shape[-1]) for k, v in bank.items()}


def standardiser(bank: dict, idx: np.ndarray) -> dict:
    """Per-feature mean/std computed on TRAIN rows only (never on val/test)."""
    stats = {}
    for k, v in bank.items():
        tr = v[idx].reshape(-1, v.shape[-1])
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd < 1e-4] = 1.0
        stats[k] = (mu.astype(np.float32), sd.astype(np.float32))
    return stats


def apply_standardiser(bank: dict, stats: dict) -> dict:
    return {k: ((v - stats[k][0]) / stats[k][1]).astype(np.float32)
            for k, v in bank.items()}
