"""Landmarks (T×D CSV schema ของ dataset) → Hand / Body / Face features + augmentation + hand-activity"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .config import cfg
from .vocab import ALL_COLS, LH_COLS, RH_COLS, POSE_COLS, FACE_COLS

HAND_DIM, BODY_DIM, FACE_DIM = 21 * 3 * 2 * 2 + 2, 6 * 3 + 5 * 3 + 6 * 3, 6 * 3 + 6 * 3 + 3   # 254, 51, 39

# layout ของ hand vector: [lh xyz 63][rh xyz 63][lh Δ 63][rh Δ 63][lh pres][rh pres]
# layout ของ body vector: [pose xyz 18][rel 15][pose Δ 18]  (pose order: l_sh, r_sh, l_el, r_el, l_wr, r_wr)


def _ffill_bfill(a: np.ndarray) -> np.ndarray:
    T = a.shape[0]
    df = pd.DataFrame(a.copy().reshape(T, -1)).interpolate(limit_direction="both", axis=0)
    return df.fillna(0.0).values.reshape(a.shape)


def resample_landmarks(lm: dict, target_fps: float | None = None) -> dict:
    """บังคับให้ทุกคลิปอยู่ที่ **fps เดียวกัน** โดยใช้ `t_ms` (เวลาจริง) เป็นแกน

    ทำไมถึงสำคัญ: landmark ใน TSL-51 ถูกดึงที่ ~24-30 fps (user_sign fps_extracted=30, t_ms dt≈42 ms)
    ส่วน `HolisticExtractor` ของเราดึงวิดีโอใหม่ที่ `cfg.TARGET_FPS` — ถ้าสองค่านี้ไม่ตรงกัน ท่าเดียวกัน
    จะมีจำนวนเฟรมต่างกัน 2-3 เท่า และ encoder ที่เรียนจาก dataset จะอ่านวิดีโอจริงไม่ออกเลย
    (นี่คือสาเหตุหลักที่ v1/v2 ทำนายคลิปจริงเป็นคนละคำทั้งหมด)

    ใช้ nearest-neighbour ตามเวลา → รักษา NaN (เฟรมที่ไม่เจอมือ) ไว้ตามเดิม
    """
    fps = target_fps or cfg.TARGET_FPS
    t = np.asarray(lm["t_ms"], np.float64)
    if len(t) < 2 or not np.isfinite(t).all():
        return lm
    t = np.maximum.accumulate(t)                       # กัน timestamp ย้อนกลับ
    dur_ms = float(t[-1] - t[0])
    if dur_ms <= 0:
        return lm
    n = max(2, int(round(dur_ms / 1000.0 * fps)) + 1)
    grid = np.linspace(t[0], t[-1], n)
    idx = np.round(np.interp(grid, t, np.arange(len(t)))).astype(int).clip(0, len(t) - 1)
    out = {k: v[idx] for k, v in lm.items() if k != "t_ms"}
    out["t_ms"] = grid.astype(np.float32)
    return out


def load_landmarks(src, fps: float | None = None) -> dict:
    """path | DataFrame → dict(lh (T,21,3), rh, pose (T,6,3), face (T,6,3), t_ms (T,))

    resample ไปที่ `fps` (default `cfg.TARGET_FPS`) เสมอ — dataset และวิดีโอใหม่จึงอยู่บนแกนเวลาเดียวกัน
    ส่ง fps=0 เพื่อปิด (ดูข้อมูลดิบ)
    """
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    T = len(df)
    lm = dict(
        lh=df[LH_COLS].values.reshape(T, 21, 3).astype(np.float32),
        rh=df[RH_COLS].values.reshape(T, 21, 3).astype(np.float32),
        pose=df[POSE_COLS].values.reshape(T, 6, 3).astype(np.float32),
        face=df[FACE_COLS].values.reshape(T, 6, 3).astype(np.float32),
        t_ms=df["t_ms"].values.astype(np.float32),
    )
    fps = cfg.TARGET_FPS if fps is None else fps
    return resample_landmarks(lm, fps) if fps else lm


def make_features(lm: dict) -> dict:
    """→ dict(hand (T,254), body (T,51), face (T,39)) — normalised by shoulder centre / width"""
    pose = _ffill_bfill(lm["pose"]); face = _ffill_bfill(lm["face"])
    ls, rs = pose[:, 0], pose[:, 1]
    center = (ls + rs) / 2
    width = np.linalg.norm((ls - rs)[:, :2], axis=-1)
    med = np.nanmedian(width) if np.isfinite(np.nanmedian(width)) else 1.0
    width = np.where(np.isfinite(width) & (width > 1e-3), width, med)
    norm = lambda p: (p - center[:, None, :]) / width[:, None, None]

    xyz, vel_, pres = [], [], []
    for h in (lm["lh"], lm["rh"]):
        present = (~np.isnan(h[:, 0, 0])).astype(np.float32)
        hn = norm(np.nan_to_num(h, nan=0.0)) * present[:, None, None]
        vel = np.diff(hn, axis=0, prepend=hn[:1]) * present[:, None, None]
        xyz.append(hn.reshape(len(hn), -1)); vel_.append(vel.reshape(len(hn), -1)); pres.append(present[:, None])
    hand = np.concatenate(xyz + vel_ + pres, axis=1)

    pn = norm(pose)
    rel = np.stack([pn[:, 2] - pn[:, 0], pn[:, 4] - pn[:, 0], pn[:, 3] - pn[:, 1], pn[:, 5] - pn[:, 1], pn[:, 4] - pn[:, 5]], 1)
    pvel = np.diff(pn, axis=0, prepend=pn[:1])
    body = np.concatenate([pn.reshape(len(pn), -1), rel.reshape(len(pn), -1), pvel.reshape(len(pn), -1)], 1)

    fn = norm(face)
    fvel = np.diff(fn, axis=0, prepend=fn[:1])
    brow_y = fn[:, :4, 1].mean(1); mouth_y = fn[:, 4:, 1].mean(1)
    geo = np.stack([mouth_y - brow_y,
                    np.linalg.norm(fn[:, 4, :2] - fn[:, 5, :2], axis=-1),
                    np.linalg.norm(fn[:, 1, :2] - fn[:, 2, :2], axis=-1)], 1)
    facef = np.concatenate([fn.reshape(len(fn), -1), fvel.reshape(len(fn), -1), geo], 1)

    return dict(hand=np.nan_to_num(hand).astype(np.float32),
                body=np.nan_to_num(body).astype(np.float32),
                face=np.nan_to_num(facef).astype(np.float32))


def temporal_resample(feat: dict, n_out: int) -> dict:
    T = len(feat["hand"]); idx = np.linspace(0, T - 1, n_out).round().astype(int)
    return {k: v[idx] for k, v in feat.items()}


def crop(feat: dict, s: int, e: int) -> dict:
    return {k: v[s:e] for k, v in feat.items()}


# ---------------- hand activity (ใช้ตัด segment ของแต่ละคำ โดยไม่พึ่ง vocab) ----------------

def hand_activity(feat: dict, smooth: int = 3) -> dict:
    """→ dict(energy (T,), raised (T,), present (T,), wrist_y (T,2))

    energy  : ความเร็วเฉลี่ยของจุดบนมือ (มือที่เร็วกว่า) — สูง = กำลังเคลื่อนไหว
    raised  : ข้อมืออย่างน้อยหนึ่งข้างอยู่สูงกว่าระดับ "พัก" (เทียบกับกึ่งกลางไหล่, หน่วย shoulder-width)
    """
    h, b = feat["hand"], feat["body"]
    T = len(h)
    lh_v = np.linalg.norm(h[:, 126:189].reshape(T, 21, 3), axis=-1).mean(1) * h[:, 252]
    rh_v = np.linalg.norm(h[:, 189:252].reshape(T, 21, 3), axis=-1).mean(1) * h[:, 253]
    wr_v = np.linalg.norm(b[:, 33 + 12:33 + 18].reshape(T, 2, 3), axis=-1).max(1)   # wrist velocity (pose)
    energy = np.maximum(np.maximum(lh_v, rh_v), wr_v)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        energy = np.convolve(energy, k, mode="same")
    wrist_y = b[:, [13, 16]]                                   # l_wrist_y, r_wrist_y (normalised, +y = ลง)
    present = np.maximum(h[:, 252], h[:, 253])
    return dict(energy=energy.astype(np.float32), wrist_y=wrist_y, present=present)


# ---------------- augmentation (feature-level) ----------------
AUG = dict(speed=(0.8, 1.25), rot_deg=12, scale=(0.85, 1.15), shift=0.05, hand_drop=0.15, mirror=0.3, noise=0.01)


def _xyz_blocks():
    return dict(hand=[(0, 21), (63, 21), (126, 21), (189, 21)], body=[(0, 6), (18, 5), (33, 6)], face=[(0, 6), (18, 6)])


def spatial_transform(f: dict, deg, scale, dx, dy) -> dict:
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32) * scale
    out = {}
    for k, v in f.items():
        v = v.copy()
        for st, n in _xyz_blocks()[k]:
            xy = v[:, st:st + 3 * n].reshape(len(v), n, 3)
            xy[..., :2] = xy[..., :2] @ R.T + np.array([dx, dy], np.float32)
            v[:, st:st + 3 * n] = xy.reshape(len(v), -1)
        out[k] = v
    return out


def mirror_swap(f: dict) -> dict:
    h = f["hand"].copy()
    h[:, 0:63], h[:, 63:126] = f["hand"][:, 63:126], f["hand"][:, 0:63]
    h[:, 126:189], h[:, 189:252] = f["hand"][:, 189:252], f["hand"][:, 126:189]
    h[:, 252], h[:, 253] = f["hand"][:, 253], f["hand"][:, 252]
    b = f["body"].copy()
    for st, n in _xyz_blocks()["body"]:
        blk = b[:, st:st + 3 * n].reshape(len(b), n, 3)
        blk = blk[:, [1, 0, 3, 2, 5, 4]] if n == 6 else blk[:, [2, 3, 0, 1, 4]]
        b[:, st:st + 3 * n] = blk.reshape(len(b), -1)
    fc = f["face"].copy()
    for st, n in _xyz_blocks()["face"]:
        fc[:, st:st + 3 * n] = fc[:, st:st + 3 * n].reshape(len(fc), n, 3)[:, [3, 2, 1, 0, 5, 4]].reshape(len(fc), -1)
    out = dict(hand=h, body=b, face=fc)
    for k in out:
        for st, n in _xyz_blocks()[k]:
            out[k][:, st:st + 3 * n:3] *= -1
    out["body"][:, 30:33] *= -1
    return out


def augment(feat: dict, strength: float = 1.0, temporal=True) -> dict:
    """feature-level augmentation: temporal + spatial + hand-dropout + mirror (จำลองกล้อง/คน/สถานที่ต่างกัน)"""
    T = len(feat["hand"]); f = feat
    if temporal:
        f = temporal_resample(feat, max(8, int(T * random.uniform(*AUG["speed"]))))
        if random.random() < 0.5:
            T2 = len(f["hand"]); c = int(T2 * random.uniform(0.0, 0.1)); e = T2 - int(T2 * random.uniform(0.0, 0.1))
            f = {k: v[c:max(e, c + 8)] for k, v in f.items()}
    rot, sh = AUG["rot_deg"] * strength, AUG["shift"] * strength
    sc = (1 - (1 - AUG["scale"][0]) * strength, 1 + (AUG["scale"][1] - 1) * strength)
    f = spatial_transform(f, random.uniform(-rot, rot), random.uniform(*sc), random.uniform(-sh, sh), random.uniform(-sh, sh))
    if random.random() < AUG["mirror"]:
        f = mirror_swap(f)
    if random.random() < AUG["hand_drop"] * strength:
        T2 = len(f["hand"]); s0 = random.randrange(T2); e0 = min(T2, s0 + random.randint(1, max(1, T2 // 6)))
        side = random.choice([(0, 126, 252), (126, 252, 253)])
        f["hand"][s0:e0, side[0]:side[1]] = 0; f["hand"][s0:e0, side[2]] = 0
    return {k: v + np.random.normal(0, AUG["noise"] * strength, v.shape).astype(np.float32) for k, v in f.items()}


def active_span(feat: dict, wrist_y: float = 1.6, energy_frac: float = 0.4, pad: int = 2, min_frames: int = 8):
    """ช่วงเฟรมที่ "กำลังทำท่า" ของทั้งคลิป (ตัดหัว-ท้ายที่มือยังพักอยู่)

    ใช้กับ isolated clips ตอน train/eval เพื่อให้ **ตรงกับสิ่งที่โมเดลเจอตอน inference** — ที่นั่นคลิปถูกตัด
    เป็น segment ด้วย hand-activity แล้ว ถ้า Stage A เรียนจากคลิปที่มีหัว-ท้ายพักยาว ๆ แต่ต้องทำนาย
    segment ที่ตัดมาแล้ว จะเป็นคนละ distribution
    """
    act = hand_activity(feat)
    e = act["energy"]; wy = np.nanmin(act["wrist_y"], axis=1)
    raised = wy < wrist_y
    ref = np.percentile(e[raised], 90) if raised.any() else np.percentile(e, 90)
    active = raised | ((act["present"] > 0) & (e > energy_frac * max(float(ref), 1e-4)))
    idx = np.flatnonzero(active)
    if len(idx) < min_frames:
        return 0, len(e)
    s, t = max(0, int(idx[0]) - pad), min(len(e), int(idx[-1]) + 1 + pad)
    return (s, t) if t - s >= min_frames else (0, len(e))


def trim_to_active(feat: dict) -> dict:
    s, e = active_span(feat)
    return crop(feat, s, e)


def time_reverse(feat: dict) -> dict:
    """pseudo-unknown sign: เล่นย้อนกลับ (ลำดับเฟรมกลับ + velocity กลับทิศ) — ท่าที่ไม่ตรงกับคำใด ๆ ใน vocab"""
    out = {k: v[::-1].copy() for k, v in feat.items()}
    out["hand"][:, 126:252] *= -1; out["body"][:, 33:51] *= -1; out["face"][:, 18:36] *= -1
    return out
