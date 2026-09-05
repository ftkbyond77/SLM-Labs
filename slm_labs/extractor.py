"""Video (.mp4 ทุก fps / ทุกสถานที่ถ่าย) → landmark DataFrame schema เดียวกับ CSV ใน dataset (MediaPipe Holistic)"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import cfg
from .vocab import ALL_COLS, POSE_PTS, FACE_PTS, POSE_COLS, FACE_COLS, LH_COLS, RH_COLS

POSE_IDX = [11, 12, 13, 14, 15, 16]                       # l_sh, r_sh, l_el, r_el, l_wr, r_wr
FACE_IDX = [105, 70, 300, 334, 61, 291]                   # lbrow_outer, lbrow_inner, rbrow_inner, rbrow_outer, mouth_right, mouth_left
HOLISTIC_TASK_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"


class HolisticExtractor:
    """วิดีโอ → DataFrame (frame, t_ms, lh_*, rh_*, pose+vis/pres, face) ที่ target_fps (time-based sampling)"""

    def __init__(self, target_fps=cfg.TARGET_FPS, verbose=True):
        import mediapipe as mp
        self.mp, self.target_fps = mp, target_fps
        self.legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
        if self.legacy:
            self.h = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            task = cfg.OUT_DIR / "holistic_landmarker.task"
            if not task.exists():
                urllib.request.urlretrieve(HOLISTIC_TASK_URL, task)
            self.vision = vision
            self.h = vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(task)), running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5, min_pose_landmarks_confidence=0.5, min_hand_landmarks_confidence=0.5,
                min_face_detection_confidence=0.5, min_face_landmarks_confidence=0.5))
        self._ts_base = 0                                       # VIDEO mode: timestamp ต้องเพิ่มขึ้นเสมอ แม้ข้ามไฟล์
        if verbose:
            print("MediaPipe backend:", "legacy solutions.holistic" if self.legacy else "Tasks HolisticLandmarker")

    def _detect(self, rgb, ts_ms):
        if self.legacy:
            r = self.h.process(rgb); g = lambda lm: (lm.landmark if lm else None)
            return g(r.left_hand_landmarks), g(r.right_hand_landmarks), g(r.pose_landmarks), g(r.face_landmarks)
        r = self.h.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), int(ts_ms))
        g = lambda lm: (lm if lm else None)
        return g(r.left_hand_landmarks), g(r.right_hand_landmarks), g(r.pose_landmarks), g(r.face_landmarks)

    def extract(self, video_path, max_seconds=None) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(video_path)); src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step_ms = 1000.0 / self.target_fps; next_ms = 0.0
        rows, fi, kept, t0, last_ts = [], 0, 0, time.time(), self._ts_base
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = 1000.0 * fi / src_fps
            if max_seconds and ts_ms > max_seconds * 1000:
                break
            if ts_ms + 1e-6 >= next_ms:
                next_ms += step_ms
                ts_int = max(self._ts_base + int(ts_ms), last_ts + 1); last_ts = ts_int
                lh, rh, pose, face = self._detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts_int)
                row = {"frame": kept, "t_ms": int(ts_ms)}
                for name, hand in (("lh", lh), ("rh", rh)):
                    for i in range(21):
                        p = hand[i] if hand else None
                        row[f"{name}_x{i}"], row[f"{name}_y{i}"], row[f"{name}_z{i}"] = (p.x, p.y, p.z) if p else (np.nan,) * 3
                for name, j in zip(POSE_PTS, POSE_IDX):
                    p = pose[j] if pose else None
                    row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = (p.x, p.y, p.z) if p else (np.nan,) * 3
                    row[f"{name}_vis"] = (getattr(p, "visibility", np.nan) if p else np.nan)
                    row[f"{name}_pres"] = (getattr(p, "presence", np.nan) if p else np.nan)
                for name, j in zip(FACE_PTS, FACE_IDX):
                    p = face[j] if face else None
                    row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = (p.x, p.y, p.z) if p else (np.nan,) * 3
                rows.append(row); kept += 1
            fi += 1
        cap.release(); dt = time.time() - t0
        self._ts_base = last_ts + 1000
        df = pd.DataFrame(rows, columns=ALL_COLS)
        df.attrs.update(dict(src_fps=src_fps, n_src_frames=fi, n_kept=kept, extract_sec=dt, fps_processed=kept / max(dt, 1e-6),
                             video=str(video_path)))
        return df


def check_video_quality(df: pd.DataFrame, verbose=True) -> dict:
    """ตรวจคุณภาพ input ก่อน inference → warnings ที่ actionable"""
    det = lambda cols: float((~df[cols[0]].isna()).mean())
    sw = np.linalg.norm(df[["l_shoulder_x", "l_shoulder_y"]].values - df[["r_shoulder_x", "r_shoulder_y"]].values, axis=1)
    wrist_below = float(((df["l_wrist_y"] > 1.0) | (df["r_wrist_y"] > 1.0)).mean())
    q = dict(frames=len(df), pose=det(POSE_COLS), face=det(FACE_COLS), lh=det(LH_COLS), rh=det(RH_COLS),
             any_hand=float((~df["lh_x0"].isna() | ~df["rh_x0"].isna()).mean()),
             shoulder_width=float(np.nanmean(sw)) if np.isfinite(np.nanmean(sw)) else 0.0, wrists_out_of_frame=wrist_below, warnings=[])
    if q["pose"] < 0.9: q["warnings"].append("pose detect < 90% — ให้เห็นไหล่ทั้งสองข้างชัด/แสงพอ")
    if q["face"] < 0.9: q["warnings"].append("face detect < 90% — หันหน้าตรง")
    if q["any_hand"] < 0.5: q["warnings"].append("เห็นมือ < 50% ของคลิป — กรอบภาพควรเห็นถึงเอว/มือตอนพัก")
    if q["wrists_out_of_frame"] > 0.3: q["warnings"].append("ข้อมือหลุดกรอบล่าง > 30% — ถอยกล้อง/เลื่อนกรอบลง")
    if not (0.12 < q["shoulder_width"] < 0.6): q["warnings"].append(f"ตัวเล็ก/ใหญ่ผิดปกติ (shoulder width={q['shoulder_width']:.2f})")
    if len(df) < 15: q["warnings"].append("คลิปสั้นเกิน (<1.5 s)")
    if verbose:
        print(f"quality: frames={q['frames']} pose={q['pose']:.2f} face={q['face']:.2f} LH={q['lh']:.2f} RH={q['rh']:.2f} shoulderW={q['shoulder_width']:.2f}")
        for w in q["warnings"]:
            print("  ⚠", w)
        if not q["warnings"]:
            print("  ✓ input OK")
    return q


def extract_cached(extractor, video_path, cache_dir: Path | None = None) -> pd.DataFrame:
    """cache landmark CSV ไว้ที่ outputs/landmarks_cache/<stem>@<fps>fps.csv (MediaPipe ช้า — ไม่ต้องรันซ้ำ)

    ชื่อไฟล์มี fps อยู่ด้วย เพราะ `cfg.TARGET_FPS` เป็นส่วนหนึ่งของ preprocessing — ถ้าเปลี่ยน fps
    ต้องดึงใหม่ ไม่ใช่ใช้ cache เดิม (bug ของ v1/v2: cache 10 fps ถูกใช้ทั้งที่ dataset อยู่ที่ ~25 fps)
    """
    cache_dir = Path(cache_dir or cfg.OUT_DIR / "landmarks_cache"); cache_dir.mkdir(parents=True, exist_ok=True)
    import json
    fps = getattr(extractor, "target_fps", cfg.TARGET_FPS)
    p = cache_dir / f"{Path(video_path).stem}@{fps:g}fps.csv"; pm = p.with_suffix(".meta.json")
    if p.exists():
        df = pd.read_csv(p); df.attrs.update(dict(video=str(video_path), cached=True, fps_processed=float("nan")))
        if pm.exists():
            df.attrs.update(json.loads(pm.read_text(encoding="utf-8")))
        return df
    df = extractor.extract(video_path); df.to_csv(p, index=False)
    pm.write_text(json.dumps({k: v for k, v in df.attrs.items() if k != "video"}), encoding="utf-8")
    return df
