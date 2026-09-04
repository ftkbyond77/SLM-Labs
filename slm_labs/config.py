"""Global configuration for SLM Labs (Thai Sign Language -> Thai -> Speech).

ทุก module import `cfg` จากที่นี่ — แก้ค่า hyper-parameter / path ได้ที่เดียว
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

try:  # .env (OPENAI_API_KEY ...) — optional
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


@dataclass
class Config:
    # ---------- data ----------
    HF_REPO: str = "Namonpas/thai-sign-language-tsl51"
    DATA_DIR: Path = ROOT / "data" / "tsl51"
    USE_EXPERT_PRIMARY: bool = True        # expert_primary_02/03 (เฉพาะคลิป original, is_augmented == False)
    USE_EXPERT_AUGMENTED: bool = False     # True = ใช้ augmented 32k คลิปด้วย (ช้า, กิน RAM)
    EXPERT_MAX_PER_CLASS: int | None = None
    # ---------- features ----------
    TARGET_FPS: float = 10.0               # dataset landmarks ถูกดึงที่ ~10 fps → วิดีโอใหม่ต้อง resample ให้ตรง
    MAX_FRAMES_ISO: int = 96
    MAX_FRAMES_SEQ: int = 256
    # ---------- model ----------
    D_MODEL: int = 256
    N_HEADS: int = 4
    N_LAYERS: int = 4
    D_FF: int = 512
    DROPOUT: float = 0.2
    EMB_DIM: int = 128                     # segment / clip embedding (L2-normalised) ที่เก็บลง memory
    # ---------- SSL: SignDINO-style global/local self-distillation ----------
    SSL_EPOCHS: int = 10
    SSL_MAX_FRAMES: int = 64               # ความยาวสูงสุดของ view (ลด compute ของ SSL)
    SSL_OUT_DIM: int = 512                 # จำนวน "prototype" ของ DINO head
    SSL_N_LOCAL: int = 2                   # local views ต่อคลิป (global view = 2 เสมอ)
    SSL_GLOBAL_CROP: tuple = (0.6, 1.0)    # สัดส่วนความยาวคลิปของ global view
    SSL_LOCAL_CROP: tuple = (0.2, 0.45)    # สัดส่วนความยาวคลิปของ local view
    SSL_TEACHER_TEMP: float = 0.04
    SSL_STUDENT_TEMP: float = 0.1
    SSL_EMA: float = 0.995
    SSL_CENTER_M: float = 0.9
    SSL_LR: float = 5e-4
    # ---------- training ----------
    EPOCHS_A: int = 30                     # Stage A: isolated sign recognition (CE)
    EPOCHS_B: int = 30                     # Stage B: continuous (CTC + CE)
    BATCH_SIZE: int = 32
    LR: float = 3e-4
    WEIGHT_DECAY: float = 0.05
    LABEL_SMOOTH: float = 0.1
    CE_WEIGHT_STAGE_B: float = 0.3
    SYN_SENTENCES: int = 300               # ประโยคสังเคราะห์ (ต่อ isolated clips) เพิ่มใน Stage B — แก้ CTC blank-collapse
    SEED: int = 42
    # ---------- open-set / unknown detection ----------
    MIN_TOKEN_CONF: float = 0.5            # CTC token confidence ต่ำกว่านี้ → ไม่เชื่อ
    PROTO_SIM_QUANTILE: float = 0.05       # threshold cosine กับ prototype = quantile นี้ของ correct-class sims บน val
    LOCAL_RESCUE_CLS_CONF: float = 0.7     # local-view classifier ต้องมั่นใจเท่านี้ถึงจะ "กู้" คำที่ CTC พลาด
    MEMORY_SIM_THRESHOLD: float = 0.80     # cosine กับ learned prototype (จาก memory ที่ annotate แล้ว)
    # ---------- segmentation (motion-based) ----------
    SEG_MIN_FRAMES: int = 6                # segment สั้นสุด (เฟรมที่ 10 fps)
    SEG_MIN_DIST: int = 8                  # ระยะห่างขั้นต่ำระหว่าง valley (0.5 s)
    SEG_ACTIVE_WRIST_Y: float = 1.6        # ข้อมือสูงกว่า (y น้อยกว่า) 1.6×shoulder-width ใต้กึ่งกลางไหล่ = กำลังทำท่า
    SEG_ENERGY_FRAC: float = 0.4           # energy > frac × p90 = เคลื่อนไหว
    SEG_PROMINENCE: float = 0.3            # valley prominence (สัดส่วนของ range) — จาก tune_segmentation (trade-off: dataset ช้า vs คลิปจริงเร็ว)
    # ---------- memory (qdrant local) ----------
    MEMORY_DIR: Path = ROOT / "memory"
    QDRANT_COLLECTION: str = "sign_segments"
    # ---------- LLM ----------
    LLM_PROVIDER: str = "openai"           # "openai" | "rule"
    LLM_MODEL: str = "gpt-5.6"
    # ---------- TTS ----------
    TTS_BACKEND: str = "mms"               # "mms" | "openai" | "off"
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    OPENAI_TTS_VOICE: str = "alloy"
    # ---------- misc ----------
    FONT_DIR: Path = ROOT / "font"
    OUT_DIR: Path = ROOT / "outputs"
    DATA_TEST_DIR: Path = ROOT / "data_test"

    def to_dict(self):
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


cfg = Config()
cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
cfg.MEMORY_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")


def seed_all(s: int = cfg.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_dev(x: dict, device: str = DEVICE):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}


seed_all(cfg.SEED)
