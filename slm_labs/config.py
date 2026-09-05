"""Global configuration for SLM Labs (Thai Sign Language -> Thai -> Speech).

ทุก module import `cfg` จากที่นี่ — แก้ค่า hyper-parameter / path ได้ที่เดียว

v3 (fixed_v2): ตัด SSL ออก, บังคับ fps เดียวกันทั้ง dataset และวิดีโอใหม่, ฝึก emb head จริง,
Stage B ไม่แตะ encoder (Stage A ปลอดภัย 100%)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
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
    USE_EXPERT_PRIMARY: bool = True        # expert_primary_02/03 (เฉพาะคลิป original, is_augmented == False) = signer 2 คน
    USE_EXPERT_AUGMENTED: bool = False     # True = ใช้ augmented 32k คลิปด้วย (ช้า, กิน RAM)
    EXPERT_MAX_PER_CLASS: int | None = None
    SPLIT_MODE: str = "official"           # "official" = train: expert + user[calib] / val,test: user[test] แบ่งครึ่ง (คลิปคนละชุด)
    #                                        "random"   = โปรโตคอลเดิมของ v1/v2 (สุ่ม 70/15/15 บน user + expert ทั้งหมดเข้า train)
    SENT_SPLIT_BY_PATTERN: bool = True     # True = 76 pattern แยก train/val/test ไม่ปนกัน (เดิม: ปนกัน 100% → leakage)
    # ---------- features ----------
    TARGET_FPS: float = 15.0               # **canonical fps** — resample ทั้ง dataset landmarks (จาก t_ms) และวิดีโอใหม่ ให้เท่ากัน
    #                                        (v1/v2 bug: dataset ~24-30 fps แต่ extractor ดึง 10 fps → ท่าเดียวกันยาวต่างกัน 2.5-3x)
    TRIM_ISOLATED: bool = True             # ตัดหัว-ท้ายที่มือยังพักออกจาก isolated clip → distribution ตรงกับ segment ตอน inference
    MAX_FRAMES_ISO: int = 64               # 64 / 15 fps = 4.3 s (คลิป isolated ยาวราว 2-3 s)
    MAX_FRAMES_SEQ: int = 192              # 192 / 15 fps = 12.8 s
    # ---------- model ----------
    D_MODEL: int = 256
    N_HEADS: int = 4
    N_LAYERS: int = 4
    D_FF: int = 512
    DROPOUT: float = 0.2
    EMB_DIM: int = 128                     # segment / clip embedding (L2-normalised) ที่เก็บลง memory
    # ---------- training ----------
    EPOCHS_A: int = 40                     # Stage A: isolated sign recognition (CE + ArcFace บน emb head)
    EPOCHS_B: int = 25                     # Stage B: CTC head บน encoder ที่ freeze ไว้
    BATCH_SIZE: int = 32
    LR: float = 3e-4
    LR_STAGE_B: float = 3e-3               # head-only → ใช้ LR สูงได้
    WEIGHT_DECAY: float = 0.05
    LABEL_SMOOTH: float = 0.1
    SEED: int = 42
    # ---------- embedding / metric learning (ทำให้ open-set ใช้งานได้จริง) ----------
    ARC_SCALE: float = 24.0                # ArcFace s
    ARC_MARGIN: float = 0.25               # ArcFace m
    EMB_LOSS_W: float = 0.5                # น้ำหนัก ArcFace ใน Stage A (0 = ปิด → กลับไปเป็นพฤติกรรม v1/v2)
    # ---------- Stage B (sequence) ----------
    CTC_STRIDE: int = 4                    # avg-pool เวลา x4 ก่อน CTC head (192 -> 48 เฟรม) ลด blank-dominance
    BLANK_PENALTY: float = 0.0             # ลบจาก blank logit ตอน decode — **ต้อง tune บน val** (tune_blank_penalty)
    FREEZE_ENCODER_STAGE_B: bool = True    # True = Stage B แตะแค่ ctc_head → Stage A ไม่มีทางพัง (v2 เคยตกจาก 0.988 -> 0.337)
    SYN_SENTENCES: int = 400               # ประโยคสังเคราะห์ (ต่อ isolated clips ของ split train) เพิ่มใน Stage B
    # ---------- open-set / unknown detection ----------
    SEG_CLS_CONF: float = 0.45             # ความมั่นใจ classifier ต่อ segment ขั้นต่ำ
    PROTO_SIM_QUANTILE: float = 0.10       # threshold cosine กับ prototype = quantile นี้ของ correct-class sims บน val
    MEMORY_SIM_THRESHOLD: float = 0.75     # cosine กับ learned prototype (จาก memory ที่ annotate แล้ว)
    HOLDOUT_CLASSES: int = 8               # จำนวน class ที่กันออกจาก train เพื่อวัด open-set กับ "คำที่ไม่เคยเห็นจริง ๆ"
    # ---------- segmentation (motion-based) ----------
    SEG_MIN_FRAMES: int = 8                # segment สั้นสุด (เฟรมที่ 15 fps = 0.53 s)
    SEG_MIN_DIST: int = 12                 # ระยะห่างขั้นต่ำระหว่าง valley (0.8 s)
    SEG_ACTIVE_WRIST_Y: float = 1.6        # ข้อมือสูงกว่า (y น้อยกว่า) 1.6x shoulder-width ใต้กึ่งกลางไหล่ = กำลังทำท่า
    SEG_ENERGY_FRAC: float = 0.4           # energy > frac x p90 = เคลื่อนไหว
    SEG_PROMINENCE: float = 0.3            # valley prominence (สัดส่วนของ range)
    USE_CTC_CUTS: bool = True              # ใช้ CTC frame-posterior เสนอขอบเขตเพิ่ม (ดู openset.ctc_boundaries)
    SEG_MAX_FRAMES: int = 48               # segment ที่ยาวกว่านี้ (3.2 s @15fps) ถือว่ามีมากกว่า 1 คำ → แตกที่ CTC cut
                                           # (คลิปจริงเซ็นเร็วกว่า dataset ~2 เท่า; ค่าที่ tune จาก dataset อย่างเดียวจะรวมคำติดกัน)
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
