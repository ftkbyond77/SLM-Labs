"""Central configuration for the TSL-ONE-S multimodal sign-language pipeline.

Every tunable lives here so the notebook stays thin and debuggable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "TSL-ONE-Pose"
TEST_DIR = ROOT / "data_test"
FONT_DIR = ROOT / "font"
RESULT_DIR = ROOT / "prompt" / "result"
ARTIFACT_DIR = ROOT / "artifacts"          # only the best checkpoint lands here

# ------------------------------------------------- landmark topology -------
# Raw clip layout is (T, 150) -> (T, 75, 2): MediaPipe Holistic
#   0..32  pose (33 landmarks, normalised image coords)
#   33..53 left hand  (21)
#   54..74 right hand (21)
N_LANDMARKS = 75
POSE_SLICE = slice(0, 33)
LHAND_SLICE = slice(33, 54)
RHAND_SLICE = slice(54, 75)

# MediaPipe pose indices we actually use (upper body only; 23..32 are
# extrapolated legs for a seated/upper-body recording and carry no signal).
FACE_IDX = np.arange(0, 11)            # nose, 6 eye pts, 2 ears, 2 mouth corners
BODY_IDX = np.arange(11, 23)           # shoulders, elbows, wrists, hand stubs
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, MOUTH_L, MOUTH_R = 0, 2, 5, 7, 8, 9, 10

# --------------------------------------------------- split definition ------
# Signer-independent split: a signer appears in exactly one partition, so no
# clip of a test signer (or any near-duplicate of it) is ever seen in training.
TEST_SIGNERS = ("05", "12", "24")
VAL_SIGNERS = ("06", "20")

# Glosses held out entirely from the open-set probe model (Model B).
N_NOVEL_GLOSSES = 24
NOVEL_SEED = 1337


@dataclass
class FeatureConfig:
    n_frames: int = 64             # every clip is resampled to this length
    use_velocity: bool = True      # dX
    use_acceleration: bool = True  # d2X
    smooth_window: int = 3         # moving-average on raw coords (0 = off)
    eps: float = 1e-6


@dataclass
class AugConfig:
    enabled: bool = True
    rotate_deg: float = 10.0
    scale_range: tuple = (0.90, 1.10)
    shift: float = 0.03
    noise_std: float = 0.008
    time_scale: tuple = (0.80, 1.20)
    frame_drop_p: float = 0.05
    hand_drop_p: float = 0.05      # simulate a missed MediaPipe hand


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.20
    stream_hidden: int = 256
    face_dim: int = 128
    embed_dim: int = 256
    arc_scale: float = 24.0
    arc_margin: float = 0.20


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 64
    lr: float = 4e-4
    weight_decay: float = 0.02
    warmup_epochs: int = 8
    label_smoothing: float = 0.10
    grad_clip: float = 1.0
    amp: bool = True
    patience: int = 30
    embed_loss_w: float = 0.30
    seed: int = 42
    num_workers: int = 0           # data is fully in RAM; workers only add overhead


@dataclass
class Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = Config()


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_openai_key() -> str | None:
    """Read OPENAI_API_KEY from .env (never printed anywhere)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")
