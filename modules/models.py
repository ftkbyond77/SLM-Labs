"""Model zoo: a lean baseline and the multi-stream cross-modal architecture.

MSSF-Net (Multi-Stream Sign Fusion Network) follows the blueprint:

    lh -> MLP-LH  \
    rh -> MLP-RH   >-- Spatial Fusion --> 256 --\
    pose-> MLP-P  /                              >-- Cross-Modal Fusion
    face-> Face Encoder -> Face Temporal Enc ---/            |
                                                             v
                                             Temporal Transformer (4L, d=256, 8H)
                                                             |
                                                    Attention Pooling
                                                     /              \
                                        Classification Head    Embedding Head
                                            (184 glosses)      (256-D, cosine)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# --------------------------------------------------------------- blocks ----
class FrameMLP(nn.Module):
    """Per-frame landmark encoder (applied to every timestep in parallel)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out),
        )

    def forward(self, x):                       # (B,T,Din) -> (B,T,Dout)
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class CrossModalFusion(nn.Module):
    """Manual (hands+body) stream queries the face stream, then gates it in."""

    def __init__(self, d_model: int, d_face: int, n_heads: int, p: float):
        super().__init__()
        self.kv = nn.Linear(d_face, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        # learned scalar gate: if the face stream is uninformative the network
        # can shut it off instead of being forced to inject noise.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, manual, face):
        q = self.norm_q(manual)
        kv = self.kv(face)
        a, w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        out = manual + torch.tanh(self.gate) * a
        return self.norm_o(out), w


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(),
                                   nn.Linear(d_model // 2, 1))

    def forward(self, x):                        # (B,T,D) -> (B,D), (B,T)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.einsum("btd,bt->bd", x, w), w


class ArcMarginHead(nn.Module):
    """Cosine classifier with additive angular margin -> compact class clusters,
    which is what makes prototype distances usable for open-set rejection."""

    def __init__(self, d_in: int, n_classes: int, scale: float, margin: float):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_classes, d_in))
        nn.init.xavier_uniform_(self.w)
        self.scale, self.margin = scale, margin

    def forward(self, emb, target=None):
        cos = F.linear(F.normalize(emb), F.normalize(self.w)).clamp(-1 + 1e-7, 1 - 1e-7)
        if target is None:
            return self.scale * cos
        theta = torch.acos(cos)
        m = torch.zeros_like(cos)
        if target.dim() == 1:
            m.scatter_(1, target.view(-1, 1), self.margin)
        else:
            m = target * self.margin
        return self.scale * torch.cos(theta + m)


# ------------------------------------------------------------- baseline ----
class BaselineGRU(nn.Module):
    """All streams concatenated per frame -> BiGRU -> mean+max pool -> logits."""

    def __init__(self, dims: dict, n_classes: int, hidden: int = 256,
                 layers: int = 2, p: float = 0.3):
        super().__init__()
        d_in = sum(dims.values())
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=p if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Dropout(p),
                                  nn.Linear(4 * hidden, n_classes))

    def forward(self, xb, target=None):
        x = torch.cat([xb["lh"], xb["rh"], xb["pose"], xb["face"]], dim=-1)
        h, _ = self.gru(self.inp(x))
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        logits = self.head(pooled)
        return {"logits": logits, "embed": pooled, "arc": None, "attn": None}


# ------------------------------------------------------------ MSSF-Net -----
class MSSFNet(nn.Module):
    def __init__(self, dims: dict, n_classes: int, cfg: ModelConfig):
        super().__init__()
        d, p = cfg.d_model, cfg.dropout
        self.mlp_lh = FrameMLP(dims["lh"], cfg.stream_hidden, d // 2, p)
        self.mlp_rh = FrameMLP(dims["rh"], cfg.stream_hidden, d // 2, p)
        self.mlp_pose = FrameMLP(dims["pose"], cfg.stream_hidden, d // 2, p)
        self.spatial_fusion = nn.Sequential(
            nn.Linear(3 * (d // 2), d), nn.GELU(), nn.LayerNorm(d))

        self.face_enc = FrameMLP(dims["face"], cfg.face_dim, cfg.face_dim, p)
        self.face_temporal = nn.GRU(cfg.face_dim, cfg.face_dim // 2, num_layers=1,
                                    batch_first=True, bidirectional=True)
        self.cross = CrossModalFusion(d, cfg.face_dim, cfg.n_heads, p)

        self.pos = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ff_mult * d,
            dropout=p, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                              norm=nn.LayerNorm(d))
        self.pool = AttentionPooling(d)

        self.cls_head = nn.Sequential(nn.Dropout(p), nn.Linear(d, n_classes))
        self.embed_head = nn.Sequential(nn.Linear(d, cfg.embed_dim), nn.BatchNorm1d(cfg.embed_dim))
        self.arc = ArcMarginHead(cfg.embed_dim, n_classes, cfg.arc_scale, cfg.arc_margin)

    def forward(self, xb, target=None):
        manual = self.spatial_fusion(torch.cat(
            [self.mlp_lh(xb["lh"]), self.mlp_rh(xb["rh"]), self.mlp_pose(xb["pose"])], dim=-1))
        face, _ = self.face_temporal(self.face_enc(xb["face"]))
        fused, xattn = self.cross(manual, face)
        h = self.temporal(self.pos(fused))
        pooled, tattn = self.pool(h)
        emb = self.embed_head(pooled)
        return {"logits": self.cls_head(pooled),
                "embed": emb,
                "arc": self.arc(emb, target),
                "attn": tattn,
                "cross_attn": xattn,
                "face_gate": torch.tanh(self.cross.gate).detach()}


# ---------------------------------------------------------------- loss -----
def soft_cross_entropy(logits, target, smoothing: float = 0.0):
    """Handles both hard int targets and mixup soft targets."""
    n = logits.shape[-1]
    if target.dim() == 1:
        target = F.one_hot(target, n).float()
    if smoothing > 0:
        target = target * (1 - smoothing) + smoothing / n
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
