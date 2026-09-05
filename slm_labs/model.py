"""Multimodal Sign Encoder (Hand/Body/Face MLP → fusion → Transformer) + heads

heads:
  cls_head  : isolated sign classification (CE)
  ctc_head  : continuous sign sequence (CTC) — Stage B, ฝึกโดย freeze encoder
  emb_head  : L2-normalised embedding ของ clip / segment (prototype, open-set, memory/qdrant)
  arc       : ArcFace margin head บน emb_head — **ตัวที่ทำให้ emb_head ถูกฝึกจริง**
  face_proj : compact face embedding (cue ให้ LLM)

v1/v2 bug ที่แก้ในเวอร์ชันนี้: emb_head ไม่เคยได้ gradient เลย (ไม่มี loss ใดแตะมัน) →
prototype / open-set / memory ทั้งหมดถูกคำนวณจาก "random projection" → AUROC ≈ 0.5
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import HAND_DIM, BODY_DIM, FACE_DIM
from .config import cfg


class ModalityMLP(nn.Module):
    def __init__(self, d_in, d, p):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x):
        return self.net(x)


def masked_mean(x, mask):
    """x (B,T,D), mask True = pad"""
    keep = (~mask).unsqueeze(-1).float()
    return (x * keep).sum(1) / keep.sum(1).clamp(min=1)


class ArcFace(nn.Module):
    """additive angular margin — บังคับให้ embedding ของ class เดียวกันเกาะกลุ่ม และต่าง class แยกกันเป็นมุม

    ผลที่ต้องการ: cos(emb, prototype) ของคำที่รู้จัก >> ของคำที่ไม่เคยเห็น → open-set threshold ใช้งานได้จริง
    """

    def __init__(self, emb_dim, n_classes, s: float = cfg.ARC_SCALE, m: float = cfg.ARC_MARGIN):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = float(s), float(m)

    def forward(self, emb, y=None):
        cos = F.linear(F.normalize(emb.float(), dim=-1), F.normalize(self.W.float(), dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        if y is None:
            return self.s * cos
        tgt = torch.cos(torch.acos(cos) + self.m)
        return self.s * torch.where(F.one_hot(y, cos.size(1)).bool(), tgt, cos)


class SignEncoder(nn.Module):
    def __init__(self, n_classes, d=cfg.D_MODEL, heads=cfg.N_HEADS, layers=cfg.N_LAYERS, ff=cfg.D_FF, p=cfg.DROPOUT,
                 emb_dim=cfg.EMB_DIM, max_len=1024):
        super().__init__()
        self.hand, self.body, self.face = ModalityMLP(HAND_DIM, d, p), ModalityMLP(BODY_DIM, d, p), ModalityMLP(FACE_DIM, d, p)
        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.LayerNorm(d), nn.Dropout(p))
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, p, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.cls_head = nn.Linear(d, n_classes)
        self.ctc_head = nn.Linear(d, n_classes + 1)          # +blank
        self.emb_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, emb_dim))
        self.arc = ArcFace(emb_dim, n_classes)
        self.face_proj = nn.Linear(d, 64)
        self.d, self.emb_dim, self.ctc_stride = d, emb_dim, cfg.CTC_STRIDE

    # ---------- helpers ----------
    def ctc_lens(self, lens):
        return (lens + self.ctc_stride - 1) // self.ctc_stride

    def embed(self, pooled):
        return F.normalize(self.emb_head(pooled), dim=-1)

    def encoder_parameters(self):
        """ทุกอย่างยกเว้น ctc_head (ใช้ freeze ตอน Stage B)"""
        return [p for n, p in self.named_parameters() if not n.startswith("ctc_head")]

    # ---------- forward ----------
    def forward(self, hand, body, face, mask, use_face=True):
        h, b, f = self.hand(hand), self.body(body), self.face(face)
        if not use_face:
            f = torch.zeros_like(f)
        x = self.fusion(torch.cat([h, b, f], -1))
        x = x + self.pos(torch.arange(x.size(1), device=x.device))[None]
        x = self.norm(self.encoder(x, src_key_padding_mask=mask))
        pooled = masked_mean(x, mask)
        xs = F.avg_pool1d(x.transpose(1, 2), self.ctc_stride, self.ctc_stride, ceil_mode=True).transpose(1, 2) if self.ctc_stride > 1 else x
        return dict(logits_cls=self.cls_head(pooled),
                    logits_ctc=self.ctc_head(xs),                      # (B, ceil(T/stride), V)
                    emb=self.embed(pooled),                            # (B,E) L2-normalised
                    pooled=pooled,
                    face_emb=self.face_proj(masked_mean(f, mask)),
                    frame_emb=x)

    def forward_batch(self, x: dict, use_face=True):
        return self(x["hand"], x["body"], x["face"], x["mask"], use_face)

    def segment_embed(self, frame_emb: torch.Tensor, s: int, e: int):
        """embedding ของช่วงเฟรม [s,e) จาก frame_emb ของ global forward"""
        return self.embed(frame_emb[:, s:e].mean(1))


def build_model(n_classes, device):
    from .config import cfg as c
    return SignEncoder(n_classes, c.D_MODEL, c.N_HEADS, c.N_LAYERS, c.D_FF, c.DROPOUT, c.EMB_DIM).to(device)
