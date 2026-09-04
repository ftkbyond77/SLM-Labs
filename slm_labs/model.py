"""Multimodal Sign Encoder (Hand/Body/Face MLP → fusion → Transformer) + heads

heads:
  cls_head  : isolated sign classification (CE)
  ctc_head  : continuous sign sequence (CTC)
  emb_head  : L2-normalised embedding ของ clip / segment (ใช้ทำ prototype, open-set, memory/qdrant)
  face_proj : compact face embedding (cue ให้ LLM)
  DINOHead  : SignDINO-style self-distillation (global ↔ local view) — SSL pre-training
"""
from __future__ import annotations

import copy

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
        self.face_proj = nn.Linear(d, 64)
        self.d, self.emb_dim = d, emb_dim

    def embed(self, pooled):
        return F.normalize(self.emb_head(pooled), dim=-1)

    def forward(self, hand, body, face, mask, use_face=True):
        h, b, f = self.hand(hand), self.body(body), self.face(face)
        if not use_face:
            f = torch.zeros_like(f)
        x = self.fusion(torch.cat([h, b, f], -1))
        x = x + self.pos(torch.arange(x.size(1), device=x.device))[None]
        x = self.norm(self.encoder(x, src_key_padding_mask=mask))
        pooled = masked_mean(x, mask)
        return dict(logits_cls=self.cls_head(pooled),
                    logits_ctc=self.ctc_head(x),                       # (B,T,V)
                    emb=self.embed(pooled),                            # (B,E) L2-normalised
                    pooled=pooled,
                    face_emb=self.face_proj(masked_mean(f, mask)),
                    frame_emb=x)

    def forward_batch(self, x: dict, use_face=True):
        return self(x["hand"], x["body"], x["face"], x["mask"], use_face)

    def segment_embed(self, frame_emb: torch.Tensor, s: int, e: int):
        """embedding ของช่วงเฟรม [s,e) จาก frame_emb ของ global forward (global view)"""
        return self.embed(frame_emb[:, s:e].mean(1))


# ---------------- SignDINO-style self-distillation ----------------

class DINOHead(nn.Module):
    def __init__(self, d_in, out_dim, hidden=512, bottleneck=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        with torch.no_grad():
            self.last.parametrizations.weight.original0.fill_(1.0)   # g = 1 (ตาม DINO)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=cfg.SSL_TEACHER_TEMP, student_temp=cfg.SSL_STUDENT_TEMP, center_m=cfg.SSL_CENTER_M):
        super().__init__()
        self.tt, self.st, self.m = teacher_temp, student_temp, center_m
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: list, teacher_out: list):
        """student_out: list ของ logits ทุก view (global ก่อน) ; teacher_out: logits ของ global views เท่านั้น"""
        t = [F.softmax((o.detach() - self.center) / self.tt, dim=-1) for o in teacher_out]
        s = [F.log_softmax(o / self.st, dim=-1) for o in student_out]
        loss, n = 0.0, 0
        for iq, q in enumerate(t):
            for v, ls in enumerate(s):
                if v == iq:
                    continue                                          # ไม่เทียบ view เดียวกัน
                loss = loss + (-(q * ls).sum(-1)).mean(); n += 1
        loss = loss / max(n, 1)
        with torch.no_grad():
            bc = torch.cat([o.detach() for o in teacher_out]).mean(0, keepdim=True)
            self.center = self.center * self.m + bc * (1 - self.m)
        return loss


class SignDINO(nn.Module):
    """student = encoder (ที่จะเอาไป fine-tune ต่อ) + head ; teacher = EMA copy"""

    def __init__(self, encoder: SignEncoder, out_dim=cfg.SSL_OUT_DIM):
        super().__init__()
        self.student = encoder
        self.s_head = DINOHead(encoder.d, out_dim)
        self.teacher = copy.deepcopy(encoder)
        self.t_head = copy.deepcopy(self.s_head)
        for p in list(self.teacher.parameters()) + list(self.t_head.parameters()):
            p.requires_grad = False
        self.loss = DINOLoss(out_dim)

    def _pool(self, net, x):
        return net(x["hand"], x["body"], x["face"], x["mask"])["pooled"]

    def forward(self, views: list):
        s_out = [self.s_head(self._pool(self.student, v)) for v in views]
        with torch.no_grad():
            t_out = [self.t_head(self._pool(self.teacher, v)) for v in views[:2]]
        return self.loss(s_out, t_out)

    @torch.no_grad()
    def ema_update(self, m=cfg.SSL_EMA):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for ps, pt in zip(self.s_head.parameters(), self.t_head.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def build_model(n_classes, device):
    from .config import cfg as c
    return SignEncoder(n_classes, c.D_MODEL, c.N_HEADS, c.N_LAYERS, c.D_FF, c.DROPOUT, c.EMB_DIM).to(device)
