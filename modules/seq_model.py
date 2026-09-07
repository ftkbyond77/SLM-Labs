"""CSR-Net: the continuous reader that sits on top of the v1 word model.

v1 answers "which sign is this clip?". A real recording asks a different
question: "how many signs are in these ten seconds, where does each one start,
which of them do I know, and which do I not?".

CSR-Net keeps the whole MSSF-Net front end - the per-stream frame encoders,
spatial fusion and the face cross-modal gate - because those learned what a
Thai sign *looks like* from 3 302 clips, and nothing about that changes when
signs are strung together. What changes is everything downstream of pooling:

    MSSF-Net (v1)                        CSR-Net (v2)
    -------------------------------      --------------------------------------
    64 resampled frames per *clip*       native frame rate, whole utterance
    attention-pool -> one vector         temporal downsample x4, stay sequential
    softmax over 184 glosses             CTC over 184 + <unk> + blank
    "what is this sign"                  "what signs, in what order, when"

Three output heads, trained together:

* **CTC head** - alignment-free sequence loss. This is what lets the model be
  trained on utterances and run on video without anyone drawing boundaries.
* **Frame head** - per-step gloss, supervised by the composer's exact frame
  mapping. CTC alone is a slow, high-variance teacher; the frame mapping is free
  supervision that we happen to have, and it is also what makes the decode
  inspectable (§ posteriorgram plots).
* **Boundary head** - per-step "is a sign being articulated here", which
  separates real signs from movement epenthesis. This is the head that answers
  "how many signs did I see" independently of whether any of them was known.

The vocabulary layout is fixed and shared with the decoder:
    [0 .. V-1] glosses    [V] <unk>    [V+1] blank
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .models import CrossModalFusion, FrameMLP, PositionalEncoding


class TemporalDownsample(nn.Module):
    """Two strided depth-preserving convolutions: T -> T/4.

    A sign lasts ~50 frames, so 4x downsampling still leaves ~13 steps per sign -
    plenty for CTC - while cutting self-attention cost by 16x, which is what
    makes a 500-frame utterance fit on a 4 GB GPU.
    """

    def __init__(self, d: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2), nn.GELU())
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)
        self.factor = 4

    def forward(self, x):                          # (B,T,D) -> (B,ceil(T/4),D)
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.norm(y))


class CSRNet(nn.Module):
    """Continuous Sign Reader. `n_gloss` real classes + <unk> + blank."""

    def __init__(self, dims: dict, n_gloss: int, cfg: ModelConfig,
                 use_unk: bool = True):
        super().__init__()
        d, p = cfg.d_model, cfg.dropout
        self.n_gloss = n_gloss
        self.use_unk = use_unk
        self.unk_id = n_gloss                       # only meaningful if use_unk
        self.blank_id = n_gloss + 1 if use_unk else n_gloss
        self.n_out = self.blank_id + 1

        # ---- front end, weight-compatible with MSSFNet so it can be inherited
        self.mlp_lh = FrameMLP(dims["lh"], cfg.stream_hidden, d // 2, p)
        self.mlp_rh = FrameMLP(dims["rh"], cfg.stream_hidden, d // 2, p)
        self.mlp_pose = FrameMLP(dims["pose"], cfg.stream_hidden, d // 2, p)
        self.spatial_fusion = nn.Sequential(
            nn.Linear(3 * (d // 2), d), nn.GELU(), nn.LayerNorm(d))
        self.face_enc = FrameMLP(dims["face"], cfg.face_dim, cfg.face_dim, p)
        self.face_temporal = nn.GRU(cfg.face_dim, cfg.face_dim // 2, num_layers=1,
                                    batch_first=True, bidirectional=True)
        self.cross = CrossModalFusion(d, cfg.face_dim, cfg.n_heads, p)

        # ---- sequence trunk
        self.down = TemporalDownsample(d, p)
        self.pos = PositionalEncoding(d, max_len=1024)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ff_mult * d,
            dropout=p, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                              norm=nn.LayerNorm(d))

        # ---- heads
        self.ctc_head = nn.Sequential(nn.Dropout(p), nn.Linear(d, self.n_out))
        self.frame_head = nn.Sequential(nn.Dropout(p), nn.Linear(d, self.n_out))
        self.bnd_head = nn.Linear(d, 1)

    # ------------------------------------------------------------------
    def forward(self, xb: dict, mask: torch.Tensor | None = None) -> dict:
        """xb: {stream: (B,T,D)}; mask: (B,T) True where valid."""
        manual = self.spatial_fusion(torch.cat(
            [self.mlp_lh(xb["lh"]), self.mlp_rh(xb["rh"]), self.mlp_pose(xb["pose"])],
            dim=-1))
        face, _ = self.face_temporal(self.face_enc(xb["face"]))
        fused, _ = self.cross(manual, face)
        h = self.down(fused)

        m4 = None
        if mask is not None:
            # a downsampled step is valid if any of its source frames was
            m4 = F.max_pool1d(mask.float().unsqueeze(1),
                              kernel_size=4, stride=4,
                              ceil_mode=True).squeeze(1) > 0.5
            m4 = m4[:, : h.shape[1]]
            if m4.shape[1] < h.shape[1]:
                m4 = F.pad(m4, (0, h.shape[1] - m4.shape[1]), value=False)

        h = self.temporal(self.pos(h),
                          src_key_padding_mask=(~m4) if m4 is not None else None)
        return {"ctc": self.ctc_head(h),          # (B,T4,n_out)
                "frame": self.frame_head(h),      # (B,T4,n_out)
                "bnd": self.bnd_head(h).squeeze(-1),   # (B,T4)
                "hidden": h,
                "mask4": m4}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_word_model(self, sd: dict, verbose: bool = True) -> dict:
        """Inherit MSSF-Net's visual front end (Stage 1 -> Stage 2).

        Only tensors whose name *and* shape match are copied, so the transfer is
        explicit and auditable rather than a silent partial load.
        """
        own = self.state_dict()
        taken, skipped = [], []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.clone()
                taken.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(own)
        if verbose:
            groups = sorted({k.split(".")[0] for k in taken})
            print(f"[CSRNet] inherited {len(taken)} tensors from the word model: "
                  f"{', '.join(groups)}")
            print(f"[CSRNet] not inherited ({len(skipped)}): "
                  f"{', '.join(sorted({k.split('.')[0] for k in skipped}))}")
        return {"taken": taken, "skipped": skipped}

    def freeze_frontend(self, frozen: bool = True):
        """Stage 2 freezes the visual encoder and trains only the sequence trunk;
        Stage 3 unfreezes everything at a lower learning rate."""
        for m in (self.mlp_lh, self.mlp_rh, self.mlp_pose, self.spatial_fusion,
                  self.face_enc, self.face_temporal, self.cross):
            for p in m.parameters():
                p.requires_grad = not frozen
        return self


# ---------------------------------------------------------------- losses ---
def ctc_loss(logits: torch.Tensor, mask4: torch.Tensor, targets: list,
             blank: int) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)     # (T,B,C)
    in_len = mask4.sum(1).long()
    tgt = torch.cat([torch.as_tensor(t, dtype=torch.long) for t in targets]).to(logits.device)
    tgt_len = torch.tensor([len(t) for t in targets], dtype=torch.long,
                           device=logits.device)
    # CTC is undefined when the input is shorter than the label sequence
    in_len = torch.maximum(in_len, tgt_len)
    return F.ctc_loss(logp, tgt, in_len, tgt_len, blank=blank,
                      reduction="mean", zero_infinity=True)


def frame_targets(spans: list, labels: list, T4: int, factor: int, blank: int,
                  unk_of: list | None = None, unk_id: int | None = None) -> np.ndarray:
    """Frame mapping -> per-downsampled-step label (blank = between signs)."""
    y = np.full(T4, blank, dtype=np.int64)
    for i, (s, e) in enumerate(spans):
        a, b = int(np.floor(s / factor)), int(np.ceil(e / factor))
        a, b = max(0, a), min(T4, b)
        if b > a:
            lab = labels[i]
            if unk_of is not None and unk_of[i] and unk_id is not None:
                lab = unk_id
            y[a:b] = lab
    return y


# ---------------------------------------------------------------- decode ---
def greedy_ctc(logp: np.ndarray, blank: int, mask: np.ndarray | None = None):
    """Collapse a CTC posterior into (token, start_step, end_step, confidence).

    Repeats are merged and blanks dropped, the standard CTC read-out. The step
    range of every surviving token is kept, because on real video *where* a sign
    happened is as useful as what it was - it is what makes the output
    inspectable and what feeds the open-set gate a crop to score.
    """
    if mask is not None:
        logp = logp[mask]
    best = logp.argmax(-1)
    conf = np.exp(logp.max(-1))
    out, i = [], 0
    while i < len(best):
        j = i
        while j + 1 < len(best) and best[j + 1] == best[i]:
            j += 1
        if best[i] != blank:
            out.append({"token": int(best[i]), "start": i, "end": j + 1,
                        "confidence": float(conf[i:j + 1].mean()),
                        "peak": int(i + np.argmax(conf[i:j + 1]))})
        i = j + 1
    return out


def collapse_runs(labels: np.ndarray, blank: int, min_run: int = 1):
    """Same read-out applied to the *frame* head, whose runs are contiguous by
    construction and therefore give cleaner spans than CTC's spiky peaks."""
    out, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if labels[i] != blank and (j - i + 1) >= min_run:
            out.append({"token": int(labels[i]), "start": i, "end": j + 1})
        i = j + 1
    return out
