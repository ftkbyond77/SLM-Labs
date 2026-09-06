"""In-RAM batching + GPU-side augmentation.

The whole corpus is ~4.1k clips, so it is materialised once as contiguous
float32 tensors and sliced directly.  No DataLoader workers, no per-item Python
work, no disk I/O in the training loop: every batch is one indexing op plus a
single host-to-device copy, and augmentation is applied batched on the GPU.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import AugConfig

STREAMS = ("lh", "rh", "pose", "face")


class StreamBank:
    """Holds (N, T, D) tensors per stream plus labels, and yields batches."""

    def __init__(self, bank: dict, labels: np.ndarray, device: torch.device,
                 pin: bool = True):
        self.device = device
        self.x = {}
        for k in STREAMS:
            t = torch.from_numpy(np.ascontiguousarray(bank[k]))
            self.x[k] = t.pin_memory() if (pin and device.type == "cuda") else t
        self.y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.n, self.t = self.x["lh"].shape[0], self.x["lh"].shape[1]

    def dims(self) -> dict:
        return {k: int(v.shape[-1]) for k, v in self.x.items()}

    def take(self, idx: np.ndarray) -> tuple:
        i = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        xb = {k: v.index_select(0, i).to(self.device, non_blocking=True)
              for k, v in self.x.items()}
        yb = self.y.index_select(0, i).to(self.device, non_blocking=True)
        return xb, yb

    def iter_batches(self, idx: np.ndarray, batch_size: int, shuffle: bool,
                     rng: np.random.Generator | None = None):
        order = np.asarray(idx)
        if shuffle:
            rng = rng or np.random.default_rng()
            order = order[rng.permutation(len(order))]
        for s in range(0, len(order), batch_size):
            yield self.take(order[s:s + batch_size])


# --------------------------------------------------------- augmentation ----
def augment(xb: dict, cfg: AugConfig, gen: torch.Generator | None = None) -> dict:
    """Batched, differentiable-free augmentation applied on-device.

    The body-frame normalisation already removes translation / scale / signer
    size, so augmentation targets what normalisation cannot: sensor noise,
    tempo, MediaPipe hand drop-outs and short occlusions.
    """
    if not cfg.enabled:
        return xb
    out = {}
    B, T = xb["lh"].shape[0], xb["lh"].shape[1]
    dev = xb["lh"].device

    # --- tempo warp: one random speed factor per clip, shared across streams
    lo, hi = cfg.time_scale
    speed = torch.empty(B, 1, device=dev).uniform_(lo, hi, generator=gen)
    base = torch.linspace(0, T - 1, T, device=dev).unsqueeze(0)
    centre = (T - 1) / 2.0
    pos = (base - centre) * speed + centre
    pos = pos.clamp(0, T - 1)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (pos - i0.float()).unsqueeze(-1)

    # --- per-clip global magnitude jitter (kinematic amplitude)
    slo, shi = cfg.scale_range
    amp = torch.empty(B, 1, 1, device=dev).uniform_(slo, shi, generator=gen)

    keep_frames = torch.rand(B, T, 1, device=dev, generator=gen) > cfg.frame_drop_p
    for k, v in xb.items():
        g0 = torch.gather(v, 1, i0.unsqueeze(-1).expand(-1, -1, v.shape[-1]))
        g1 = torch.gather(v, 1, i1.unsqueeze(-1).expand(-1, -1, v.shape[-1]))
        z = g0 * (1 - w) + g1 * w
        z = z * amp
        z = z + torch.randn(z.shape, device=dev, generator=gen) * cfg.noise_std
        # short occlusions: blank a few frames, the transformer must interpolate
        z = z * keep_frames
        out[k] = z

    # --- simulate a MediaPipe hand miss on one side
    for k in ("lh", "rh"):
        drop = (torch.rand(B, 1, 1, device=dev, generator=gen) < cfg.hand_drop_p).float()
        out[k] = out[k] * (1 - drop)
    return out


def mixup(xb: dict, y: torch.Tensor, n_classes: int, alpha: float,
          gen: torch.Generator | None = None):
    """Manifold-free input mixup; returns soft targets."""
    if alpha <= 0:
        return xb, torch.nn.functional.one_hot(y, n_classes).float()
    B = y.shape[0]
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B, device=y.device, generator=gen)
    xm = {k: lam * v + (1 - lam) * v[perm] for k, v in xb.items()}
    y1 = torch.nn.functional.one_hot(y, n_classes).float()
    ym = lam * y1 + (1 - lam) * y1[perm]
    return xm, ym
