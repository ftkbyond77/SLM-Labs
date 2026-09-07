"""Stage-2 corpus builder: isolated word clips -> continuous utterances.

Why this module exists
----------------------
TSL-ONE-S is an **isolated-sign** corpus. A model trained on it has never seen
what actually happens in a real recording:

* **movement epenthesis** - the hands must travel from where sign A ends to
  where sign B starts, and that travel is itself a burst of motion that looks
  like a sign but is not one;
* **coarticulation** - the handshape of A bleeds into the start of B and the
  handshape of B is anticipated at the end of A, so no sign is ever produced in
  its citation form;
* **prosody** - signs are re-timed (0.75x - 1.35x), and holds appear at clause
  boundaries but *not* between every pair of signs.

v1.2 measured exactly this: WER on stitched utterances rose 0.36 -> 0.64 the
moment the rest gap between signs was removed. That gap is the training
distribution, not the decoder.

This module synthesises the missing distribution from the word clips we have,
and - crucially - returns the **frame mapping** (which output frames belong to
which gloss), so the sequence model can be trained with alignment-free CTC *and*
frame-level auxiliary supervision, and so every decode can be visualised against
the truth.

Camera-level augmentation is applied on top, calibrated to the statistics
measured on the real `data_test/` recordings, so the training distribution
covers the deployment distribution rather than sitting beside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import L_SHOULDER, R_SHOULDER

EPS = 1e-6
LH, RH = slice(33, 54), slice(54, 75)
POSE = slice(0, 33)
L_HAND_WRIST, R_HAND_WRIST = 33, 54

# MediaPipe Pose left/right landmark pairs, for the mirror transform.
_POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                  (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                  (27, 28), (29, 30), (31, 32)]


# --------------------------------------------------------------- config ----
@dataclass
class ComposeConfig:
    """Everything that shapes one synthetic utterance.

    Defaults describe *fluent* signing, which is what the real recordings are.
    `preset_paused()` reproduces the easy, pause-separated regime for
    comparison; both are used in evaluation so the two are never conflated.
    """
    # ---- sequence structure
    min_signs: int = 2
    max_signs: int = 6
    speed: tuple = (0.75, 1.35)          # per-sign tempo
    lead_rest: tuple = (6, 20)           # frames before the first sign
    tail_rest: tuple = (6, 20)           # frames after the last sign

    # ---- inter-sign behaviour
    trans_frames: tuple = (4, 22)        # movement-epenthesis length, distance-scaled
    hold_p: float = 0.22                 # chance of a prosodic hold at a boundary
    hold_frames: tuple = (4, 14)
    coartic: float = 0.45                # handshape carry-over / anticipation strength
    coartic_frames: int = 7

    # ---- camera / capture, calibrated on data_test/
    scale_range: tuple = (0.80, 1.25)    # signer distance from camera
    aspect_range: tuple = (0.80, 1.25)   # residual frame-aspect anisotropy
    rot_deg: float = 6.0
    shift: float = 0.04
    fps_scale: tuple = (0.85, 1.20)      # 25 fps vs 30 fps recordings
    coord_noise: float = 0.0025
    mirror_p: float = 0.0                # left-handed signers (off by default; measured)
    hand_drop_burst_p: float = 0.18      # MediaPipe losing a hand for a run of frames
    drop_len: tuple = (3, 14)

    # ---- unknown-class material
    chimera_p: float = 0.0               # fraction of slots filled with a non-sign motion

    def copy(self, **kw) -> "ComposeConfig":
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(kw)
        return ComposeConfig(**d)

    def preset_paused(self) -> "ComposeConfig":
        """The easy regime: a hold at every boundary, no handshape bleed."""
        return self.copy(hold_p=1.0, hold_frames=(10, 20), coartic=0.0,
                         trans_frames=(6, 14))

    def preset_v12(self) -> "ComposeConfig":
        """Exactly the v1.2 control set: fixed tempo, a 14-frame rest at every
        boundary, a short linear blend, no camera effects. Kept so the v2 numbers
        can be read against the v1.2 numbers on the same footing."""
        return self.copy(speed=(1.0, 1.0), hold_p=1.0, hold_frames=(14, 15),
                         coartic=0.0, trans_frames=(6, 7), lead_rest=(1, 2),
                         tail_rest=(1, 2), scale_range=(1.0, 1.0), rot_deg=0.0,
                         shift=0.0, fps_scale=(1.0, 1.0), coord_noise=0.0,
                         hand_drop_burst_p=0.0, mirror_p=0.0, aspect_range=(1.0, 1.0))

    def preset_clean(self) -> "ComposeConfig":
        """No camera augmentation - isolates the effect of coarticulation alone."""
        return self.copy(scale_range=(1.0, 1.0), aspect_range=(1.0, 1.0),
                         rot_deg=0.0, shift=0.0, fps_scale=(1.0, 1.0),
                         coord_noise=0.0, hand_drop_burst_p=0.0, mirror_p=0.0)


# --------------------------------------------------------- primitives ------
def _presence(clip: np.ndarray) -> np.ndarray:
    """(T,2) float mask: is the left / right hand block detected this frame."""
    return np.stack([~(clip[:, LH] == 0).all(-1).all(-1),
                     ~(clip[:, RH] == 0).all(-1).all(-1)], axis=1).astype(np.float32)


def _apply_presence(clip: np.ndarray, pres: np.ndarray) -> np.ndarray:
    out = clip.copy()
    out[pres[:, 0] < 0.5, LH] = 0.0
    out[pres[:, 1] < 0.5, RH] = 0.0
    return out


def retime(clip: np.ndarray, factor: float) -> np.ndarray:
    """Resample a clip to `len/factor` frames.

    Landmarks and the detection mask are resampled separately, then the mask is
    re-thresholded, so a missing hand never gets blended into a plausible-looking
    position halfway to the origin.
    """
    T = len(clip)
    n = max(8, int(round(T / max(factor, 1e-3))))
    if n == T:
        return clip.astype(np.float32)
    pres = _presence(clip)
    src = clip.reshape(T, -1).astype(np.float32)
    dst = np.linspace(0.0, T - 1.0, n, dtype=np.float32)
    i0 = np.floor(dst).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (dst - i0)[:, None]
    out = (src[i0] * (1 - w) + src[i1] * w).reshape(n, 75, 2)
    pr = (pres[i0] * (1 - w) + pres[i1] * w)
    return _apply_presence(out.astype(np.float32), (pr > 0.5).astype(np.float32))


def _minjerk(n: int) -> np.ndarray:
    """Minimum-jerk time profile - how a human arm actually moves between two
    postures (smooth start, smooth stop), instead of a linear ramp."""
    t = np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
    return (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5).astype(np.float32)


def _hand_local(block: np.ndarray) -> np.ndarray:
    """21 hand points expressed relative to their own wrist = the handshape."""
    return block - block[:1]


def _neck_scale(clip: np.ndarray):
    pose = clip[:, POSE]
    neck = 0.5 * (pose[:, L_SHOULDER] + pose[:, R_SHOULDER])
    sw = np.linalg.norm(pose[:, L_SHOULDER] - pose[:, R_SHOULDER], axis=-1)
    s = float(np.median(sw[sw > EPS])) if np.any(sw > EPS) else 0.25
    return neck, max(s, 1e-3)


# ------------------------------------------------------- coarticulation ----
def coarticulate(a: np.ndarray, b: np.ndarray, cfg: ComposeConfig,
                 rng: np.random.Generator) -> tuple:
    """Bleed A's final handshape into the start of B and anticipate B's initial
    handshape at the end of A.

    Only the *shape* (wrist-local geometry) is exchanged; each sign keeps its own
    hand *location*, because in TSL location is a separate phonological parameter
    and swapping it would change the sign rather than blur it.
    """
    if cfg.coartic <= 0 or len(a) < 3 or len(b) < 3:
        return a, b
    k = int(min(cfg.coartic_frames, len(a) // 2, len(b) // 2))
    if k < 1:
        return a, b
    a, b = a.copy(), b.copy()
    w = cfg.coartic * np.linspace(1.0, 0.0, k, dtype=np.float32)   # decays from the seam
    for blk in (LH, RH):
        if (a[-1, blk] == 0).all() or (b[0, blk] == 0).all():
            continue
        a_shape, b_shape = _hand_local(a[-1, blk]), _hand_local(b[0, blk])
        for i in range(k):                      # carry-over into B
            if (b[i, blk] == 0).all():
                continue
            b[i, blk] = b[i, blk][:1] + (1 - w[i]) * _hand_local(b[i, blk]) + w[i] * a_shape
        for i in range(k):                      # anticipation at the end of A
            j = len(a) - 1 - i
            if (a[j, blk] == 0).all():
                continue
            a[j, blk] = a[j, blk][:1] + (1 - w[i]) * _hand_local(a[j, blk]) + w[i] * b_shape
    return a, b


def transition(a_last: np.ndarray, b_first: np.ndarray, cfg: ComposeConfig,
               rng: np.random.Generator) -> np.ndarray:
    """Movement epenthesis: the travel between two signs.

    Length scales with how far the hands actually have to move (in shoulder-width
    units), which is why a decoder cannot use "there was motion" as evidence of a
    sign - a long transition looks exactly like a short sign.
    """
    sw = np.linalg.norm(b_first[L_SHOULDER] - b_first[R_SHOULDER])
    scale = max(float(sw), 1e-3)
    d = 0.0
    for wrist in (L_HAND_WRIST, R_HAND_WRIST):
        if not (a_last[wrist] == 0).all() and not (b_first[wrist] == 0).all():
            d = max(d, float(np.linalg.norm(a_last[wrist] - b_first[wrist]) / scale))
    lo, hi = cfg.trans_frames
    n = int(np.clip(round(lo + (hi - lo) * min(d / 1.6, 1.0)), lo, hi))
    n = max(n + int(rng.integers(-2, 3)), 2)

    p = _minjerk(n)[:, None, None]
    out = a_last[None] * (1 - p) + b_first[None] * p
    bad = (a_last == 0).all(-1) | (b_first == 0).all(-1)    # never blend through a gap
    out[:, bad] = b_first[None][:, bad]
    return out.astype(np.float32)


def hold(frame: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A prosodic hold: the posture is kept, but a real signer is never still."""
    out = np.repeat(frame[None], n, axis=0).astype(np.float32)
    jit = rng.normal(0, 0.0012, out.shape).astype(np.float32)
    jit[(out == 0).all(-1)] = 0.0
    return (out + jit).astype(np.float32)


def chimera(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A motion that is *not* any vocabulary sign: the first half of one sign
    glued to the second half of another.

    Positive training material for the `<unk>` class - a deployed reader meets
    signs outside its vocabulary far more often than clean citation forms inside it.
    """
    ha, hb = a[: max(3, len(a) // 2)], b[max(3, len(b) // 2):]
    mid = transition(ha[-1], hb[0], ComposeConfig(), rng)
    return np.concatenate([ha, mid, hb]).astype(np.float32)


# ------------------------------------------------------- camera effects ----
def mirror(clip: np.ndarray) -> np.ndarray:
    """Left-handed signer / mirrored camera: flip x, swap the two hand blocks and
    every left/right pose pair, so the anatomy stays consistent."""
    perm = np.arange(75)
    for i, j in _POSE_LR_PAIRS:
        perm[i], perm[j] = j, i
    perm[LH], perm[RH] = np.arange(54, 75), np.arange(33, 54)
    out = clip[:, perm].astype(np.float32).copy()     # zeros are carried along
    miss = (out == 0).all(-1)
    out[..., 0] = 1.0 - out[..., 0]
    out[miss] = 0.0
    return out


def camera_jitter(clip: np.ndarray, cfg: ComposeConfig,
                  rng: np.random.Generator):
    """Signer distance, framing, camera roll, sensor noise and MediaPipe drop-outs.

    Applied in *raw image coordinates*, i.e. before the body-frame normalisation,
    so it exercises exactly the part of the pipeline that a new recording stresses.
    Returns (clip, idx_map) where idx_map maps output frames back to input frames
    when the global tempo was changed (None otherwise).
    """
    out = clip.astype(np.float32).copy()
    miss = (out == 0).all(-1)

    neck, _ = _neck_scale(out)
    c = neck.mean(0)
    s = float(rng.uniform(*cfg.scale_range))
    # anisotropic component: MediaPipe normalises x and y by width and height
    # independently, so a recording whose frame aspect differs from the corpus
    # arrives geometrically distorted along one axis. `extract_from_video`
    # corrects the bulk of it; this covers the residual.
    a = float(rng.uniform(*cfg.aspect_range))
    sxy = np.array([s * a, s], dtype=np.float32)
    th = np.deg2rad(float(rng.uniform(-cfg.rot_deg, cfg.rot_deg)))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    t = rng.uniform(-cfg.shift, cfg.shift, 2).astype(np.float32)
    out = ((out - c) * sxy) @ R.T + c + t

    if cfg.coord_noise > 0:
        out = out + rng.normal(0, cfg.coord_noise, out.shape).astype(np.float32)

    idx_map = None
    f = float(rng.uniform(*cfg.fps_scale))
    if abs(f - 1.0) > 1e-3:
        T = len(out)
        n = max(16, int(round(T / f)))
        dst = np.linspace(0.0, T - 1.0, n, dtype=np.float32)
        i0 = np.floor(dst).astype(np.int64)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (dst - i0)[:, None]
        flat = out.reshape(T, -1)
        out = (flat[i0] * (1 - w) + flat[i1] * w).reshape(n, 75, 2)
        mf = miss.astype(np.float32)
        miss = ((mf[i0] * (1 - w) + mf[i1] * w) > 0.5)
        idx_map = dst

    out = out.astype(np.float32)
    out[miss] = 0.0

    # bursty hand loss, the way MediaPipe actually fails (a run, not a coin flip)
    if cfg.hand_drop_burst_p > 0:
        T = len(out)
        for blk in (LH, RH):
            p = 0
            while p < T:
                if rng.random() < cfg.hand_drop_burst_p:
                    n = int(rng.integers(*cfg.drop_len))
                    out[p:p + n, blk] = 0.0
                    p += n
                p += 12
    return out, idx_map


# ------------------------------------------------------------- compose -----
@dataclass
class Utterance:
    clip: np.ndarray                               # (T, 75, 2)
    labels: list = field(default_factory=list)     # int gloss labels, in order
    spans: list = field(default_factory=list)      # [(start, end)] per label
    source: list = field(default_factory=list)     # source clip index per label
    is_unk: list = field(default_factory=list)     # per label: was it a chimera?

    def __len__(self):
        return len(self.clip)

    @property
    def n_signs(self) -> int:
        return len(self.labels)


def compose(pool_clips: list, pool_labels: np.ndarray, rng: np.random.Generator,
            cfg: ComposeConfig, n_signs: int | None = None,
            picks: list | None = None, pool_unk: np.ndarray | None = None) -> Utterance:
    """Build one continuous utterance out of isolated clips.

    `picks` forces a specific sign sequence, used to build *paired* control sets
    where the reference gloss sequence is identical across continuity regimes -
    so a WER difference is attributable to the regime and nothing else.
    """
    if picks is None:
        k = n_signs or int(rng.integers(cfg.min_signs, cfg.max_signs + 1))
        picks = [int(i) for i in rng.choice(len(pool_clips), size=k, replace=False)]
    k = len(picks)

    units, labels, is_unk, srcs = [], [], [], []
    for p in picks:
        c = retime(pool_clips[p], float(rng.uniform(*cfg.speed)))
        if cfg.chimera_p > 0 and rng.random() < cfg.chimera_p:
            q = int(rng.integers(len(pool_clips)))
            c = chimera(c, retime(pool_clips[q], float(rng.uniform(*cfg.speed))), rng)
            is_unk.append(True)
        else:
            # a clip flagged in `pool_unk` is a genuine out-of-vocabulary sign:
            # real material for a gloss the model is not allowed to know
            is_unk.append(bool(pool_unk[p]) if pool_unk is not None else False)
        units.append(c)
        labels.append(int(pool_labels[p]))
        srcs.append(p)

    for i in range(k - 1):                          # coarticulate adjacent pairs
        units[i], units[i + 1] = coarticulate(units[i], units[i + 1], cfg, rng)

    parts, spans, t = [], [], 0
    lead = int(rng.integers(*cfg.lead_rest))
    if lead > 0:
        parts.append(hold(units[0][0], lead, rng)); t += lead
    for i, u in enumerate(units):
        parts.append(u)
        spans.append((t, t + len(u)))
        t += len(u)
        if i < k - 1:
            if rng.random() < cfg.hold_p:
                n = int(rng.integers(*cfg.hold_frames))
                parts.append(hold(u[-1], n, rng)); t += n
            tr = transition(parts[-1][-1], units[i + 1][0], cfg, rng)
            parts.append(tr); t += len(tr)
    tail = int(rng.integers(*cfg.tail_rest))
    if tail > 0:
        parts.append(hold(units[-1][-1], tail, rng)); t += tail

    clip = np.concatenate(parts).astype(np.float32)
    if cfg.mirror_p > 0 and rng.random() < cfg.mirror_p:
        clip = mirror(clip)
    clip, idx_map = camera_jitter(clip, cfg, rng)

    if idx_map is not None:                         # re-index spans through the tempo warp
        spans = [(int(np.searchsorted(idx_map, s)), int(np.searchsorted(idx_map, e)))
                 for s, e in spans]
    spans = [(max(0, s), min(len(clip), max(e, s + 1))) for s, e in spans]

    return Utterance(clip=clip, labels=labels, spans=spans, source=srcs, is_unk=is_unk)


def compose_many(pool_clips: list, pool_labels: np.ndarray, n: int, seed: int,
                 cfg: ComposeConfig, progress: bool = False,
                 pool_unk: np.ndarray | None = None) -> list:
    rng = np.random.default_rng(seed)
    it = range(n)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="composing utterances", unit="utt")
        except Exception:
            pass
    return [compose(pool_clips, pool_labels, rng, cfg, pool_unk=pool_unk) for _ in it]


def paired_regimes(pool_clips: list, pool_labels: np.ndarray, n: int, seed: int,
                   cfgs: dict) -> dict:
    """The *same* gloss sequences rendered under several continuity regimes."""
    rng = np.random.default_rng(seed)
    plans = [[int(i) for i in rng.choice(len(pool_clips),
                                         size=int(rng.integers(3, 7)), replace=False)]
             for _ in range(n)]
    out = {}
    for name, cfg in cfgs.items():
        r = np.random.default_rng(seed + 991)
        out[name] = [compose(pool_clips, pool_labels, r, cfg, picks=p) for p in plans]
    return out
