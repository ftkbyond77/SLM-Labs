"""Utterance batching and the three-stage curriculum that trains CSR-Net.

Curriculum
----------
    Stage 1  memorisation   (v1, already done)  word clips -> MSSF-Net
    Stage 2  composition    frontend FROZEN     utterances -> sequence trunk + CTC
    Stage 3  sentence       everything unfrozen, harder distribution, low LR

Stage 2 exists because the visual encoder is the expensive thing to learn and we
already have it: 3 302 clean word clips taught it what a Thai sign looks like.
Letting CTC gradients loose on it from step one would wash that out on a corpus
of synthetic sentences. Freezing it first forces the *sequence* trunk to do the
sequence job, and only then is the encoder allowed to adapt to what signs look
like when they are run together.

Utterances are synthesised, not stored: `compose_pool` builds a fresh pool from
the word clips of one partition, so no signer and no clip ever crosses the
signer-independent split, and the pool can be refreshed mid-training to keep the
model from memorising particular sentences.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .composer import ComposeConfig, compose_many
from .config import FeatureConfig
from .evaluate import sequence_metrics
from .feature_engineering import clip_to_streams
from .seq_model import collapse_runs, ctc_loss, frame_targets, greedy_ctc

STREAMS = ("lh", "rh", "pose", "face")


# ------------------------------------------------------------------ pool ---
class UtterancePool:
    """Composed utterances, featurised once and held as float16.

    float16 halves a 1 GB pool with no measurable effect: the features are
    z-scored, so their dynamic range is tiny compared with fp16's, and they are
    cast back to fp32 the moment a batch is built.
    """

    def __init__(self, utts: list, stats: dict, fcfg: FeatureConfig,
                 max_len: int = 560, progress: bool = False):
        self.fcfg = fcfg
        self.utts, self.feats = [], []
        it = utts
        if progress:
            try:
                from tqdm.auto import tqdm
                it = tqdm(utts, desc="featurising utterances", unit="utt")
            except Exception:
                pass
        for u in it:
            if len(u) > max_len:
                continue
            s = clip_to_streams(u.clip, fcfg, resample=False)
            self.feats.append({k: ((s[k] - stats[k][0]) / stats[k][1]).astype(np.float16)
                               for k in STREAMS})
            self.utts.append(u)
        self.lens = np.array([f["lh"].shape[0] for f in self.feats], dtype=np.int64)

    def __len__(self):
        return len(self.utts)

    @property
    def dims(self) -> dict:
        return {k: int(v.shape[-1]) for k, v in self.feats[0].items()}

    def nbytes(self) -> int:
        return sum(sum(v.nbytes for v in f.values()) for f in self.feats)

    def references(self) -> list:
        return [[str(l) for l in u.labels] for u in self.utts]

    # ---- batching -----------------------------------------------------
    def _order(self, rng, shuffle: bool, batch_size: int):
        """Length-bucketed batches: sort by length, cut into batches, then
        shuffle the *batch order*. Padding waste drops to a few percent while the
        model still sees a random batch sequence."""
        idx = np.argsort(self.lens, kind="stable")
        if shuffle:
            jitter = rng.normal(0, 6, len(idx))
            idx = idx[np.argsort(self.lens[idx] + jitter, kind="stable")]
        batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
        if shuffle:
            rng.shuffle(batches)
        return batches

    def batch(self, ids: np.ndarray, device, model=None, factor: int = 4):
        T = int(self.lens[ids].max())
        B = len(ids)
        xb, mask = {}, np.zeros((B, T), dtype=bool)
        for k in STREAMS:
            D = self.feats[ids[0]][k].shape[-1]
            a = np.zeros((B, T, D), dtype=np.float32)
            for r, i in enumerate(ids):
                f = self.feats[i][k]
                a[r, : len(f)] = f
            xb[k] = torch.from_numpy(a).to(device, non_blocking=True)
        for r, i in enumerate(ids):
            mask[r, : self.lens[i]] = True
        m = torch.from_numpy(mask).to(device)

        targets = [np.asarray(self.utts[i].labels, dtype=np.int64) for i in ids]
        T4 = int(np.ceil(T / factor))
        blank = model.blank_id if model is not None else 0
        unk = model.unk_id if (model is not None and model.use_unk) else None
        fy = np.stack([frame_targets(self.utts[i].spans, self.utts[i].labels, T4,
                                     factor, blank, self.utts[i].is_unk, unk)
                       for i in ids])
        by = (fy != blank).astype(np.float32)
        return (xb, m, targets,
                torch.from_numpy(fy).to(device),
                torch.from_numpy(by).to(device))

    def iter_batches(self, batch_size: int, device, model, shuffle: bool,
                     rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        for ids in self._order(rng, shuffle, batch_size):
            yield self.batch(ids, device, model)


def compose_pool(clips: list, labels: np.ndarray, n: int, seed: int,
                 cfg: ComposeConfig, stats: dict, fcfg: FeatureConfig,
                 progress: bool = True, unk: np.ndarray | None = None) -> UtterancePool:
    utts = compose_many(clips, labels, n, seed, cfg, progress=progress, pool_unk=unk)
    return UtterancePool(utts, stats, fcfg, progress=progress)


# -------------------------------------------------------------- training ---
@dataclass
class StageConfig:
    name: str = "stage2"
    epochs: int = 40
    lr: float = 3e-4
    batch_size: int = 8
    freeze_frontend: bool = True
    w_frame: float = 0.5           # frame-mapping cross-entropy
    w_bnd: float = 0.2             # sign-activity BCE
    label_smoothing: float = 0.05
    weight_decay: float = 0.02
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    patience: int = 12
    refresh_every: int = 0         # regenerate the training pool every N epochs (0 = never)
    amp: bool = True


def _lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))


def _tok_name(model, tok: int, class_names):
    if model.use_unk and tok == model.unk_id:
        return "_"
    return class_names[tok] if class_names else str(tok)


@torch.no_grad()
def decode_pool(model, pool: UtterancePool, device, batch_size: int = 8,
                class_names: list | None = None, head: str = "ctc",
                min_run: int = 2):
    """Whole pool -> hypothesis gloss sequences.

    Two read-outs are available and both are reported, because they fail
    differently:

    * ``head="ctc"``   - the alignment-free path. Spiky: it emits one peak per
      sign and is the only one of the two that is trained by a *sequence* loss,
      so it is the honest measure of the sequence model.
    * ``head="frame"`` - argmax of the frame head, runs collapsed. It inherits
      the composer's dense frame supervision, so it converges far earlier and it
      also hands back contiguous spans, which is what a person needs to see when
      checking a decode against a video.
    """
    model.eval()
    hyps, order = [], []
    rng = np.random.default_rng(0)
    for ids in pool._order(rng, False, batch_size):
        xb, m, _, _, _ = pool.batch(ids, device, model)
        out = model(xb, m)
        m4 = out["mask4"].cpu().numpy()
        if head == "ctc":
            logp = F.log_softmax(out["ctc"].float(), -1).cpu().numpy()
        else:
            fr = out["frame"].float().argmax(-1).cpu().numpy()
        for r, i in enumerate(ids):
            if head == "ctc":
                toks = greedy_ctc(logp[r], model.blank_id, m4[r])
            else:
                toks = collapse_runs(fr[r][m4[r]], model.blank_id, min_run)
            hyps.append([_tok_name(model, t["token"], class_names) for t in toks])
            order.append(int(i))
    inv = np.argsort(order)
    return [hyps[j] for j in inv]


@torch.no_grad()
def evaluate_pool(model, pool: UtterancePool, device, class_names: list,
                  batch_size: int = 8, head: str = "ctc") -> dict:
    hyps = decode_pool(model, pool, device, batch_size, class_names, head)
    # an out-of-vocabulary sign's reference IS "_": the correct answer for a sign
    # the model was never allowed to learn is "I do not know this one"
    refs = [["_" if u.is_unk[i] else class_names[l] for i, l in enumerate(u.labels)]
            for u in pool.utts]
    m = sequence_metrics(refs, hyps)
    m["mean_hyp_len"] = float(np.mean([len(h) for h in hyps]))
    m["mean_ref_len"] = float(np.mean([len(r) for r in refs]))
    return m


def fit_stage(model, pool_tr: UtterancePool, pool_va: UtterancePool, device,
              scfg: StageConfig, class_names: list,
              refresh_fn=None, verbose: bool = True, best_wer0: float = 1e9):
    """One curriculum stage. Selection is on validation WER, never on test.

    `best_wer0` carries the previous stage's best validation WER forward, so
    model selection is global across the curriculum. Without it a later stage
    that fails to improve would still overwrite the weights with its own local
    best - which is how a curriculum quietly ends up shipping a worse model than
    the one it started from.
    """
    model.to(device).freeze_frontend(scfg.freeze_frontend)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=scfg.lr, weight_decay=scfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=scfg.amp and device.type == "cuda")
    rng = np.random.default_rng(42)

    spe = int(np.ceil(len(pool_tr) / scfg.batch_size))
    total, warm = spe * scfg.epochs, spe * scfg.warmup_epochs
    hist = {"epoch": [], "loss": [], "ctc": [], "frame": [], "val_wer": [],
            "val_wer_frame": [], "val_exact": [], "lr": [], "sec": []}
    best = {"wer": best_wer0, "epoch": -1, "state": None}
    step, bad = 0, 0

    for ep in range(scfg.epochs):
        if refresh_fn is not None and scfg.refresh_every and ep and \
                ep % scfg.refresh_every == 0:
            pool_tr = refresh_fn(ep)
            if verbose:
                print(f"[{scfg.name}] pool refreshed -> {len(pool_tr)} utterances")
        t0 = time.time()
        model.train()
        agg = {"loss": 0.0, "ctc": 0.0, "frame": 0.0, "n": 0}
        for xb, m, targets, fy, by in pool_tr.iter_batches(
                scfg.batch_size, device, model, True, rng):
            lr = _lr_at(step, total, warm, scfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(xb, m)
                T4 = out["ctc"].shape[1]
                l_ctc = ctc_loss(out["ctc"], out["mask4"], targets, model.blank_id)
                fyc, byc, m4 = fy[:, :T4], by[:, :T4], out["mask4"]
                l_frm = F.cross_entropy(
                    out["frame"].reshape(-1, model.n_out)[m4.reshape(-1)],
                    fyc.reshape(-1)[m4.reshape(-1)],
                    label_smoothing=scfg.label_smoothing)
                l_bnd = F.binary_cross_entropy_with_logits(
                    out["bnd"][m4], byc[m4])
                loss = l_ctc + scfg.w_frame * l_frm + scfg.w_bnd * l_bnd
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, scfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            b = len(targets)
            agg["loss"] += float(loss) * b; agg["ctc"] += float(l_ctc) * b
            agg["frame"] += float(l_frm) * b; agg["n"] += b
            step += 1

        vm = evaluate_pool(model, pool_va, device, class_names, scfg.batch_size, "ctc")
        vf = evaluate_pool(model, pool_va, device, class_names, scfg.batch_size, "frame")
        vm["wer_frame"] = vf["wer"]
        vm["exact_frame"] = vf["sentence_exact_match"]
        vm["hyp_len_frame"] = vf["mean_hyp_len"]
        n = max(agg["n"], 1)
        hist["epoch"].append(ep); hist["loss"].append(agg["loss"] / n)
        hist["ctc"].append(agg["ctc"] / n); hist["frame"].append(agg["frame"] / n)
        hist["val_wer"].append(vm["wer"]); hist["val_wer_frame"].append(vm["wer_frame"])
        hist["val_exact"].append(vm["sentence_exact_match"])
        hist["lr"].append(lr); hist["sec"].append(time.time() - t0)

        sel = min(vm["wer"], vm["wer_frame"])
        if sel < best["wer"] - 1e-4:
            best = {"wer": sel, "wer_ctc": vm["wer"], "wer_frame": vm["wer_frame"],
                    "epoch": ep,
                    "state": copy.deepcopy(model.state_dict()), "metrics": vm}
            bad = 0
        else:
            bad += 1
        if verbose and (ep % 2 == 0 or bad == 0 or ep == scfg.epochs - 1):
            print(f"[{scfg.name}] ep {ep:3d} | loss {agg['loss']/n:.3f} "
                  f"(ctc {agg['ctc']/n:.3f} frm {agg['frame']/n:.3f}) | "
                  f"val WER ctc {vm['wer']:.4f} / frame {vm['wer_frame']:.4f} | "
                  f"exact {vm['sentence_exact_match']:.3f} | "
                  f"len {vm['mean_hyp_len']:.1f}|{vm['hyp_len_frame']:.1f}/{vm['mean_ref_len']:.1f} | "
                  f"{time.time()-t0:.1f}s{'  *' if bad == 0 else ''}")
        if bad >= scfg.patience:
            if verbose:
                print(f"[{scfg.name}] early stop at {ep} (best {best['epoch']}, "
                      f"WER {best['wer']:.4f})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    elif verbose:
        print(f"[{scfg.name}] no epoch beat the incoming best "
              f"({best_wer0:.4f}); weights left unchanged")
    best["improved"] = best["state"] is not None
    return model, hist, best
