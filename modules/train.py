"""Training loop: AMP, cosine schedule with warm-up, early stopping on val F1.

The best weights are kept in RAM (a state_dict copy); nothing touches disk until
the notebook explicitly saves the single production checkpoint.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from .config import AugConfig, TrainConfig
from .dataset import StreamBank, augment, mixup
from .models import soft_cross_entropy


def _lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))


@torch.no_grad()
def predict(model, bank: StreamBank, idx: np.ndarray, batch_size: int = 256,
            want_embed: bool = False):
    model.eval()
    logits, embeds, ys = [], [], []
    for xb, yb in bank.iter_batches(idx, batch_size, shuffle=False):
        out = model(xb)
        logits.append(out["logits"].float().cpu())
        if want_embed:
            embeds.append(out["embed"].float().cpu())
        ys.append(yb.cpu())
    L = torch.cat(logits).numpy()
    Y = torch.cat(ys).numpy()
    E = torch.cat(embeds).numpy() if want_embed else None
    return L, Y, E


def fit(model, bank: StreamBank, train_idx, val_idx, n_classes: int,
        tcfg: TrainConfig, acfg: AugConfig, mixup_alpha: float = 0.2,
        use_arc: bool = True, verbose: bool = True, tag: str = "model"):
    device = bank.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.amp and device.type == "cuda")
    rng = np.random.default_rng(tcfg.seed)
    gen = torch.Generator(device=device); gen.manual_seed(tcfg.seed)

    steps_per_epoch = int(np.ceil(len(train_idx) / tcfg.batch_size))
    total_steps = steps_per_epoch * tcfg.epochs
    warmup_steps = steps_per_epoch * tcfg.warmup_epochs

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
            "val_f1": [], "lr": [], "sec": []}
    best = {"f1": -1.0, "epoch": -1, "state": None}
    step, bad = 0, 0

    for ep in range(tcfg.epochs):
        t0 = time.time()
        model.train()
        run, seen = 0.0, 0
        for xb, yb in bank.iter_batches(train_idx, tcfg.batch_size, True, rng):
            lr = _lr_at(step, total_steps, warmup_steps, tcfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            xb = augment(xb, acfg, gen)
            xb, yt = mixup(xb, yb, n_classes, mixup_alpha, gen)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(xb, target=yt if use_arc else None)
                loss = soft_cross_entropy(out["logits"], yt, tcfg.label_smoothing)
                if use_arc and out.get("arc") is not None:
                    loss = loss + tcfg.embed_loss_w * soft_cross_entropy(out["arc"], yt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            run += float(loss) * yb.shape[0]
            seen += yb.shape[0]
            step += 1

        L, Y, _ = predict(model, bank, val_idx)
        vloss = float(torch.nn.functional.cross_entropy(
            torch.from_numpy(L), torch.from_numpy(Y)))
        pred = L.argmax(1)
        acc = float((pred == Y).mean())
        f1 = float(f1_score(Y, pred, average="macro", zero_division=0))

        hist["epoch"].append(ep); hist["train_loss"].append(run / max(seen, 1))
        hist["val_loss"].append(vloss); hist["val_acc"].append(acc)
        hist["val_f1"].append(f1); hist["lr"].append(lr)
        hist["sec"].append(time.time() - t0)

        if f1 > best["f1"]:
            best = {"f1": f1, "acc": acc, "epoch": ep,
                    "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == tcfg.epochs - 1 or bad == 0):
            print(f"[{tag}] ep {ep:3d} | train {run/max(seen,1):.3f} | "
                  f"val {vloss:.3f} | acc {acc:.4f} | macroF1 {f1:.4f} | "
                  f"{time.time()-t0:.1f}s{'  *' if bad == 0 else ''}")
        if bad >= tcfg.patience:
            if verbose:
                print(f"[{tag}] early stop at epoch {ep} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    return model, hist, best
