"""Training loops: SSL (SignDINO) → Stage A (isolated CE) → Stage B (CTC + CE)"""
from __future__ import annotations

import math
import time

import pandas as pd
import torch
import torch.nn as nn

from .config import cfg, DEVICE, to_dev
from .model import SignDINO
from .metrics import eval_isolated, eval_sequence
from .vocab import BLANK

use_amp = DEVICE == "cuda"


def make_optim(params, epochs, steps_per_epoch, lr=None):
    opt = torch.optim.AdamW(params, lr=lr or cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.98))
    total = max(1, epochs * steps_per_epoch); warm = max(1, int(0.1 * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    return opt, sch


def _step(opt, sch, scaler, loss, params):
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sch.step()


def train_ssl(model, dl_views, epochs=cfg.SSL_EPOCHS, tag="ssl", log_every=1):
    """SignDINO-style: student เห็น global+local views, teacher (EMA) เห็นแค่ global → บังคับให้ local view ของ "บางคำ"
    map ไปที่ representation เดียวกับ global view ของทั้งคลิป (ไม่ใช้ label)"""
    dino = SignDINO(model).to(DEVICE)
    params = list(dino.student.parameters()) + [p for p in dino.s_head.parameters() if p.requires_grad]
    opt, sch = make_optim(params, epochs, len(dl_views), lr=cfg.SSL_LR)
    scaler = torch.amp.GradScaler(enabled=use_amp); hist = []
    for ep in range(1, epochs + 1):
        dino.train(); tot, t0 = 0.0, time.time()
        for views in dl_views:
            views = [to_dev(v) for v in views]
            with torch.autocast(DEVICE, enabled=use_amp):
                loss = dino(views)
            _step(opt, sch, scaler, loss, params); dino.ema_update(); tot += loss.item()
        hist.append((ep, tot / len(dl_views), time.time() - t0))
        if ep % log_every == 0 or ep == 1:
            print(f"[SSL] ep{ep:3d} dino_loss={tot / len(dl_views):.4f}  ({time.time() - t0:.0f}s)")
    torch.save(dino.student.state_dict(), cfg.OUT_DIR / f"{tag}_encoder.pt")
    return pd.DataFrame(hist, columns=["ep", "dino_loss", "sec"])


def train_stage_a(model, dl_tr, dl_va, epochs=cfg.EPOCHS_A, tag="stageA", log_every=5):
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    opt, sch = make_optim(model.parameters(), epochs, len(dl_tr)); scaler = torch.amp.GradScaler(enabled=use_amp)
    best, hist, ckpt = -1, [], cfg.OUT_DIR / f"{tag}_best.pt"
    for ep in range(1, epochs + 1):
        model.train(); tot, t0 = 0.0, time.time()
        for x in dl_tr:
            x = to_dev(x)
            with torch.autocast(DEVICE, enabled=use_amp):
                o = model.forward_batch(x); loss = ce(o["logits_cls"], x["targets"])
            _step(opt, sch, scaler, loss, model.parameters()); tot += loss.item()
        va = eval_isolated(model, dl_va); hist.append((ep, tot / len(dl_tr), va["acc"], va["macro_f1"], time.time() - t0))
        if va["macro_f1"] > best:
            best = va["macro_f1"]; torch.save(model.state_dict(), ckpt)
        if ep % log_every == 0 or ep == 1:
            print(f"[A:{tag}] ep{ep:3d} loss={tot / len(dl_tr):.3f}  val acc={va['acc']:.3f}  macroF1={va['macro_f1']:.3f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return pd.DataFrame(hist, columns=["ep", "loss", "val_acc", "val_f1", "sec"])


def train_stage_b(model, dl_seq_tr, dl_seq_va, dl_iso_tr, epochs=cfg.EPOCHS_B, tag="stageB", log_every=5):
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH); ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt, sch = make_optim(model.parameters(), epochs, len(dl_seq_tr)); scaler = torch.amp.GradScaler(enabled=use_amp)
    best, hist, ckpt = 1e9, [], cfg.OUT_DIR / f"{tag}_best.pt"; iso_iter = iter(dl_iso_tr)
    for ep in range(1, epochs + 1):
        model.train(); tot, t0 = 0.0, time.time()
        for x in dl_seq_tr:
            x = to_dev(x)
            try:
                xi = to_dev(next(iso_iter))
            except StopIteration:
                iso_iter = iter(dl_iso_tr); xi = to_dev(next(iso_iter))
            with torch.autocast(DEVICE, enabled=use_amp):
                o = model.forward_batch(x)
                lp = o["logits_ctc"].float().log_softmax(-1).transpose(0, 1)
                l_ctc = ctc(lp, x["targets"], x["lens"], x["target_lens"])
                oi = model.forward_batch(xi); l_ce = ce(oi["logits_cls"], xi["targets"])
                loss = l_ctc + cfg.CE_WEIGHT_STAGE_B * l_ce
            _step(opt, sch, scaler, loss, model.parameters()); tot += l_ctc.item()
        va = eval_sequence(model, dl_seq_va); hist.append((ep, tot / len(dl_seq_tr), va["wer"], va["cer"], va["sent_acc"], time.time() - t0))
        if va["wer"] < best:
            best = va["wer"]; torch.save(model.state_dict(), ckpt)
        if ep % log_every == 0 or ep == 1:
            print(f"[B:{tag}] ep{ep:3d} ctc={tot / len(dl_seq_tr):.3f}  val WER={va['wer']:.3f}  CER={va['cer']:.3f}  sentAcc={va['sent_acc']:.3f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return pd.DataFrame(hist, columns=["ep", "ctc", "val_wer", "val_cer", "val_sent_acc", "sec"])


def save_checkpoint(model, path, extra: dict | None = None):
    from .vocab import SIGN_CLASSES
    torch.save({"state_dict": model.state_dict(), "classes": SIGN_CLASSES, "cfg": cfg.to_dict(), **(extra or {})}, path)


def load_checkpoint(model, path):
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    return model
