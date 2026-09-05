"""Training loops

Stage A : isolated sign → CE(cls_head) + ArcFace(emb_head)     ← emb_head ถูกฝึกจริง (v1/v2 ไม่เคยฝึก)
Stage B : sentence → CTC(ctc_head) โดย **freeze encoder**       ← Stage A ไม่มีทางพัง (v2 เคยตก 0.988 → 0.337)
"""
from __future__ import annotations

import math
import time

import pandas as pd
import torch
import torch.nn as nn

from .config import cfg, DEVICE, to_dev
from .metrics import eval_isolated, eval_sequence
from .vocab import BLANK

use_amp = DEVICE == "cuda"


def make_optim(params, epochs, steps_per_epoch, lr=None):
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr or cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.98))
    total = max(1, epochs * steps_per_epoch); warm = max(1, int(0.1 * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    return opt, sch, params


def _step(opt, sch, scaler, loss, params):
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sch.step()


def train_stage_a(model, dl_tr, dl_va, epochs=cfg.EPOCHS_A, tag="stageA", log_every=5, emb_w=None):
    """CE บน cls_head + ArcFace บน emb_head (128-d, L2)

    ArcFace คือส่วนที่ทำให้ cosine กับ prototype มีความหมาย → open-set / memory / retrieval ใช้ได้จริง
    ตั้ง emb_w=0 เพื่อกลับไปเป็นพฤติกรรมเดิมของ v1/v2 (ablation)
    """
    emb_w = cfg.EMB_LOSS_W if emb_w is None else emb_w
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    ce_arc = nn.CrossEntropyLoss()
    opt, sch, params = make_optim(model.parameters(), epochs, len(dl_tr)); scaler = torch.amp.GradScaler(enabled=use_amp)
    best, hist, ckpt = -1, [], cfg.OUT_DIR / f"{tag}_best.pt"
    for ep in range(1, epochs + 1):
        model.train(); tot, tot_a, t0 = 0.0, 0.0, time.time()
        for x in dl_tr:
            x = to_dev(x)
            with torch.autocast(DEVICE, enabled=use_amp):
                o = model.forward_batch(x)
                l_cls = ce(o["logits_cls"], x["targets"])
            l_arc = ce_arc(model.arc(o["emb"], x["targets"]), x["targets"]) if emb_w else torch.zeros((), device=DEVICE)
            loss = l_cls + emb_w * l_arc
            _step(opt, sch, scaler, loss, params); tot += float(l_cls); tot_a += float(l_arc)
        va = eval_isolated(model, dl_va)
        hist.append((ep, tot / len(dl_tr), tot_a / len(dl_tr), va["acc"], va["macro_f1"], time.time() - t0))
        if va["macro_f1"] > best:
            best = va["macro_f1"]; torch.save(model.state_dict(), ckpt)
        if ep % log_every == 0 or ep == 1:
            print(f"[A:{tag}] ep{ep:3d} ce={tot / len(dl_tr):.3f} arc={tot_a / len(dl_tr):.3f}  "
                  f"val acc={va['acc']:.3f} macroF1={va['macro_f1']:.3f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return pd.DataFrame(hist, columns=["ep", "ce", "arc", "val_acc", "val_f1", "sec"])


def train_stage_b(model, dl_seq_tr, dl_seq_va, epochs=cfg.EPOCHS_B, tag="stageB", log_every=5, lr=None,
                  freeze_encoder=None):
    """CTC บน ctc_head

    freeze_encoder=True (default): encoder + cls_head + emb_head ถูก freeze และรันใน eval mode
    → Stage B เป็น "หัวอ่านลำดับ" ที่ต่อบน Stage A ล้วน ๆ ความแม่นของ isolated ไม่เปลี่ยนแม้แต่นิดเดียว
    """
    freeze_encoder = cfg.FREEZE_ENCODER_STAGE_B if freeze_encoder is None else freeze_encoder
    if freeze_encoder:
        for p in model.encoder_parameters():
            p.requires_grad = False
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    lr = lr or cfg.LR_STAGE_B
    opt, sch, params = make_optim(model.parameters(), epochs, len(dl_seq_tr), lr=lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, hist, ckpt = (1e9, 1e9), [], cfg.OUT_DIR / f"{tag}_best.pt"
    print(f"[B:{tag}] trainable params: {sum(p.numel() for p in params)/1e3:.1f}k "
          f"({'ctc_head only — encoder frozen' if freeze_encoder else 'full model'})")
    for ep in range(1, epochs + 1):
        model.eval() if freeze_encoder else model.train()
        if freeze_encoder:
            model.ctc_head.train()
        tot, t0 = 0.0, time.time()
        for x in dl_seq_tr:
            x = to_dev(x)
            with torch.autocast(DEVICE, enabled=use_amp):
                o = model.forward_batch(x)
                lp = o["logits_ctc"].float().log_softmax(-1).transpose(0, 1)
                loss = ctc(lp, x["targets"], model.ctc_lens(x["lens"]), x["target_lens"])
            _step(opt, sch, scaler, loss, params); tot += float(loss)
        va = eval_sequence(model, dl_seq_va)
        hist.append((ep, tot / len(dl_seq_tr), va["wer"], va["cer"], va["sent_acc"], time.time() - t0))
        key = (round(va["wer"], 4), tot / len(dl_seq_tr))
        if key < best:
            best = key; torch.save(model.state_dict(), ckpt)
        if ep % log_every == 0 or ep == 1:
            print(f"[B:{tag}] ep{ep:3d} ctc={tot / len(dl_seq_tr):.3f}  val WER={va['wer']:.3f} CER={va['cer']:.3f} "
                  f"sentAcc={va['sent_acc']:.3f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    if freeze_encoder:
        for p in model.encoder_parameters():
            p.requires_grad = True
    return pd.DataFrame(hist, columns=["ep", "ctc", "val_wer", "val_cer", "val_sent_acc", "sec"])


def save_checkpoint(model, path, extra: dict | None = None):
    from .vocab import SIGN_CLASSES
    torch.save({"state_dict": model.state_dict(), "classes": SIGN_CLASSES, "cfg": cfg.to_dict(), **(extra or {})}, path)


def load_checkpoint(model, path, restore_calibration=True):
    """โหลด weights + (ถ้ามี) ค่าที่ calibrate ไว้ตอน train: blank penalty, segmentation params, SEG_CLS_CONF

    สำคัญ: threshold เหล่านี้เป็นส่วนหนึ่งของโมเดล ไม่ใช่ default ใน config — ถ้าไม่โหลดกลับมาด้วย
    การ inference จะใช้ค่าที่ยังไม่ได้ tune แล้วได้ผลไม่ตรงกับที่รายงานไว้
    """
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    if restore_calibration and isinstance(ck, dict):
        if "blank_penalty" in ck:
            cfg.BLANK_PENALTY = float(ck["blank_penalty"])
        if "seg_cls_conf" in ck:
            cfg.SEG_CLS_CONF = float(ck["seg_cls_conf"])
        if "seg_params" in ck:
            from .openset import SEG_PARAMS
            SEG_PARAMS.update(ck["seg_params"])
    return model
