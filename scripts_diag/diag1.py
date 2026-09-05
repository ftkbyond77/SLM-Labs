# -*- coding: utf-8 -*-
"""Diagnostics for SLM Labs: split leakage, Stage A vs Stage B checkpoints, CTC blank, emb-head training."""
import os, sys, pickle, json
from pathlib import Path
import numpy as np, torch
ROOT = Path("D:/SLM-labs/SLM-Labs"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from slm_labs.config import cfg, DEVICE, seed_all
from slm_labs.vocab import N_CLASSES, SIGN_CLASSES, CLASS2WORD, BLANK, NULL_CLASS
from slm_labs.data import make_loader
from slm_labs.model import build_model
from slm_labs.train import load_checkpoint
from slm_labs.metrics import eval_isolated, eval_sequence

print("DEVICE:", DEVICE, torch.__version__)
iso, sent = pickle.load(open(cfg.OUT_DIR / "items_cache.pkl", "rb"))
print({k: len(v) for k, v in iso.items()}, {k: len(v) for k, v in sent.items()})

# ---------- D1: leakage ----------
print("\n===== D1  split leakage =====")
for a, b in [("train","val"),("train","test"),("val","test")]:
    ida = {(it["src"], it["id"]) for it in iso[a]}; idb = {(it["src"], it["id"]) for it in iso[b]}
    print(f"  iso id overlap {a}/{b}: {len(ida & idb)}")
pat = {k: {it["pattern"] for it in v} for k, v in sent.items()}
for a, b in [("train","val"),("train","test"),("val","test")]:
    print(f"  sent PATTERN overlap {a}/{b}: {len(pat[a] & pat[b])} / |{a}|={len(pat[a])} |{b}|={len(pat[b])}")
ids = {k: {it["id"] for it in v} for k, v in sent.items()}
for a, b in [("train","val"),("train","test"),("val","test")]:
    print(f"  sent id overlap {a}/{b}: {len(ids[a] & ids[b])}")
# duplicate feature check (exact array hash) across iso splits
def fh(it): return hash(it["feat"]["hand"][:5].tobytes())
h = {k: [fh(it) for it in v] for k, v in iso.items()}
print("  iso feat-hash overlap train/test:", len(set(h["train"]) & set(h["test"])))
print("  glosses per sentence (test):", np.mean([len(it["glosses"]) for it in sent["test"]]).round(2),
      "| frames per sentence (test):", np.mean([len(it["feat"]["hand"]) for it in sent["test"]]).round(1))

# ---------- D2/D3: Stage A vs Stage B checkpoints ----------
print("\n===== D2/D3  isolated test acc: after Stage A vs after Stage B =====")
dl_te = make_loader(iso["test"], cfg.MAX_FRAMES_ISO, False, False)
dl_seq_va = make_loader(sent["val"], cfg.MAX_FRAMES_SEQ, False, True, bs=16)
dl_seq_te = make_loader(sent["test"], cfg.MAX_FRAMES_SEQ, False, True, bs=16)
seed_all(cfg.SEED); fresh = build_model(N_CLASSES, DEVICE)
fresh_emb = {k: v.clone() for k, v in fresh.state_dict().items() if k.startswith("emb_head")}
for tag in ["v1_stageA_best", "v1_stageB_best", "v2_stageA_best", "v2_stageB_best"]:
    p = cfg.OUT_DIR / f"{tag}.pt"
    if not p.exists(): print("  missing", tag); continue
    m = build_model(N_CLASSES, DEVICE); load_checkpoint(m, p)
    r = eval_isolated(m, dl_te)
    # emb_head drift vs fresh init
    sd = m.state_dict()
    drift = max(float((sd[k].float() - v.float()).abs().max()) for k, v in fresh_emb.items())
    print(f"  {tag:16s} iso-test acc={r['acc']:.4f} F1={r['macro_f1']:.4f}   emb_head max|Δ| vs random init = {drift:.3e}")

# ---------- D4: CTC blank diagnostic on stage B ckpt ----------
print("\n===== D4  CTC diagnostic (v1_stageB_best on sentence val) =====")
m = build_model(N_CLASSES, DEVICE); load_checkpoint(m, cfg.OUT_DIR / "v1_stageB_best.pt"); m.eval()
x = next(iter(dl_seq_va))
with torch.no_grad():
    o = m(x["hand"].to(DEVICE), x["body"].to(DEVICE), x["face"].to(DEVICE), x["mask"].to(DEVICE))
L = m.ctc_lens(x["lens"]).tolist()
pr = o["logits_ctc"].float().softmax(-1).cpu().numpy()
print("  logits_ctc shape:", tuple(o["logits_ctc"].shape), "| ctc_lens:", L[:6], "| raw lens:", x["lens"].tolist()[:6])
i = 0
p0 = pr[i, :L[i]]
print(f"  clip0: T_ctc={L[i]}  mean P(blank)={p0[:,BLANK].mean():.4f}  min P(blank)={p0[:,BLANK].min():.4f}")
print("  target lens:", x["target_lens"].tolist()[:6])
tgt = x["targets"][:x["target_lens"][0]].tolist()
print("  clip0 ref glosses:", [CLASS2WORD[SIGN_CLASSES[t-1]] for t in tgt])
top = p0[:, 1:].max(1); topi = p0[:, 1:].argmax(1)
print("  clip0 per-frame best non-blank (word, prob):")
print("   ", [(CLASS2WORD[SIGN_CLASSES[c]], round(float(v),3)) for c, v in zip(topi, top)])
for bp in [0, 2, 4, 6, 8, 10, 14]:
    r = eval_sequence(m, dl_seq_va, blank_penalty=bp)
    print(f"  blank_penalty={bp:>4}: val WER={r['wer']:.3f} CER={r['cer']:.3f}  hyp0={r['hyps'][0][:8]}")
