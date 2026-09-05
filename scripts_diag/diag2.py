# -*- coding: utf-8 -*-
"""D5: real-video domain gap. Feature stats dataset vs data_test + Stage-A classification of real segments."""
import os, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path("D:/SLM-labs/SLM-Labs"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from slm_labs.config import cfg, DEVICE, to_dev
from slm_labs.vocab import N_CLASSES, SIGN_CLASSES, CLASS2WORD
from slm_labs.features import load_landmarks, make_features, hand_activity, temporal_resample
from slm_labs.data import pad_feats
from slm_labs.openset import segment_timeline
from slm_labs.model import build_model
from slm_labs.train import load_checkpoint

iso, sent = pickle.load(open(cfg.OUT_DIR / "items_cache.pkl", "rb"))
print("iso frames  : mean %.1f  p10 %.0f  p90 %.0f" % tuple(np.percentile([len(i["feat"]["hand"]) for i in iso["train"]], [50,10,90])[[0,1,2]]) if False else "")
L = [len(i["feat"]["hand"]) for i in iso["train"]]
print("iso train frames: mean=%.1f median=%.0f p10=%.0f p90=%.0f max=%d" % (np.mean(L), np.median(L), np.percentile(L,10), np.percentile(L,90), max(L)))

def stats(feat, name):
    h, b, f = feat["hand"], feat["body"], feat["face"]
    print(f"  {name:22s} T={len(h):4d} hand|xyz| p50={np.abs(h[:,:126]).mean():.3f} vel p50={np.abs(h[:,126:252]).mean():.4f} "
          f"pres_lh={h[:,252].mean():.2f} pres_rh={h[:,253].mean():.2f} body|xyz|={np.abs(b[:,:18]).mean():.3f} face|xyz|={np.abs(f[:,:18]).mean():.3f}")

print("\n===== D5a  feature stats: dataset vs real videos =====")
for it in iso["train"][:3]: stats(it["feat"], "dataset iso " + CLASS2WORD[SIGN_CLASSES[it["label"]]])
for it in sent["train"][:2]: stats(it["feat"], "dataset sent")
real = {}
for p in sorted((cfg.OUT_DIR / "landmarks_cache").glob("*.csv")):
    lm = load_landmarks(p); real[p.stem] = make_features(lm); stats(real[p.stem], "REAL " + p.stem[:14])

# aggregate distribution comparison
def agg(feats, key, sl):
    return np.concatenate([f[key][:, sl] for f in feats], 0)
ds = [it["feat"] for it in iso["train"][:200]]
print("\n  dataset hand-xyz  mean=%.4f std=%.4f  | real hand-xyz mean=%.4f std=%.4f" % (
    agg(ds,"hand",slice(0,126)).mean(), agg(ds,"hand",slice(0,126)).std(),
    agg(list(real.values()),"hand",slice(0,126)).mean(), agg(list(real.values()),"hand",slice(0,126)).std()))
print("  dataset hand-vel  mean=%.4f std=%.4f  | real hand-vel mean=%.4f std=%.4f" % (
    agg(ds,"hand",slice(126,252)).mean(), agg(ds,"hand",slice(126,252)).std(),
    agg(list(real.values()),"hand",slice(126,252)).mean(), agg(list(real.values()),"hand",slice(126,252)).std()))
print("  dataset body      mean=%.4f std=%.4f  | real body     mean=%.4f std=%.4f" % (
    agg(ds,"body",slice(0,51)).mean(), agg(ds,"body",slice(0,51)).std(),
    agg(list(real.values()),"body",slice(0,51)).mean(), agg(list(real.values()),"body",slice(0,51)).std()))
print("  dataset hand-pres mean=%.3f | real hand-pres mean=%.3f" % (
    agg(ds,"hand",slice(252,254)).mean(), agg(list(real.values()),"hand",slice(252,254)).mean()))

# ---------- D5b: Stage-A classifier on real segments ----------
print("\n===== D5b  Stage-A (v1_stageA_best, iso-test 0.976) applied to real-video segments =====")
m = build_model(N_CLASSES, DEVICE); load_checkpoint(m, cfg.OUT_DIR / "v1_stageA_best.pt"); m.eval()
EXPECTED = {"ไปทานข้าวด้วยกันมั้ย": ["ไป","(ทาน)","ข้าว","ด้วยกัน","(มั้ย)"], "ฉันปลอบเพื่อนร้องไห้": ["ฉัน","(ปลอบ)","(เพื่อน)","(ร้องไห้)"]}
for clip, feat in real.items():
    segs = segment_timeline(feat)
    print(f"\n  {clip}  T={len(feat['hand'])}  segments={len(segs)}  expected~{EXPECTED.get(clip)}")
    crops = [{k: v[s:e] for k, v in feat.items()} for s, e in segs]
    with torch.no_grad():
        o = m.forward_batch(to_dev(pad_feats(crops)))
    pr = o["logits_cls"].softmax(-1).cpu().numpy()
    for (s, e), p in zip(segs, pr):
        top3 = np.argsort(-p)[:3]
        print(f"    [{s:3d}-{e:3d}] ({s/10:.1f}-{e/10:.1f}s) " + "  ".join(f"{CLASS2WORD[SIGN_CLASSES[c]]}={p[c]:.3f}" for c in top3))
    # whole-clip
    with torch.no_grad():
        o = m.forward_batch(to_dev(pad_feats([feat])))
    p = o["logits_cls"].softmax(-1).cpu().numpy()[0]; top3 = np.argsort(-p)[:3]
    print("    whole clip:", "  ".join(f"{CLASS2WORD[SIGN_CLASSES[c]]}={p[c]:.3f}" for c in top3))

# sanity: same model on dataset isolated test segments (upper bound of confidence)
print("\n  reference: Stage-A confidence on dataset isolated TEST clips")
crops = [it["feat"] if len(it["feat"]["hand"]) <= cfg.MAX_FRAMES_ISO else temporal_resample(it["feat"], cfg.MAX_FRAMES_ISO) for it in iso["test"][:32]]
with torch.no_grad():
    o = m.forward_batch(to_dev(pad_feats(crops)))
p = o["logits_cls"].softmax(-1).cpu().numpy()
print("    mean top-1 prob = %.3f  (median %.3f)" % (p.max(1).mean(), np.median(p.max(1))))
