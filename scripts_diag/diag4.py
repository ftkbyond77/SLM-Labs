# -*- coding: utf-8 -*-
"""D7: dataset native frame-rate vs cfg.TARGET_FPS (=10) used by our extractor."""
import os, sys, glob, io, zipfile, random
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("D:/SLM-labs/SLM-Labs"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from slm_labs.config import cfg

def fps_of(df):
    t = df["t_ms"].values.astype(float)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return 1000.0 / np.median(d) if len(d) else np.nan

for sub in ["landmarks/user_sign", "landmarks/user_sentence"]:
    fs = sorted(glob.glob(str(cfg.DATA_DIR / sub / "*.csv")))
    random.seed(0); sample = random.sample(fs, min(60, len(fs)))
    r = [(fps_of(pd.read_csv(f)), len(pd.read_csv(f))) for f in sample]
    fps = np.array([x[0] for x in r]); T = np.array([x[1] for x in r])
    print(f"{sub:26s} n={len(fs):5d}  fps: med={np.nanmedian(fps):.2f} p05={np.nanpercentile(fps,5):.2f} p95={np.nanpercentile(fps,95):.2f}  "
          f"| frames med={np.median(T):.0f}  → duration med = {np.median(T)/np.nanmedian(fps):.2f} s")

zs = list(cfg.DATA_DIR.glob("landmarks/expert_primary_*.zip"))
if zs:
    z = zipfile.ZipFile(zs[0]); names = [n for n in z.namelist() if n.endswith(".csv")][:40]
    fps, T = [], []
    for n in names:
        df = pd.read_csv(io.BytesIO(z.read(n))); fps.append(fps_of(df)); T.append(len(df))
    print(f"{'expert_primary (zip)':26s} n={len(names):5d}  fps: med={np.nanmedian(fps):.2f}  frames med={np.median(T):.0f}  "
          f"→ duration med = {np.median(T)/np.nanmedian(fps):.2f} s")

print("\nour extractor TARGET_FPS =", cfg.TARGET_FPS, " → data_test clips: 103 frames = %.1f s, 67 frames = %.1f s" % (103/10, 67/10))
print("if dataset is ~27 fps, an isolated sign of 77 frames lasts %.2f s (plausible);" % (77/27),
      "at 10 fps it would be %.1f s (implausible)" % (77/10))
