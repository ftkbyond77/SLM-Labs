# -*- coding: utf-8 -*-
"""D6: raw landmark CSV comparison — dataset vs our MediaPipe extraction."""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("D:/SLM-labs/SLM-Labs"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from slm_labs.config import cfg
from slm_labs.vocab import LH_COLS, RH_COLS, POSE_COLS, FACE_COLS

ds_files = sorted(glob.glob(str(cfg.DATA_DIR / "landmarks/user_sign/*.csv")))[:40]
re_files = sorted(glob.glob(str(cfg.OUT_DIR / "landmarks_cache/*.csv")))
print("dataset csv files:", len(sorted(glob.glob(str(cfg.DATA_DIR / 'landmarks/user_sign/*.csv')))), "| sample cols:")
d0 = pd.read_csv(ds_files[0]); print("  ", list(d0.columns)[:8], "...", list(d0.columns)[-6:], "| n_cols =", len(d0.columns))
r0 = pd.read_csv(re_files[0]); print("  real cols n =", len(r0.columns))
print("  cols only in dataset:", [c for c in d0.columns if c not in r0.columns][:10])
print("  cols only in real   :", [c for c in r0.columns if c not in d0.columns][:10])

def summarize(files, tag):
    dfs = [pd.read_csv(f) for f in files]
    D = pd.concat(dfs, ignore_index=True)
    def rng(cols, lbl):
        v = D[[c for c in cols if c in D.columns]].values.astype(float)
        fin = np.isfinite(v)
        print(f"   {tag:8s} {lbl:10s} nan%={100*(1-fin.mean()):5.1f}  min={np.nanmin(v):8.3f} p05={np.nanpercentile(v,5):7.3f} "
              f"med={np.nanmedian(v):7.3f} p95={np.nanpercentile(v,95):7.3f} max={np.nanmax(v):8.3f}")
    for lbl, cols in [("lh_x", [c for c in LH_COLS if "_x" in c]), ("lh_y", [c for c in LH_COLS if "_y" in c]),
                      ("lh_z", [c for c in LH_COLS if "_z" in c]), ("rh_z", [c for c in RH_COLS if "_z" in c]),
                      ("pose_x", [c for c in POSE_COLS if c.endswith("_x")]), ("pose_y", [c for c in POSE_COLS if c.endswith("_y")]),
                      ("pose_z", [c for c in POSE_COLS if c.endswith("_z")]), ("face_x", [c for c in FACE_COLS if c.endswith("_x")])]:
        rng(cols, lbl)
    # frame rate
    for f, df in zip(files[:3], dfs[:3]):
        t = df["t_ms"].values
        print(f"   {tag:8s} {Path(f).name[:28]:30s} T={len(df)} dt_ms median={np.median(np.diff(t)) if len(t)>1 else 0:.0f}")
    return D

print("\n===== DATASET landmark CSVs =====")
Dd = summarize(ds_files, "dataset")
print("\n===== OUR EXTRACTION (data_test) =====")
Dr = summarize(re_files, "real")

print("\n===== hand-presence per clip =====")
for f in ds_files[:8]:
    d = pd.read_csv(f); print("   dataset %-40s lh=%.2f rh=%.2f" % (Path(f).name[:38], (~d["lh_x0"].isna()).mean(), (~d["rh_x0"].isna()).mean()))
for f in re_files:
    d = pd.read_csv(f); print("   real    %-40s lh=%.2f rh=%.2f" % (Path(f).stem[:38], (~d["lh_x0"].isna()).mean(), (~d["rh_x0"].isna()).mean()))
print("\n   dataset overall lh=%.3f rh=%.3f" % ((~Dd["lh_x0"].isna()).mean(), (~Dd["rh_x0"].isna()).mean()))
print("   real    overall lh=%.3f rh=%.3f" % ((~Dr["lh_x0"].isna()).mean(), (~Dr["rh_x0"].isna()).mean()))
print("\n   shoulder width  dataset=%.3f  real=%.3f" % (
    np.nanmean(np.linalg.norm(Dd[["l_shoulder_x","l_shoulder_y"]].values - Dd[["r_shoulder_x","r_shoulder_y"]].values, axis=1)),
    np.nanmean(np.linalg.norm(Dr[["l_shoulder_x","l_shoulder_y"]].values - Dr[["r_shoulder_x","r_shoulder_y"]].values, axis=1))))
# hand-to-shoulder distance in shoulder-width units (scale-invariant sanity)
def rel(D):
    c = (D[["l_shoulder_x","l_shoulder_y"]].values + D[["r_shoulder_x","r_shoulder_y"]].values)/2
    w = np.linalg.norm(D[["l_shoulder_x","l_shoulder_y"]].values - D[["r_shoulder_x","r_shoulder_y"]].values, axis=1)
    h = D[["rh_x0","rh_y0"]].values
    return np.nanmedian(np.linalg.norm(h-c, axis=1)/w)
print("   median |rh_wrist - shoulderCentre| / shoulderWidth : dataset=%.2f real=%.2f" % (rel(Dd), rel(Dr)))
