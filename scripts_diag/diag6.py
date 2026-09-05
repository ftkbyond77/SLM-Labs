# -*- coding: utf-8 -*-
import os, sys, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("D:/SLM-labs/SLM-Labs"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from slm_labs.config import cfg
pd.set_option("display.width", 200)

sm = pd.read_csv(cfg.DATA_DIR / "metadata/user_sign_metadata.csv")
print("=== user_sign subset ===", sm["subset"].value_counts().to_dict())
print("   fps_extracted:", sm["fps_extracted"].value_counts().to_dict())
print("   source       :", sm["source"].value_counts().to_dict())
print("   frames med   :", sm["frames_extracted"].median(), " per-class clips: min/med/max =",
      sm.groupby("sign_id").size().min(), sm.groupby("sign_id").size().median(), sm.groupby("sign_id").size().max())
print("   subset x sign_id coverage: classes in train/val/test =",
      {s: sm[sm.subset==s]["sign_id"].nunique() for s in sm["subset"].unique()})
print("   video_id examples:", sm["video_id"].head(3).tolist(), "| recording_variation:", sm["recording_variation"].value_counts(dropna=False).to_dict())

stm = pd.read_csv(cfg.DATA_DIR / "metadata/sentence_metadata.csv")
print("\n=== sentence metadata ===", "n =", len(stm))
print("   fps:", stm["fps_extracted"].value_counts().to_dict(), "| recording_variation:", stm["recording_variation"].value_counts().to_dict())
print("   unique sentence_id:", stm["sentence_id"].nunique(), "| clips per sentence_id: ", stm.groupby("sentence_id").size().value_counts().to_dict())
if "subset" in stm.columns: print("   subset:", stm["subset"].value_counts().to_dict())

em = pd.read_csv(cfg.DATA_DIR / "metadata/expert_metadata.csv")
m = em[em["source_group"].astype(str).str.contains("primary", case=False)]
m0 = m[m["is_augmented"].astype(str).str.upper() != "TRUE"]
print("\n=== expert (source_group~primary, is_augmented False) ===", "n =", len(m0), "of", len(m))
print("   fps_extracted:", m0["fps_extracted"].value_counts().to_dict())
print("   frames med:", m0["frames_extracted"].median(), "→ duration med = %.2f s" % (m0["frames_extracted"].median()/m0["fps_extracted"].median()))
print("   source:", m0["source"].value_counts().to_dict(), "| signers:", m0["signer_name"].nunique())
print("   classes:", m0["sign_id"].nunique())
# actual csv timing of a few used expert clips
zips = [zipfile.ZipFile(p) for p in cfg.DATA_DIR.glob("landmarks/expert_primary_*.zip")]
index = {}
for z in zips:
    for n in z.namelist():
        if n.endswith(".csv"): index[Path(n).name] = (z, n)
cnt = 0
for _, r in m0.iterrows():
    key = Path(str(r["landmark_path"]).replace("\\","/")).name
    if key not in index: continue
    z, n = index[key]; df = pd.read_csv(io.BytesIO(z.read(n)))
    t = df["t_ms"].values.astype(float); d = np.diff(t); d = d[np.isfinite(d)&(d>0)]
    print(f"   used expert {key[:44]:46s} T={len(df):4d} dt_med={np.median(d):6.1f} → {1000/np.median(d):5.1f} fps  dur={t[-1]/1000:5.2f}s  sign={r['sign_id']}")
    cnt += 1
    if cnt >= 8: break
