# -*- coding: utf-8 -*-
"""Build the feature-item cache at cfg.TARGET_FPS (run once; ~few minutes)."""
import os, sys, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from slm_labs.config import cfg
from slm_labs.data import TSLMeta, build_isolated, read_expert_from_zip, build_sentences, download_dataset

out = cfg.OUT_DIR / f"items_raw_{cfg.TARGET_FPS:g}fps.pkl"
if out.exists():
    print("exists:", out); sys.exit(0)
t0 = time.time(); download_dataset(); meta = TSLMeta()
iso_user = build_isolated(meta); print("user_sign items:", len(iso_user), f"{time.time()-t0:.0f}s")
iso_expert = read_expert_from_zip(meta, cfg.EXPERT_MAX_PER_CLASS); print("expert items:", len(iso_expert), f"{time.time()-t0:.0f}s")
sent_all = build_sentences(meta); print("sentence items:", len(sent_all), f"{time.time()-t0:.0f}s")
pickle.dump((iso_user, iso_expert, sent_all), open(out, "wb"))
import numpy as np
print("frames @ %g fps  iso_user med=%.0f  expert med=%.0f  sentence med=%.0f" % (
    cfg.TARGET_FPS, np.median([len(i["feat"]["hand"]) for i in iso_user]),
    np.median([len(i["feat"]["hand"]) for i in iso_expert]), np.median([len(i["feat"]["hand"]) for i in sent_all])))
print("saved", out, f"({out.stat().st_size/1e6:.0f} MB)  total {time.time()-t0:.0f}s")
