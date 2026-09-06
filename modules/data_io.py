"""Dataset indexing, loading and leakage-safe splitting for TSL-ONE-S."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DATA_DIR, N_LANDMARKS, NOVEL_SEED, N_NOVEL_GLOSSES,
                     TEST_SIGNERS, VAL_SIGNERS)

# File name grammar:  <signer>_<category>_<gloss>.npy   e.g. 01_04_0133.npy
#   signer   29 unique ids  -> the identity we split on
#   category  8 unique ids  -> semantic group the gloss belongs to
#   gloss   184 unique ids  -> the class label


def build_index(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(data_dir).glob("*.npy")):
        signer, category, gloss = f.stem.split("_")
        rows.append({"path": str(f), "stem": f.stem, "signer": signer,
                     "category": category, "gloss": gloss})
    df = pd.DataFrame(rows)
    glosses = sorted(df["gloss"].unique())
    g2i = {g: i for i, g in enumerate(glosses)}
    df["label"] = df["gloss"].map(g2i)
    return df


def gloss_vocabulary(df: pd.DataFrame) -> list[str]:
    return sorted(df["gloss"].unique())


def load_clip(path: str) -> np.ndarray:
    """Return (T, 75, 2) float32. Exact zeros mean 'landmark not detected'."""
    a = np.load(path).astype(np.float32)
    return a.reshape(len(a), N_LANDMARKS, 2)


def load_all(df: pd.DataFrame, progress: bool = True) -> list[np.ndarray]:
    it = df["path"].tolist()
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="loading clips", unit="clip")
        except Exception:
            pass
    return [load_clip(p) for p in it]


# ------------------------------------------------------------ splitting ----
@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def signer_independent_split(df: pd.DataFrame,
                             test_signers=TEST_SIGNERS,
                             val_signers=VAL_SIGNERS) -> Split:
    """Partition by signer identity -> a test signer is never seen in training.

    This is the strictest realistic protocol for sign language: a random split
    would leak signer-specific motion style (and near-duplicate repetitions of
    the same gloss by the same person) straight into the test set.
    """
    test = df.index[df["signer"].isin(test_signers)].to_numpy()
    val = df.index[df["signer"].isin(val_signers)].to_numpy()
    train = df.index[~df["signer"].isin(set(test_signers) | set(val_signers))].to_numpy()
    return Split(train, val, test)


def novel_gloss_split(df: pd.DataFrame, n_novel: int = N_NOVEL_GLOSSES,
                      seed: int = NOVEL_SEED) -> tuple[list[str], list[str]]:
    """Pick glosses that the open-set probe model never trains on."""
    rng = np.random.default_rng(seed)
    glosses = gloss_vocabulary(df)
    novel = sorted(rng.choice(glosses, size=n_novel, replace=False).tolist())
    known = [g for g in glosses if g not in set(novel)]
    return known, novel


# -------------------------------------------------------- leakage audit ----
def _clip_fingerprint(path: str) -> str:
    a = np.load(path)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.round(a.astype(np.float64), 5)).tobytes())
    return h.hexdigest()


def audit_leakage(df: pd.DataFrame, split: Split, check_content: bool = True) -> dict:
    """Prove the partitions are disjoint on identity *and* on raw content."""
    rep = {}
    tr, va, te = (set(df.loc[split.train, "signer"]),
                  set(df.loc[split.val, "signer"]),
                  set(df.loc[split.test, "signer"]))
    rep["signer_overlap_train_test"] = sorted(tr & te)
    rep["signer_overlap_train_val"] = sorted(tr & va)
    rep["signer_overlap_val_test"] = sorted(va & te)

    idx_all = np.concatenate([split.train, split.val, split.test])
    rep["index_duplicates"] = int(len(idx_all) - len(set(idx_all.tolist())))
    rep["covers_dataset"] = bool(len(set(idx_all.tolist())) == len(df))

    s_tr, s_te = set(df.loc[split.train, "stem"]), set(df.loc[split.test, "stem"])
    rep["filename_overlap_train_test"] = sorted(s_tr & s_te)

    rep["glosses_train"] = int(df.loc[split.train, "gloss"].nunique())
    rep["glosses_val"] = int(df.loc[split.val, "gloss"].nunique())
    rep["glosses_test"] = int(df.loc[split.test, "gloss"].nunique())
    rep["test_glosses_unseen_in_train"] = sorted(
        set(df.loc[split.test, "gloss"]) - set(df.loc[split.train, "gloss"]))

    if check_content:
        fp_tr = {_clip_fingerprint(p) for p in df.loc[split.train, "path"]}
        dup = [s for s, p in zip(df.loc[split.test, "stem"], df.loc[split.test, "path"])
               if _clip_fingerprint(p) in fp_tr]
        rep["content_duplicate_test_clips"] = dup
    rep["clean"] = (not rep["signer_overlap_train_test"]
                    and not rep["signer_overlap_train_val"]
                    and not rep["signer_overlap_val_test"]
                    and not rep["filename_overlap_train_test"]
                    and rep["index_duplicates"] == 0
                    and rep["covers_dataset"]
                    and not rep.get("content_duplicate_test_clips", []))
    return rep
