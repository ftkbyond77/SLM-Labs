"""Dataset: TSL-51 metadata → feature items → splits → DataLoaders (isolated / sentence / SSL views)"""
from __future__ import annotations

import io
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .config import cfg
from .vocab import SIGN_CLASSES, CLS2ID, CLASS2WORD, parse_sentence_glosses
from .features import load_landmarks, make_features, temporal_resample, augment, random_view, crop


# ---------------- download / metadata ----------------

def download_dataset(force: bool = False) -> Path:
    from huggingface_hub import snapshot_download
    allow = ["README.md", "metadata/*", "landmarks/user_sign/*", "landmarks/user_sentence/*"]
    if cfg.USE_EXPERT_PRIMARY or cfg.USE_EXPERT_AUGMENTED:
        allow += ["landmarks/expert_primary_02.zip", "landmarks/expert_primary_03.zip"]
    if (cfg.DATA_DIR / "metadata" / "user_sign_metadata.csv").exists() and not force:
        return cfg.DATA_DIR
    local = snapshot_download(repo_id=cfg.HF_REPO, repo_type="dataset", local_dir=str(cfg.DATA_DIR), allow_patterns=allow)
    return Path(local)


class TSLMeta:
    """metadata ทั้ง 3 ไฟล์ + helper resolve path"""

    def __init__(self, data_dir: Path | None = None):
        self.DATA = Path(data_dir or cfg.DATA_DIR)
        self.sign = pd.read_csv(self.DATA / "metadata/user_sign_metadata.csv")
        self.sent = pd.read_csv(self.DATA / "metadata/sentence_metadata.csv")
        p = self.DATA / "metadata/expert_metadata.csv"
        self.expert = pd.read_csv(p) if p.exists() else None
        self.sid_col = "sign_id" if "sign_id" in self.sign.columns else self.sign.columns[1]
        self.lp_col = "landmark_path" if "landmark_path" in self.sent.columns else self.sent.columns[-2]
        self.vcol = "video_path" if "video_path" in self.sent.columns else None
        self.sent["glosses"] = self.sent[self.lp_col].map(parse_sentence_glosses)
        self.sent["pattern"] = self.sent["glosses"].map(lambda g: " ".join(CLASS2WORD.get(x, x) for x in g))

    def resolve(self, rel):
        rel = str(rel).replace("\\", "/")
        p = self.DATA / rel
        if p.exists():
            return p
        for sub in ["landmarks/user_sign", "landmarks/user_sentence"]:
            q = self.DATA / sub / Path(rel).name
            if q.exists():
                return q
        return None


# ---------------- item builders ----------------

def build_isolated(meta: TSLMeta):
    items = []
    for _, r in tqdm(meta.sign.iterrows(), total=len(meta.sign), desc="user_sign csv", leave=False):
        p = meta.resolve(r["landmark_path"])
        if p is None or r[meta.sid_col] not in CLS2ID:
            continue
        items.append(dict(feat=make_features(load_landmarks(p)), label=CLS2ID[r[meta.sid_col]], src="user", id=str(r["video_id"])))
    return items


def read_expert_from_zip(meta: TSLMeta, max_per_class=None):
    if meta.expert is None:
        return []
    m = meta.expert.copy()
    if "source_group" in m.columns:
        m = m[m["source_group"].astype(str).str.contains("primary", case=False)]
    if "is_augmented" in m.columns and not cfg.USE_EXPERT_AUGMENTED:
        m = m[m["is_augmented"].astype(str).str.upper() != "TRUE"]
    zips = [zipfile.ZipFile(p) for p in meta.DATA.glob("landmarks/expert_primary_*.zip")]
    index = {}
    for z in zips:
        for n in z.namelist():
            if n.endswith(".csv"):
                index[Path(n).name] = (z, n)
    out, per = [], Counter()
    for _, r in tqdm(m.iterrows(), total=len(m), desc="expert csv", leave=False):
        sid = r["sign_id"]
        if sid not in CLS2ID:
            continue
        if max_per_class and per[sid] >= max_per_class:
            continue
        key = Path(str(r["landmark_path"]).replace("\\", "/")).name
        if key not in index:
            continue
        z, n = index[key]
        df = pd.read_csv(io.BytesIO(z.read(n)))
        out.append(dict(feat=make_features(load_landmarks(df)), label=CLS2ID[sid], src="expert", id=str(r.get("video_id", key))))
        per[sid] += 1
    return out


def build_sentences(meta: TSLMeta):
    items = []
    for _, r in tqdm(meta.sent.iterrows(), total=len(meta.sent), desc="user_sentence csv", leave=False):
        p = meta.resolve(r[meta.lp_col])
        if p is None or any(g not in CLS2ID for g in r["glosses"]):
            continue
        items.append(dict(feat=make_features(load_landmarks(p)), labels=[CLS2ID[g] + 1 for g in r["glosses"]],
                          glosses=r["glosses"], pattern=r["pattern"], id=str(r["video_id"]), path=str(p)))
    return items


def make_splits(iso_user, iso_expert, sent_all, seed=cfg.SEED):
    from sklearn.model_selection import train_test_split
    y_user = [it["label"] for it in iso_user]
    idx_tr, idx_tmp = train_test_split(range(len(iso_user)), test_size=0.30, stratify=y_user, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=[y_user[i] for i in idx_tmp], random_state=seed)
    iso = dict(train=[iso_user[i] for i in idx_tr] + iso_expert, val=[iso_user[i] for i in idx_va], test=[iso_user[i] for i in idx_te])

    rng = random.Random(seed); by_pat = defaultdict(list)
    for it in sent_all:
        by_pat[it["pattern"]].append(it)
    tr, va, te = [], [], []
    for k, (pat, its) in enumerate(sorted(by_pat.items())):
        rng.shuffle(its)
        if len(its) >= 3:
            tr += its[2:]; va.append(its[0]); te.append(its[1])
        elif len(its) == 2:
            tr.append(its[0]); (va if k % 2 else te).append(its[1])
        else:
            tr += its
    return iso, dict(train=tr, val=va, test=te)


def load_all(verbose=True):
    """one-call: download → metadata → items → splits"""
    download_dataset()
    meta = TSLMeta()
    iso_user = build_isolated(meta)
    iso_expert = read_expert_from_zip(meta, cfg.EXPERT_MAX_PER_CLASS) if cfg.USE_EXPERT_PRIMARY else []
    sent_all = build_sentences(meta)
    iso, sent = make_splits(iso_user, iso_expert, sent_all)
    if verbose:
        print(f"isolated user={len(iso_user)} expert={len(iso_expert)} | sentences={len(sent_all)}")
        print(f"isolated  train/val/test = {len(iso['train'])}/{len(iso['val'])}/{len(iso['test'])}")
        print(f"sentence  train/val/test = {len(sent['train'])}/{len(sent['val'])}/{len(sent['test'])}")
    return meta, iso, sent


# ---------------- torch datasets ----------------

class SignDataset(Dataset):
    def __init__(self, items, max_frames, train=False, seq=False):
        self.items, self.max_frames, self.train, self.seq = items, max_frames, train, seq

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]; f = it["feat"]
        if self.train:
            f = augment(f)
        if len(f["hand"]) > self.max_frames:
            f = temporal_resample(f, self.max_frames)
        tgt = it["labels"] if self.seq else it["label"]
        return f, tgt


def pad_feats(feats):
    lens = torch.tensor([len(f["hand"]) for f in feats]); Tm = int(lens.max())

    def pad(k):
        D = feats[0][k].shape[1]; out = torch.zeros(len(feats), Tm, D)
        for i, f in enumerate(feats):
            out[i, :len(f[k])] = torch.from_numpy(np.ascontiguousarray(f[k]))
        return out

    mask = torch.arange(Tm)[None, :] >= lens[:, None]
    return dict(hand=pad("hand"), body=pad("body"), face=pad("face"), mask=mask, lens=lens)


def collate(batch):
    feats, tgts = zip(*batch)
    x = pad_feats(feats)
    if isinstance(tgts[0], list):
        x["target_lens"] = torch.tensor([len(t) for t in tgts]); x["targets"] = torch.tensor([t for s in tgts for t in s])
    else:
        x["targets"] = torch.tensor(tgts)
    return x


def make_loader(items, max_frames, train, seq, bs=None):
    return DataLoader(SignDataset(items, max_frames, train, seq), batch_size=bs or cfg.BATCH_SIZE,
                      shuffle=train, collate_fn=collate, num_workers=0, drop_last=False)


class SSLViewDataset(Dataset):
    """SignDINO-style multi-crop: 2 global views + N local views ต่อคลิป (ไม่ใช้ label)"""

    def __init__(self, items, n_local=cfg.SSL_N_LOCAL, max_frames=cfg.SSL_MAX_FRAMES):
        self.items, self.n_local, self.max_frames = items, n_local, max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f = self.items[i]["feat"]
        if len(f["hand"]) > self.max_frames * 2:
            f = temporal_resample(f, self.max_frames * 2)
        g = [random_view(f, cfg.SSL_GLOBAL_CROP, 0.6) for _ in range(2)]
        l = [random_view(f, cfg.SSL_LOCAL_CROP, 1.0) for _ in range(self.n_local)]
        return [temporal_resample(v, self.max_frames) if len(v["hand"]) > self.max_frames else v for v in g + l]


def collate_views(batch):
    n_views = len(batch[0])
    return [pad_feats([b[v] for b in batch]) for v in range(n_views)]


def make_ssl_loader(items, bs=None):
    return DataLoader(SSLViewDataset(items), batch_size=bs or cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_views,
                      num_workers=0, drop_last=True)


# ---------------- synthetic continuous sentences (isolated clips → concatenation) ----------------

def make_synthetic_sentences(iso_items, n, real_sentences=None, k_range=(2, 5), seed=cfg.SEED, p_pattern=0.7, p_gap=0.5):
    """สร้างประโยคสังเคราะห์สำหรับ CTC: ต่อ isolated clips ตามลำดับ gloss (70% ใช้ pattern จริงจาก sentence-train, 30% สุ่ม)
    คั่นด้วยท่าพัก (null_act) สั้น ๆ บางครั้ง → ให้ CTC เห็น alignment หลากหลาย (แก้ปัญหา blank-collapse เมื่อประโยคจริงมีแค่ ~130 คลิป)"""
    from .vocab import NULL_CLASS, SIGN_CLASSES
    rng = random.Random(seed); by_cls = defaultdict(list)
    for it in iso_items:
        by_cls[it["label"]].append(it)
    null_id = CLS2ID[NULL_CLASS]; null_clips = by_cls.get(null_id, [])
    classes = [c for c in by_cls if c != null_id]
    patterns = [[l - 1 for l in it["labels"]] for it in (real_sentences or [])]
    out = []
    for i in range(n):
        if patterns and rng.random() < p_pattern:
            glosses = list(rng.choice(patterns))
        else:
            glosses = [rng.choice(classes) for _ in range(rng.randint(*k_range))]
        parts = []
        for c in glosses:
            parts.append(rng.choice(by_cls[c])["feat"])
            if null_clips and rng.random() < p_gap:
                nc = rng.choice(null_clips)["feat"]; L = rng.randint(3, 10); s0 = rng.randrange(max(1, len(nc["hand"]) - L))
                parts.append(crop(nc, s0, s0 + L))
        feat = {k: np.concatenate([p[k] for p in parts], 0) for k in ("hand", "body", "face")}
        out.append(dict(feat=feat, labels=[c + 1 for c in glosses], glosses=[SIGN_CLASSES[c] for c in glosses],
                        pattern=" ".join(CLASS2WORD[SIGN_CLASSES[c]] for c in glosses), id=f"syn_{i}", synthetic=True))
    return out
