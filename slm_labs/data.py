"""Dataset: TSL-51 metadata → feature items → splits (ไม่รั่ว) → DataLoaders (isolated / sentence)

การแบ่ง split (v3) — แก้ leakage ของ v1/v2
  isolated  "official"     : train = expert(signer 02,03) + user[subset=calib] ; val/test = user[subset=test] แบ่งครึ่ง
            "cross_signer" : train = expert เท่านั้น       ; val/test = user ทั้งหมดแบ่งครึ่ง  (วัด generalisation ข้ามคน)
            "random"       : โปรโตคอลเดิมของ v1/v2 (สุ่ม 70/15/15 บน user + expert เข้า train ทั้งหมด)
  sentence  แบ่งตาม **pattern** (76 pattern, 3 คลิป/pattern) → pattern ใน test ไม่เคยอยู่ใน train
            (v1/v2 ใส่คลิปที่ 0/1/2 ของ pattern เดียวกันลง val/test/train → pattern ซ้ำกัน 100%)
"""
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
from .features import load_landmarks, make_features, temporal_resample, augment, crop, trim_to_active


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
    """user_sign (signer เดียว: primary_collection_01) — เก็บคอลัมน์ `subset` ของ dataset ไว้ใช้แบ่ง split"""
    items = []
    for _, r in tqdm(meta.sign.iterrows(), total=len(meta.sign), desc="user_sign csv", leave=False):
        p = meta.resolve(r["landmark_path"])
        if p is None or r[meta.sid_col] not in CLS2ID:
            continue
        items.append(dict(feat=make_features(load_landmarks(p)), label=CLS2ID[r[meta.sid_col]], src="user",
                          id=str(r["video_id"]), subset=str(r.get("subset", "")), signer="user_01"))
    return items


def read_expert_from_zip(meta: TSLMeta, max_per_class=None):
    """expert_primary_02/03 (signer อีก 2 คน) เฉพาะคลิป original — ใช้เป็น train pool หลัก"""
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
        out.append(dict(feat=make_features(load_landmarks(df)), label=CLS2ID[sid], src="expert",
                        id=str(r.get("video_id", key)), subset="expert", signer=str(r.get("source_group", "expert"))))
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


def trim_items(items, on: bool | None = None):
    """ตัด isolated clip ให้เหลือเฉพาะช่วงที่ทำท่าจริง (ดู features.active_span)

    ตอน inference โมเดลเห็นเฉพาะ segment ที่ตัดมาแล้ว — ถ้า train ด้วยคลิปเต็มที่มีหัว-ท้ายพักยาว
    Stage A จะเจอ distribution คนละแบบ (นี่คือหนึ่งในเหตุผลที่ v1/v2 ทำนาย segment ของคลิปจริงพลาด)"""
    if not (cfg.TRIM_ISOLATED if on is None else on):
        return items
    out = []
    for it in items:
        f = trim_to_active(it["feat"])
        out.append({**it, "feat": f if len(f["hand"]) >= 8 else it["feat"]})
    return out


# ---------------- splits ----------------

def _half_split(items, seed):
    from sklearn.model_selection import train_test_split
    y = [it["label"] for it in items]
    strat = y if min(Counter(y).values()) >= 2 else None
    a, b = train_test_split(range(len(items)), test_size=0.5, stratify=strat, random_state=seed)
    return [items[i] for i in a], [items[i] for i in b]


def make_splits(iso_user, iso_expert, sent_all, seed=cfg.SEED, mode: str | None = None, sent_by_pattern: bool | None = None):
    mode = mode or cfg.SPLIT_MODE
    sent_by_pattern = cfg.SENT_SPLIT_BY_PATTERN if sent_by_pattern is None else sent_by_pattern

    # ---- isolated ----
    if mode == "official":
        calib = [it for it in iso_user if it.get("subset") == "calib"]
        rest = [it for it in iso_user if it.get("subset") != "calib"]
        va, te = _half_split(rest or iso_user, seed)
        iso = dict(train=iso_expert + calib, val=va, test=te)
    elif mode == "cross_signer":
        va, te = _half_split(iso_user, seed)
        iso = dict(train=list(iso_expert), val=va, test=te)
    else:                                                    # "random" = โปรโตคอลเดิมของ v1/v2
        from sklearn.model_selection import train_test_split
        y_user = [it["label"] for it in iso_user]
        idx_tr, idx_tmp = train_test_split(range(len(iso_user)), test_size=0.30, stratify=y_user, random_state=seed)
        idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=[y_user[i] for i in idx_tmp], random_state=seed)
        iso = dict(train=[iso_user[i] for i in idx_tr] + iso_expert, val=[iso_user[i] for i in idx_va], test=[iso_user[i] for i in idx_te])

    # ---- sentence ----
    rng = random.Random(seed)
    by_pat = defaultdict(list)
    for it in sent_all:
        by_pat[it["pattern"]].append(it)
    if sent_by_pattern:
        pats = sorted(by_pat)
        rng.shuffle(pats)
        n = len(pats); n_te = max(1, round(0.15 * n)); n_va = max(1, round(0.15 * n))
        groups = dict(test=pats[:n_te], val=pats[n_te:n_te + n_va], train=pats[n_te + n_va:])
        sent = {k: [it for p in ps for it in by_pat[p]] for k, ps in groups.items()}
    else:                                                    # โปรโตคอลเดิม (pattern ปนกันทุก split)
        tr, va, te = [], [], []
        for k, (pat, its) in enumerate(sorted(by_pat.items())):
            rng.shuffle(its)
            if len(its) >= 3:
                tr += its[2:]; va.append(its[0]); te.append(its[1])
            elif len(its) == 2:
                tr.append(its[0]); (va if k % 2 else te).append(its[1])
            else:
                tr += its
        sent = dict(train=tr, val=va, test=te)
    return iso, sent


def split_report(iso, sent) -> pd.DataFrame:
    """ตรวจ leakage: id ซ้ำ + pattern ซ้ำ ระหว่าง split (ควรเป็น 0 ทุกคู่)"""
    rows = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        rows.append(dict(kind="isolated id", pair=f"{a}/{b}",
                         overlap=len({it["id"] for it in iso[a]} & {it["id"] for it in iso[b]})))
        rows.append(dict(kind="sentence id", pair=f"{a}/{b}",
                         overlap=len({it["id"] for it in sent[a]} & {it["id"] for it in sent[b]})))
        rows.append(dict(kind="sentence pattern", pair=f"{a}/{b}",
                         overlap=len({it["pattern"] for it in sent[a]} & {it["pattern"] for it in sent[b]})))
    return pd.DataFrame(rows)


def holdout_classes(iso, n=cfg.HOLDOUT_CLASSES, seed=cfg.SEED):
    """กัน n class ออกจาก train/val ทั้งหมด → คลิปของ class เหล่านั้นคือ "คำที่ไม่เคยเห็นจริง ๆ" สำหรับวัด open-set
    คืน (iso_ho, unknown_items, held_ids) ; test ยังเหลือเฉพาะ class ที่เห็นแล้ว"""
    from .vocab import CLS2ID, NULL_CLASS
    rng = random.Random(seed)
    pool = sorted({it["label"] for it in iso["train"]} - {CLS2ID[NULL_CLASS]})
    held = set(rng.sample(pool, min(n, len(pool))))
    keep = lambda its: [it for it in its if it["label"] not in held]
    drop = lambda its: [it for it in its if it["label"] in held]
    iso_ho = {k: keep(v) for k, v in iso.items()}
    unknown = drop(iso["val"]) + drop(iso["test"])
    return iso_ho, unknown, sorted(held)


def load_all(verbose=True, mode: str | None = None):
    """one-call: download → metadata → items → splits"""
    download_dataset()
    meta = TSLMeta()
    iso_user = build_isolated(meta)
    iso_expert = read_expert_from_zip(meta, cfg.EXPERT_MAX_PER_CLASS) if cfg.USE_EXPERT_PRIMARY else []
    sent_all = build_sentences(meta)
    iso, sent = make_splits(iso_user, iso_expert, sent_all, mode=mode)
    if verbose:
        print(f"isolated user={len(iso_user)} expert={len(iso_expert)} | sentences={len(sent_all)} "
              f"| split mode={mode or cfg.SPLIT_MODE} @ {cfg.TARGET_FPS:g} fps")
        print(f"isolated  train/val/test = {len(iso['train'])}/{len(iso['val'])}/{len(iso['test'])}")
        print(f"sentence  train/val/test = {len(sent['train'])}/{len(sent['val'])}/{len(sent['test'])}"
              f"  (patterns {len({i['pattern'] for i in sent['train']})}/{len({i['pattern'] for i in sent['val']})}/{len({i['pattern'] for i in sent['test']})})")
    return meta, iso, sent, (iso_user, iso_expert, sent_all)


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


# ---------------- synthetic continuous sentences (isolated clips → concatenation) ----------------

def make_synthetic_sentences(iso_items, n, real_sentences=None, k_range=(2, 5), seed=cfg.SEED, p_pattern=0.7, p_gap=0.5):
    """สร้างประโยคสังเคราะห์สำหรับ CTC: ต่อ isolated clips (จาก **split train เท่านั้น**) ตามลำดับ gloss
    (70% ใช้ pattern จาก sentence-train, 30% สุ่ม) คั่นด้วยท่าพัก (null_act) สั้น ๆ บางครั้ง"""
    from .vocab import NULL_CLASS
    rng = random.Random(seed); by_cls = defaultdict(list)
    for it in iso_items:
        by_cls[it["label"]].append(it)
    null_id = CLS2ID[NULL_CLASS]; null_clips = by_cls.get(null_id, [])
    classes = [c for c in by_cls if c != null_id]
    patterns = [[l - 1 for l in it["labels"]] for it in (real_sentences or [])]
    patterns = [p for p in patterns if all(c in by_cls for c in p)]
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
                nc = rng.choice(null_clips)["feat"]; L = rng.randint(3, 8); s0 = rng.randrange(max(1, len(nc["hand"]) - L))
                parts.append(crop(nc, s0, s0 + L))
        feat = {k: np.concatenate([p[k] for p in parts], 0) for k in ("hand", "body", "face")}
        out.append(dict(feat=feat, labels=[c + 1 for c in glosses], glosses=[SIGN_CLASSES[c] for c in glosses],
                        pattern=" ".join(CLASS2WORD[SIGN_CLASSES[c]] for c in glosses), id=f"syn_{i}", synthetic=True))
    return out
