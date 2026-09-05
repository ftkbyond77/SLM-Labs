# -*- coding: utf-8 -*-
"""Generate SLM_Labs_main.ipynb (v3 lab notebook) from this file's cell list."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent

MD = "md"
CODE = "code"

CELLS = [
(MD, r"""# SLM Labs v3 — Thai Sign Language → open-vocabulary extraction → LLM → Thai → เสียงพูด

**env: `conda activate hugging` (torch GPU)** · package [`slm_labs/`](slm_labs) · driver [`run_v3.py`](run_v3.py) · บันทึกผล [`prompt/result/fixed_v2.md`](prompt/result/fixed_v2.md)

```
.mp4 (fps อะไรก็ได้ / ถ่ายที่ไหนก็ได้)
   │  extractor.py  MediaPipe Holistic → landmark  ─┐
   │                                                ├─► **resample ที่ cfg.TARGET_FPS เดียวกันทั้งคู่**
   │  dataset TSL-51 landmark CSV (~24-30 fps)     ─┘   (v1/v2 พังตรงนี้: dataset 25 fps แต่ inference 10 fps)
   ▼
 features (Hand 254 | Body 51 | Face 39, normalise ด้วยความกว้างไหล่)
   │
   ├─► hand-activity segmentation  ──┐
   │                                  ├─► segment (start,end)  ── "มีกี่คำ" โดยไม่ใช้ vocab
   ├─► CTC frame-posterior (ตัวเสนอขอบเขต, ไม่ใช่ตัว decode) ──┘
   ▼
 SignEncoder (MLP×3 → fusion → Transformer)
   ├─ cls head  (CE)                  ── Stage A
   ├─ emb head  (128-d, L2) + ArcFace ── Stage A  ← **ใหม่: emb head ถูกฝึกจริง**
   └─ ctc head  (CTC)                 ── Stage B (encoder ถูก freeze → Stage A ไม่พัง)
   ▼
 ต่อ segment: cls + cosine กับ class prototype → known / '_' (unknown) / learned (จาก memory) / null
   ▼
 memory (Qdrant local) เก็บ embedding + metadata ของ '_' → annotate ทีหลัง → รู้จักโดยไม่ retrain
   ▼
 LLM → ประโยคไทย (คง ___ ไว้)  →  TTS (MMS-TTS-tha) → .wav
```

| สิ่งที่เปลี่ยนจาก v2 | เหตุผล (มีหลักฐานใน §1-§2) |
|---|---|
| ตัด SSL (SignDINO) ออก | Stage A v2 ได้ 0.988 vs v1 0.976 — ต่างกัน 1 คลิป ไม่คุ้มความซับซ้อน |
| บังคับ fps เดียวกัน dataset ↔ วิดีโอใหม่ | สาเหตุหลักที่ v1/v2 อ่านคลิปจริงไม่ออก |
| ฝึก emb head ด้วย ArcFace | ของเดิม emb head ไม่เคยได้ gradient → open-set ทำงานบน random projection |
| Stage B freeze encoder | ของเดิม Stage B ทำ isolated acc ของ v2 ตกจาก 0.988 → 0.337 |
| split ไม่รั่ว (pattern แยกกัน) | ของเดิม pattern ของ val/test อยู่ใน train ครบ 100% |
"""),

(MD, "## 0. Setup"),
(CODE, r"""import os, sys, json, time, pickle, warnings
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "slm_labs").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display, Audio, Video

from slm_labs.config import cfg, DEVICE, seed_all
from slm_labs.viz import setup_thai_font, plot_timeline, render_overlay_video
FONT = setup_thai_font()
RETRAIN = os.environ.get("SLM_RETRAIN", "0") == "1"     # 0 = โหลด checkpoint จาก run_v3.py
print(f"device={DEVICE} | torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"font={FONT[0]} | TARGET_FPS={cfg.TARGET_FPS:g} | split={cfg.SPLIT_MODE} | RETRAIN={RETRAIN}")
pd.Series({k: v for k, v in cfg.to_dict().items() if "KEY" not in k}).to_frame("value").T"""),

(MD, r"""## 1. ข้อมูล + ตรวจ leakage

TSL-51 มี 3 แหล่ง: `user_sign` (547 คลิป, signer 1 คน, dataset ระบุ subset = `calib`/`test`), `expert_primary_02/03`
(1033 คลิป original, signer อีก 2 คน) และ `user_sentence` (252 คลิป, **76 pattern × 3 การถ่าย**)

- **isolated** : train = expert + user[calib] · val/test = user[test] แบ่งครึ่ง
- **sentence** : แบ่งตาม pattern — pattern ที่อยู่ใน test จะไม่มีใน train เลย

> v1/v2 เอาคลิปที่ 0/1/2 ของ pattern เดียวกันไปใส่ val/test/train → **pattern ซ้ำกัน 100%** ตัวเลข WER เดิมจึงมองโลกในแง่ดีเกินจริง"""),
(CODE, r"""from slm_labs.data import make_splits, split_report, make_loader, make_synthetic_sentences, trim_items, holdout_classes
from slm_labs.vocab import SIGN_CLASSES, CLASS2WORD, N_CLASSES, KNOWN_WORDS, UNK, NULL_CLASS

iso_user, iso_expert, sent_all = pickle.load(open(cfg.OUT_DIR / f"items_raw_{cfg.TARGET_FPS:g}fps.pkl", "rb"))
iso_user, iso_expert = trim_items(iso_user), trim_items(iso_expert)
iso, sent = make_splits(iso_user, iso_expert, sent_all)
print(f"isolated {len(iso['train'])}/{len(iso['val'])}/{len(iso['test'])} | sentence {len(sent['train'])}/{len(sent['val'])}/{len(sent['test'])}")
display(split_report(iso, sent))          # ทุกค่าต้องเป็น 0

fig, ax = plt.subplots(1, 3, figsize=(18, 3.4))
pd.Series([CLASS2WORD[SIGN_CLASSES[it["label"]]] for it in iso["train"]]).value_counts().plot.bar(ax=ax[0], color="#2a9d8f")
ax[0].set_title("isolated train: คลิปต่อคำ"); ax[0].tick_params(labelsize=8)
ax[1].hist([len(it["feat"]["hand"]) for it in iso["train"]], 30, alpha=.6, label="isolated")
ax[1].hist([len(it["feat"]["hand"]) for it in sent["train"]], 30, alpha=.6, label="sentence")
ax[1].legend(); ax[1].set_title(f"เฟรมต่อคลิป @ {cfg.TARGET_FPS:g} fps")
pd.Series([it["pattern"] for it in sent["train"]]).value_counts().head(10).plot.barh(ax=ax[2], color="#e76f51")
ax[2].invert_yaxis(); ax[2].set_title("sentence patterns (train)")
plt.tight_layout(); plt.show()"""),

(MD, r"""## 2. บั๊กที่ทำให้ v1/v2 อ่านวิดีโอจริงไม่ออก — frame rate

`user_sign_metadata.fps_extracted = 30` (t_ms dt ≈ 42 ms → ~24 fps จริง) แต่ `cfg.TARGET_FPS` เดิม = 10
→ `HolisticExtractor` ดึงวิดีโอใหม่ที่ 10 fps ท่าเดียวกันจึงสั้นกว่าที่ Stage A เคยเห็น 2.5 เท่า

เซลล์นี้เทียบ "กี่วินาทีต่อหนึ่งท่า" ของ dataset กับวิดีโอทดสอบ ทั้งก่อนและหลังแก้"""),
(CODE, r"""import pandas as pd
raw = pd.read_csv(cfg.DATA_DIR / "metadata/user_sign_metadata.csv")
rows = [dict(source="dataset user_sign", fps_meta=raw["fps_extracted"].median(), frames_meta=raw["frames_extracted"].median(),
             sec=raw["frames_extracted"].median() / raw["fps_extracted"].median()),
        dict(source=f"หลัง resample @ {cfg.TARGET_FPS:g} fps", fps_meta=cfg.TARGET_FPS,
             frames_meta=np.median([len(i["feat"]["hand"]) for i in iso_user]),
             sec=np.median([len(i["feat"]["hand"]) for i in iso_user]) / cfg.TARGET_FPS)]
for p in sorted((cfg.OUT_DIR / "landmarks_cache").glob(f"*@{cfg.TARGET_FPS:g}fps.csv")):
    d = pd.read_csv(p); rows.append(dict(source="วิดีโอจริง " + p.stem.split("@")[0][:16], fps_meta=cfg.TARGET_FPS,
                                         frames_meta=len(d), sec=len(d) / cfg.TARGET_FPS))
display(pd.DataFrame(rows).round(2))
print("→ dataset กับวิดีโอใหม่ต้องอยู่บนแกนเวลาเดียวกัน ไม่งั้น Transformer เห็นคนละความเร็วของท่าเดียวกัน")"""),

(MD, r"""## 3. Stage A — SignEncoder + ArcFace

```
Hand ─► MLP ─┐                                                ┌─► masked-mean ─► cls head  ─ CE
Body ─► MLP ─┼─► fusion ─► +pos ─► Transformer (pre-LN, 4L) ─┼─► masked-mean ─► emb head ─ ArcFace(s=24, m=0.25)
Face ─► MLP ─┘                                                └─► per-frame(÷4) ─► ctc head (Stage B)
```
ArcFace คือส่วนที่ทำให้ `cos(emb, prototype)` มีความหมาย — ของเดิม emb head ไม่มี loss ใดแตะเลย
(ตรวจได้: `max|Δweight|` ระหว่าง checkpoint ที่ train แล้วกับ random init = 0 พอดี)"""),
(CODE, r"""from slm_labs.model import build_model
from slm_labs.train import train_stage_a, train_stage_b, load_checkpoint, save_checkpoint
from slm_labs.metrics import eval_isolated, eval_sequence, eval_isolated_shifted, openset_auroc, tune_blank_penalty

dl_iso_tr = make_loader(iso["train"], cfg.MAX_FRAMES_ISO, True,  False)
dl_iso_va = make_loader(iso["val"],   cfg.MAX_FRAMES_ISO, False, False)
dl_iso_te = make_loader(iso["test"],  cfg.MAX_FRAMES_ISO, False, False)

seed_all(cfg.SEED); model = build_model(N_CLASSES, DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
if RETRAIN:
    histA = train_stage_a(model, dl_iso_tr, dl_iso_va, cfg.EPOCHS_A, tag="v3_stageA")
else:
    load_checkpoint(model, cfg.OUT_DIR / "v3_stageA_best.pt"); histA = pd.read_csv(cfg.OUT_DIR / "hist_v3_stageA.csv")
resA = eval_isolated(model, dl_iso_te)
print(f"isolated TEST  acc={resA['acc']:.4f}  macro-F1={resA['macro_f1']:.4f}")
histA.plot(x="ep", y=["val_acc", "val_f1"], figsize=(6, 2.6), title="Stage A validation"); plt.show()"""),

(MD, r"""## 4. Embedding / prototypes / open-set

- prototype ต่อ class = mean ของ emb บน train · `τ_sim` = quantile ที่ 10 ของ cosine(class ถูก) บน val
- วัด open-set 2 แบบ
  1. **pseudo-unknown** — คลิปเดิมเล่นย้อนกลับ (โปรโตคอลเดิมของ v1/v2)
  2. **held-out classes** — กัน 8 คำออกจาก train ทั้งหมด แล้วถามว่าโมเดลรู้ตัวไหมว่า "ไม่รู้จักคำนี้" ← อันนี้คือของจริง"""),
(CODE, r"""from slm_labs.openset import (Prototypes, analyze_clip, eval_sequence_segments, eval_segment_count,
                              known_unknown_scores, tune_segmentation, tune_openset_thresholds, SEG_PARAMS)
from slm_labs.features import time_reverse

protos = Prototypes.load(cfg.OUT_DIR / "prototypes.npz") if not RETRAIN else Prototypes.build(model, iso["train"], iso["val"])
print("tau_sim=%.3f" % protos.sim_thr, {k: round(v, 3) for k, v in protos.stats.items()})

summary = json.loads((cfg.OUT_DIR / "summary_v3.json").read_text(encoding="utf-8"))
display(pd.DataFrame(summary["openset"]))
print("held-out:", summary["holdout_classes"])

ks, us = known_unknown_scores(model, protos, iso["test"], [dict(feat=time_reverse(it["feat"])) for it in iso["test"]])
plt.figure(figsize=(6, 2.8)); plt.hist(ks, 25, alpha=.6, label="known (isolated test)")
plt.hist(us, 25, alpha=.6, label="pseudo-unknown"); plt.axvline(protos.sim_thr, color="k", ls="--", label="tau_sim")
plt.legend(); plt.title("cosine กับ prototype ที่ใกล้ที่สุด"); plt.show()"""),
(CODE, r"""# embedding space (PCA) — 12 คำที่มีคลิปมากที่สุดใน test
from sklearn.decomposition import PCA
E, _ = Prototypes.embed_items(model, iso["test"]); y = np.array([it["label"] for it in iso["test"]])
top = pd.Series(y).value_counts().head(12).index.tolist()
mask = np.isin(y, top); P2 = PCA(2).fit_transform(E)
plt.figure(figsize=(6.5, 5))
for c in top:
    sel = (y == c)
    plt.scatter(P2[sel, 0], P2[sel, 1], s=18, label=CLASS2WORD[SIGN_CLASSES[c]])
plt.legend(fontsize=9, ncol=2); plt.title("emb head (ArcFace) — PCA ของ isolated test"); plt.show()"""),

(MD, r"""## 5. Segmentation — "มีกี่คำ" โดยไม่ใช้ vocab

สองสัญญาณรวมกัน
1. **hand-activity** : ความเร็วมือ + ข้อมือยกเหนือระดับพัก → ช่วงที่กำลังทำท่า (grid-search บน sentence-train)
2. **CTC frame-posterior** : ใช้เป็น *ตัวเสนอขอบเขต* เท่านั้น — diagnostic พบว่า CTC เรียง "ตำแหน่ง" ของคำถูก
   แม้ P(blank)≈0.97 จะทำให้ greedy decode ออกมาว่าง คือ alignment ดีแต่ calibration พัง จึงเอาเฉพาะส่วนที่มันเก่งมาใช้
   → ช่วยตัด segment ที่ยาวเกินในคลิปจริง ซึ่งเซ็นเร็วกว่า dataset ~2 เท่า"""),
(CODE, r"""tune = tune_segmentation(sent["train"], set_global=True)
display(tune.head(5)); print("SEG_PARAMS =", SEG_PARAMS)
display(pd.DataFrame(summary["segcount"]))

it = sent["test"][0]
a = analyze_clip(model, it["feat"], protos)
plot_timeline(a, f"dataset sentence — ref: {it['pattern']}"); plt.show()"""),

(MD, r"""## 6. Stage B — สองทางเลือก (ทั้งคู่ไม่แตะ encoder)

| | ฝึกอะไร | ใช้ทำอะไร |
|---|---|---|
| **B-ctc** | `ctc_head` 13.6k พารามิเตอร์ บน encoder ที่ freeze | ถอดลำดับ gloss ทั้งประโยค + เป็นตัวเสนอขอบเขตให้ B-seg |
| **B-seg** | ไม่ต้องฝึกเพิ่ม | segment → classify + open-set → ได้ `_` สำหรับคำนอก vocab (ทางหลักของ pipeline) |

เพราะ encoder ถูก freeze ความแม่นของ Stage A จึง **เท่าเดิมเป๊ะ** หลัง Stage B (v2 เคยตกจาก 0.988 → 0.337)"""),
(CODE, r"""dl_seq_va = make_loader(sent["val"],  cfg.MAX_FRAMES_SEQ, False, True, bs=16)
dl_seq_te = make_loader(sent["test"], cfg.MAX_FRAMES_SEQ, False, True, bs=16)
if RETRAIN:
    syn = make_synthetic_sentences(iso["train"], cfg.SYN_SENTENCES, sent["train"])
    dl_seq_tr = make_loader(sent["train"] + syn, cfg.MAX_FRAMES_SEQ, True, True, bs=16)
    train_stage_b(model, dl_seq_tr, dl_seq_va, cfg.EPOCHS_B, tag="v3_stageB")
    display(tune_blank_penalty(model, dl_seq_va).head())
else:
    load_checkpoint(model, cfg.OUT_DIR / "sign_encoder_v3.pt")
    cfg.BLANK_PENALTY = summary["stage_b"]["ctc"]["blank_penalty"]
    cfg.SEG_CLS_CONF  = summary["stage_b"]["seg"]["seg_cls_conf"]; protos.sim_thr = summary["stage_b"]["seg"]["tau"]

print("BLANK_PENALTY =", cfg.BLANK_PENALTY, "| SEG_CLS_CONF =", cfg.SEG_CLS_CONF, "| tau_sim =", round(protos.sim_thr, 3))
b = summary["stage_b"]
display(pd.DataFrame([
    dict(**{"Stage B": "B-ctc (blank penalty = 0)"}, WER=b["ctc"]["wer_bp0"], CER=None, sent_acc=None),
    dict(**{"Stage B": f"B-ctc (blank penalty = {b['ctc']['blank_penalty']:g})"}, WER=b["ctc"]["wer"], CER=b["ctc"]["cer"], sent_acc=b["ctc"]["sent_acc"]),
    dict(**{"Stage B": "B-seg (motion segments เท่านั้น)"}, WER=b["seg"]["wer_no_ctc_cuts"], CER=None, sent_acc=None),
    dict(**{"Stage B": "B-seg (+ CTC boundary cuts)"}, WER=b["seg"]["wer"], CER=b["seg"]["cer"], sent_acc=b["seg"]["sent_acc"]),
]).round(4))
print(f"isolated TEST ก่อน Stage B = {b['iso_acc_before']:.4f} / หลัง = {b['iso_acc_after']:.4f}  → encoder ถูก freeze จริง")"""),
(CODE, r"""r = eval_sequence_segments(model, protos, sent["test"], keep_unknown=True)
display(pd.DataFrame({"ref (gloss จริง)": [" ".join(x) for x in r["refs"][:8]],
                      "hyp (B-seg, คง '_')": [" ".join(x) for x in r["hyps"][:8]]}))"""),

(MD, r"""## 7. วิดีโอจริง `data_test/*.mp4` → คำ → memory → LLM → **เสียง**

- ชื่อไฟล์คือประโยคที่ผู้ใช้ annotate ไว้ ใช้เป็น metadata เท่านั้น (ไม่ใช่ ground truth ของ segment)
- `EXPECTED` = gloss ที่ควรได้ (`_` = คำนอก vocab 51 คำ) ใช้คิด known-recall"""),
(CODE, r"""from slm_labs.extractor import HolisticExtractor
from slm_labs.memory import SignMemory
from slm_labs.pipeline import SLMPipeline
from slm_labs.llm import set_face_baseline
from slm_labs.tts import tts, speakable

set_face_baseline(iso["train"], cfg.OUT_DIR / "face_baseline.npz")
EXPECTED = {"ไปทานข้าวด้วยกันมั้ย": ["ไป", "_", "ข้าว", "ด้วยกัน", "_"],
            "ฉันปลอบเพื่อนร้องไห้": ["ฉัน", "_", "_", "_"]}
memory = SignMemory(root=cfg.MEMORY_DIR / "v3")
print("memory backend:", memory.backend, "| segments:", memory.count())
pipe = SLMPipeline(model=model, protos=protos, memory=memory, extractor=HolisticExtractor())

results = {}
for v in sorted(cfg.DATA_TEST_DIR.glob("*.mp4")):
    display(Video(str(v), width=300))
    results[v.stem] = pipe.run(v, expected_words=EXPECTED.get(v.stem), speak=True)
    plot_timeline(results[v.stem]["_analysis"], f"{v.stem}  →  {' '.join(results[v.stem]['words'])}"); plt.show()
    if results[v.stem]["wav"]:
        print("TTS:", speakable(results[v.stem]["thai"])); display(Audio(results[v.stem]["wav"]))"""),
(CODE, r"""vdf = pd.DataFrame([dict(clip=k, words=" ".join(r["words"]), expected=" ".join(EXPECTED.get(k, [])),
                         known=r["n_known"], unknown=r["n_unknown"], known_recall=r.get("known_recall"),
                         thai=r["thai"], llm=r["llm"], stored=r["memory_stored"]) for k, r in results.items()])
display(vdf); vdf.to_csv(cfg.OUT_DIR / "data_test_results_v3.csv", index=False, encoding="utf-8-sig")
display(memory.table())"""),

(MD, r"""## 8. เรียนคำใหม่โดยไม่ retrain (annotate → learned prototype)

ดู `memory.table()` / timeline / overlay → บอกว่า slot ไหนคือคำอะไร → `memory.annotate()` →
segment นั้นกลายเป็น learned prototype → รอบหน้า `analyze_clip` คืนคำนั้นทันที (status `learned`)"""),
(CODE, r"""import shutil
demo = cfg.MEMORY_DIR / "_demo_v3"; shutil.rmtree(demo, ignore_errors=True)
mem_demo = SignMemory(root=demo)
pipe_demo = SLMPipeline(model=model, protos=protos, memory=mem_demo, extractor=pipe.extractor)
clip = "ไปทานข้าวด้วยกันมั้ย"
r0 = pipe_demo.run(cfg.DATA_TEST_DIR / f"{clip}.mp4", translate=False, speak=False, show=False)
slots0 = [s for s in r0["slots"] if s["status"] != "null"]
print("ก่อน annotate:", [f"{s['idx']}:{s['word']}({s['status']})" for s in slots0])

unk_tokens = [t for t in r0["filename_annotation"]["tokens"] if t not in KNOWN_WORDS] or ["?"]
labels, j = {}, 0
for s in slots0:
    if s["status"] in ("known", "learned"):
        labels[s["idx"]] = s["word"]
    else:
        labels[s["idx"]] = unk_tokens[min(j, len(unk_tokens) - 1)]; j += 1
print("annotate:", labels, "→", mem_demo.annotate(clip, labels, note="demo: map จากชื่อไฟล์"), "segments")
print("learned prototypes:", {k: v["n"] for k, v in mem_demo.learned_prototypes().items()})

r1 = pipe_demo.run(cfg.DATA_TEST_DIR / f"{clip}.mp4", translate=True, speak=True, show=False, save_memory=False)
print("หลัง annotate :", r1["words"], "→ THAI:", r1["thai"], f"({r1['llm']})")
print("   status:", [(s["idx"], s["word"], s["status"], round(s["sim"], 3)) for s in r1["slots"] if s["status"] != "null"])
if r1["wav"]: display(Audio(r1["wav"]))"""),

(MD, "## 9. Overlay video (ตรวจว่า segment ตรงกับท่าจริงไหม)"),
(CODE, r"""k0 = sorted(results)[0]
ov = render_overlay_video(cfg.DATA_TEST_DIR / f"{k0}.mp4", results[k0]["_analysis"],
                          cfg.OUT_DIR / "inference" / f"{k0}_overlay_v3.mp4")
print("overlay →", ov); display(Video(str(ov), width=320))"""),

(MD, r"""## 10. สรุป — v1 / v2 (เดิม) เทียบ v3

> **อ่านตัวเลขให้ถูก**: คอลัมน์ v1/v2 มาจาก split เดิมที่ pattern ของประโยครั่ว 100% และ isolated แบ่งแบบสุ่ม
> (signer เดียวกันอยู่ทั้ง train และ test) ส่วน v3 ใช้ split ที่ไม่รั่ว ตัวเลข isolated ที่ต่ำลงคือ *ตัวเลขที่จริง*
> ไม่ใช่ regression — เซลล์สุดท้ายรัน v3 บน split เดิมเพื่อเทียบแบบ apples-to-apples ด้วย"""),
(CODE, r"""S = summary
rows = [
 ("Sign recognition (isolated test)", "accuracy",           0.9759, 0.3373, S["stage_a"]["acc"]),
 ("Sign recognition (isolated test)", "macro-F1",           0.9806, 0.2518, S["stage_a"]["macro_f1"]),
 ("Sign recognition หลัง Stage B",    "accuracy",           0.9759, 0.3373, S["stage_b"]["iso_acc_after"]),
 ("Sentence (WER)",                   "Stage B ดีที่สุด",   1.0,    1.0,    min(S["stage_b"]["seg"]["wer"], S["stage_b"]["ctc"]["wer"])),
 ("Sentence (CER)",                   "Stage B ดีที่สุด",   1.0,    1.0,    min(S["stage_b"]["seg"]["cer"], S["stage_b"]["ctc"]["cer"])),
 ("Open-set (pseudo-unknown)",        "AUROC",              0.5622, 0.5007, [r for r in S["openset"] if r["protocol"].startswith("pseudo") and r["model"].startswith("v3")][0]["auroc"]),
 ("Open-set (held-out 8 คำ)",         "AUROC",              None,   None,   [r for r in S["openset"] if r["protocol"].startswith("held") and r["model"].startswith("v3")][0]["auroc"]),
 ("Segment count (sentence test)",    "MAE",                2.581,  2.419,  S["segcount"][-1]["mae"]),
]
summary_df = pd.DataFrame(rows, columns=["stage", "metric", "v1 (leaky split)", "v2 (leaky split)", "v3"])
display(summary_df.round(4)); summary_df.to_csv(cfg.OUT_DIR / "metrics_summary_v3.csv", index=False, encoding="utf-8-sig")
display(pd.DataFrame(S["shift"]))
display(pd.DataFrame([dict(clip=v["clip"], v1_v2="_ _ _ _ / (ไม่มีอะไร)", v3=v["words"], expected=v["expected"],
                           known_recall=v["known_recall"]) for v in S.get("videos", [])]))"""),
(CODE, r"""# apples-to-apples: v3 บน split เดิมของ v1/v2 (pattern รั่ว + isolated สุ่ม)
from slm_labs.data import make_splits as mk
iso_old, sent_old = mk(iso_user, iso_expert, sent_all, mode="random", sent_by_pattern=False)
print("pattern overlap train/test ของ split เดิม =",
      len({i["pattern"] for i in sent_old["train"]} & {i["pattern"] for i in sent_old["test"]}), "(ควรเป็น 0 ถ้าไม่รั่ว)")
if RETRAIN:
    seed_all(cfg.SEED); m_old = build_model(N_CLASSES, DEVICE)
    train_stage_a(m_old, make_loader(iso_old["train"], cfg.MAX_FRAMES_ISO, True, False),
                  make_loader(iso_old["val"], cfg.MAX_FRAMES_ISO, False, False), cfg.EPOCHS_A, tag="v3_oldsplit")
else:
    m_old = build_model(N_CLASSES, DEVICE); load_checkpoint(m_old, cfg.OUT_DIR / "v3_oldsplit_best.pt")
r_old = eval_isolated(m_old, make_loader(iso_old["test"], cfg.MAX_FRAMES_ISO, False, False))
print(f"v3 บน split เดิม: isolated TEST acc={r_old['acc']:.4f} F1={r_old['macro_f1']:.4f}   "
      f"(v1=0.9759 / v2=0.3373 / v3 บน split ที่ไม่รั่ว={S['stage_a']['acc']:.4f})")"""),

(MD, r"""### สิ่งที่ยังทำต่อได้
- **segmentation** ยังเป็นจุดคอขวดของการอ่านทั้งประโยค — ทางที่ตรงที่สุดคือเก็บ boundary annotation ของ
  sentence clip สัก 50-100 คลิป แล้ว train boundary detector แทน heuristic
- **ข้อมูล** — 51 คำ / signer 3 คน / 76 pattern เล็กเกินไปสำหรับ continuous recognition; expert augmented 32k คลิป
  ใช้เป็น unlabeled pool ได้ถ้าจะกลับไปทำ SSL อีกครั้ง (คราวนี้วัดผลบน split ที่ไม่รั่ว)
- **memory** — เมื่อ annotate สะสมพอ ให้เอา segment เหล่านั้นมาเป็น isolated sample แล้ว fine-tune Stage A เพิ่ม class จริง
- **open-set** — ตอนนี้ใช้ threshold เดียวทั้งระบบ; per-class threshold หรือ energy-based OOD น่าจะดีกว่า"""),
]


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [(nbf.v4.new_markdown_cell if t == MD else nbf.v4.new_code_cell)(src) for t, src in CELLS]
    nb.metadata = {"kernelspec": {"display_name": "hugging", "language": "python", "name": "hugging"},
                   "language_info": {"name": "python", "version": "3.10.16"}}
    p = ROOT / "SLM_Labs_main.ipynb"
    nbf.write(nb, p)
    print("wrote", p, f"({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
