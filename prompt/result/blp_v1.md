# BLP v1 — TSL-ONE-S Multimodal Sign Language → Speech

**Blueprint, implementation and results.**
Run date: 2026-09-06 · Environment: conda `hugging` (Python 3.10.16, PyTorch 2.7.0+cu128) · GPU: NVIDIA RTX 2050 (4.3 GB)
Deliverables: [`slm_labs.ipynb`](../../slm_labs.ipynb) (executed, 60 cells, 14 figures, 0 errors) · [`modules/`](../../modules) · `artifacts/mssf_best.pt`

---

## 1 · What the data actually is

Before designing anything, the released bundle was inspected rather than assumed.

| Property | Finding |
|---|---|
| Files | 4 152 `.npy` clips in `data/TSL-ONE-Pose/` |
| Filename grammar | `<signer>_<category>_<gloss>.npy` — 29 signers × 8 categories × **184 gloss classes** |
| Array shape | `(T, 150)` → reshaped `(T, 75, 2)`; `T` ranges 16–194, median 54; 240 554 frames total |
| Landmark layout | **MediaPipe Holistic**: `0–32` pose (33), `33–53` left hand (21), `54–74` right hand (21) |
| Coordinates | normalised image space, x/y only (no z, no visibility) |
| Missing data | exact `(0,0)` = "not detected": 8.5 % of frames for the left hand, 7.1 % for the right |
| Samples per gloss | 17–27 |
| Signers per gloss | ≥ 17 for every gloss |

**Correction to the brief.** The task description says *"54 landmarks … and a neck landmark for SPOTER"*. The
released `.npy` does **not** carry the 54-point SPOTER subset — it carries the raw
75-point MediaPipe Holistic set (33 + 21 + 21). Landmark identity was verified
empirically from mean positions (nose at the top-centre, shoulders symmetric about
it, hand blocks tracking their pose wrists) and from the zero-rate pattern, which
matches the hand blocks exactly. The neck is not a landmark; it is derived here as
the shoulder midpoint. Pose landmarks `23–32` (legs) are never zero but are
MediaPipe extrapolations for an upper-body recording and carry no signal, so they
are dropped.

**Consequence for "Face Impression".** There is **no Face Mesh in the release**.
The only facial evidence is pose landmarks `0–10`: nose, six eye points, two ears,
two mouth corners. That is enough for head pose, head motion, gaze line and
mouth-width dynamics — a *coarse* face-impression channel — but **not** for brow,
eyelid or cheek articulation. The face stream was therefore built and fused
exactly as a dense face stream would be, so Face Mesh is a drop-in upgrade, and
its real contribution was **measured by ablation** rather than asserted (§6).

---

## 2 · Blueprint

### 2.1 Architecture — MSSF-Net (Multi-Stream Sign Fusion Network)

```
TSL-ONE-S .npy  (T, 75, 2)
        │
   Preprocessing:  drop-out interpolation → body-frame normalise → ΔX, Δ²X → resample to 64 frames
        │
 ┌──────┴────────┬──────────────┬─────────────────┐
 │               │              │                 │
Left Hand    Right Hand       Pose          Face (coarse)
 MLP-LH        MLP-RH        MLP-P          Face Encoder
 (151→128)     (151→128)     (87→128)        (60→128)
 └──────┬────────┴──────────────┘                 │
        │                                  Face Temporal Encoder (BiGRU)
   Spatial Fusion → 256-D / frame                 │ 128-D / frame
        └────────────────┬─────────────────────────┘
                Cross-Modal Fusion  (manual queries face, gated by tanh(g))
                         │
              Temporal Transformer  (4 pre-norm layers · d=256 · 8 heads)
                         │
                 Attention Pooling → 256-D sign embedding
              ┌──────────┴───────────┐
      Classification Head      Embedding Head (ArcFace, s=24, m=0.2)
         184 glosses              256-D metric space
              │                        │
         Known sign        prototype cosine + margin → Unknown `_`
                         │
              Gloss sequence → LLM → Thai sentence → TTS
```

**Design decisions and the reason for each**

| Decision | Why |
|---|---|
| Per-stream encoders before fusion | hands, body and face differ in dimensionality, noise and missing-data behaviour; a shared encoder would have to average over all three |
| Hand *shape* wrist-local & scale-free, hand *place* kept in body coords | in TSL, handshape and location are independent phonological parameters — collapsing them loses one |
| Learned `tanh(g)` gate on the face channel | lets the model shut a weak modality off instead of being forced to inject noise; the gate value is then readable evidence |
| Transformer over GRU for the temporal trunk | a sign is a stroke bracketed by preparation/retraction; attention can down-weight the brackets, recurrence cannot as cleanly |
| Attention pooling, not mean pooling | same reason — measured in §6, the model concentrates on the middle of the clip |
| ArcFace embedding head alongside softmax | angular margin compacts class clusters, which is what makes prototype distance a usable open-set score |
| 64-frame resampling | median clip is 54 frames; fixed length makes every batch a dense tensor with no padding mask overhead |

### 2.2 Feature engineering

One vectorised pass per clip (`modules/feature_engineering.py`):

1. **Drop-out repair** — `(0,0)` → `NaN` → linear interpolation along time; a per-frame **presence flag** is kept as an explicit feature so the model knows the difference between "hand at rest" and "hand not seen".
2. **Body frame** — origin at the neck (shoulder midpoint), scale = median shoulder width ⇒ translation- and size-invariant.
3. **Hand streams** — 21 wrist-local scale-free points + wrist position in body coords + 5 fingertip-to-wrist distances + finger span.
4. **Pose stream** — 12 upper-body points in body coords + 5 relational distances (inter-hand, each hand↔face, each hand↔neck).
5. **Face stream** — 11 nose-centred, inter-ocular-scaled points + 8 geometric descriptors (mouth width, jaw-drop proxy, ear-span/yaw proxy, eye-mouth/pitch proxy, head roll cos/sin, head shift x/y).
6. **Dynamics** — `ΔX` and `Δ²X` appended to every stream.
7. **Uniform time** — linear resample to 64 frames.

Resulting per-frame dimensionality: **LH 151 · RH 151 · Pose 87 · Face 60 = 449**.

### 2.3 Leakage protocol

A random split is not acceptable here: each signer repeats each gloss, so random
shuffling puts near-duplicate performances of the same person on both sides.

* **Signer-independent split** — test = signers `05, 12, 24`; val = `06, 20`; train = the other 24. Sizes **3 302 / 298 / 552**.
* Test signers were chosen among those with full 184-gloss coverage, so the test set exercises the entire vocabulary.
* **Audited in the notebook (§5)** on: signer overlap (all three pairs), index duplication, dataset coverage, filename overlap, unseen-gloss check, and a **SHA-1 content hash of every raw test clip against every train clip**. The cell asserts and halts on failure.
* The standardiser (per-feature mean/std) is fitted on **train rows only**. Early stopping uses validation. The open-set threshold is calibrated on **validation only**.

**Audit result: PASS** — 0 signer overlaps, 0 duplicate indices, 0 filename overlaps, 0 content-duplicate test clips, full dataset coverage, 184/184 glosses present in each partition.

### 2.4 Optimisation for scale

| Concern | Treatment |
|---|---|
| Disk I/O in the training loop | none — the whole corpus is materialised once as contiguous `float32` `(N,64,D)` tensors (477 MB) and sliced by index |
| Per-item Python work | none — no `Dataset.__getitem__`, no DataLoader workers; a batch is one `index_select` plus one pinned host→device copy |
| Augmentation cost | fully batched **on the GPU**: tempo warp by gathered linear interpolation, amplitude jitter, Gaussian noise, frame dropout, hand-dropout, all as tensor ops |
| Feature pass | vectorised — sliding mean via cumulative sum, resampling via a single two-index gather-and-blend (no `np.interp` per column). 4 152 clips in **~16 s** |
| Numerics | mixed precision (AMP) + gradient clipping; cosine LR with warm-up |

Throughput: **~4.0 s/epoch** for MSSF-Net (52 batches), ~3.0 s/epoch for the baseline.

---

## 3 · Model selection (validation only)

A sweep was run first; the winner was picked on **validation macro-F1**, never on test.

| run | layers | dropout | mixup α | lr | **val macro-F1** |
|---|---|---|---|---|---|
| MSSF-Net A | 4 | 0.20 | 0.2 | 3e-4 | 0.8721 |
| MSSF-Net B (deeper) | 6 | 0.25 | 0.2 | 3e-4 | 0.8591 |
| **MSSF-Net C — selected** | **4** | **0.20** | **0.4** | **4e-4** | **0.8895** |

Depth beyond 4 layers over-fits 3.3 k training clips; stronger mixup with a
slightly higher LR is what actually helps. The selected recipe was then used for
**every** model in the notebook so all comparisons are like-for-like.

---

## 4 · Results — sign recognition (test = 3 unseen signers, 552 clips)

| model | params | accuracy | top-5 | macro-F1 | balanced acc. | ECE |
|---|---|---|---|---|---|---|
| Baseline BiGRU | 2.28 M | 0.8859 | 0.9783 | 0.8797 | 0.8859 | 0.2038 |
| MSSF-Net, face zeroed (retrained) | 4.05 M | 0.9203 | 0.9801 | 0.9182 | 0.9203 | 0.1079 |
| **MSSF-Net (final)** | **4.05 M** | **0.9293** | 0.9728 | **0.9281** | **0.9293** | 0.1147 |

* **+4.8 macro-F1 points over the baseline**, on signers the model has never seen.
* Calibration also improves markedly (ECE 0.204 → 0.115), i.e. the confidences are more trustworthy, which is what makes the rejection stage possible.
* 39 / 552 clips misclassified. The residual errors are concentrated in glosses with only 3 test samples and in a handful of visually adjacent pairs.

---

## 5 · Results — open-set detection

Scoring a closed-set model on its own test set cannot measure unknown-sign
behaviour, so a **probe model** was trained on 160 glosses with **24 glosses held
out entirely**; those 24 are genuine unknowns, produced by unseen signers.

| rejection score | AUROC | AUPR (unknown) |
|---|---|---|
| max-softmax | 0.9346 | 0.6985 |
| prototype cosine | 0.9322 | 0.7015 |
| prototype margin | 0.9154 | 0.5953 |
| **fused (deployed)** | **0.9346** | **0.7040** |

* **OSCR-AUC = 0.8758**
* At the validation-calibrated operating point (90 % known-acceptance target): **92.5 % of known signs accepted, 66.7 % of novel signs correctly rejected**.
* Applied to the production 184-gloss model, rejection turns 92.93 % closed-set accuracy into **89.86 % accuracy with a `_` escape hatch**: 31 clips returned `_`, and **14 of those 31 would have been wrong answers anyway**. The system prefers `_` to a confident lie.

---

## 6 · Results — does the face channel help?

Three independent pieces of evidence:

1. **Signal test.** Between-gloss variance ratio (η²) of the eight face descriptors: `roll_cos` 0.427, `head_dx` 0.172, `eye_mouth` 0.157, `jaw_drop` 0.133, `roll_sin` 0.130, `head_dy` 0.117, `ear_span` 0.076, `mouth_width` 0.048. Head orientation and head translation carry the most gloss-discriminative information; mouth width carries the least — consistent with only having two mouth corners.
2. **Ablation retrain.** Identical architecture, face stream zeroed, same seed and budget: **0.9203 → 0.9293 accuracy (+0.90 pt)**, **0.9182 → 0.9281 macro-F1 (+0.99 pt)**.
3. **Learned gate.** `tanh(g) = +0.060` — small but clearly non-zero; the model keeps the channel rather than switching it off.

**Verdict:** the coarse face channel is a real but modest contributor at this
landmark resolution. It is *not* facial-expression recognition, and the report
does not claim it is. The headroom is in re-extracting TSL-ONE-S with MediaPipe
Face Mesh; the architecture is already wired for it.

---

## 7 · Results — sequence and translation stages

TSL-ONE-S is an **isolated-sign** corpus: no sentence annotations, no paired Thai
translations. To exercise and measure the sequence stage honestly, 80 continuous
utterances (3–6 signs each) were **composed from held-out test clips** — every
clip is still one the model has never seen, from a signer it has never seen — and
read left-to-right by the recogniser with rejection enabled.

| stage | metric | value |
|---|---|---|
| Sign sequence | **WER** | 0.0950 |
| Sign sequence | **CER** | 0.0627 |
| Sign sequence | sentence exact match | 0.5875 |
| Gloss-string output | **BLEU (char-tokenised)** | 93.47 |
| Gloss-string output | **chrF** | 93.26 |

BLEU/chrF use character tokenisation, which is the correct setting for Thai (no
orthographic word boundaries). **These score gloss-string fidelity, not Thai
fluency** — a true SLT BLEU needs paired Thai sentences the corpus does not ship.

---

## 8 · Results — inference

**Route 1 — held-out test set (552 clips, 3 unseen signers).** Closed-set accuracy
0.9293; with rejection on, 0.8986 with 31 `_` outputs. The notebook renders a
visual panel per clip: mid-stroke skeleton, predicted gloss coloured by
correctness, and the top-5 posterior.

**Route 2 — `data_test/`.** **The folder was empty at run time (0 files).** Nothing
new could be scored. The route is implemented and ready in both formats:

* `.npy` of shape `(T,150)` or `(T,75,2)` → `INF.load_npy_clip()`
* raw video `.mp4/.mov/.avi/.webm/.mkv` → `INF.extract_from_video()`, which runs the MediaPipe **Tasks** API (mediapipe 1.0.1 no longer ships the legacy `mp.solutions` graphs) and emits the identical 75-landmark ordering, so the same feature pass applies unchanged.

Drop files into `data_test/` and re-run that single cell — nothing else changes.

**Route 3 — sequence → sentence → speech.** The recogniser emits **gloss ids**,
because the release contains no Thai lemma table. The sentence stage therefore
**declines to call the LLM on opaque ids** and passes them through — asking a
language model to "translate" `GLOSS_0272` produces a confident hallucination,
which would be worse than no output:

```
recognised gloss sequence : 0272 0282 0207 0240 _ 0007
reference gloss sequence  : 0272 0282 0207 0240 0922 0007
sentence stage (no-lexicon: LLM skipped, gloss ids carry no lexical content)
  GLOSS_0272 GLOSS_0282 GLOSS_0207 GLOSS_0240 _ GLOSS_0007
```

The stage is then demonstrated end to end with an explicitly illustrative lexicon,
which shows the LLM doing the job gloss order requires — adding particles and
Thai word order — while **preserving `_` instead of guessing**:

```
gloss sequence (illustrative) : ฉัน ชื่อ _ ยินดี ที่ได้รู้จัก
LLM composition (gpt-4o-mini) : ฉันชื่อ_ ยินดีที่ได้รู้จักค่ะ
TTS (gpt-4o-mini-tts)         : artifacts/inference_demo.mp3
```

Supplying the real lemma table via `lexicon=` is the only change needed to make
that the live path for recognised signs.

---

## 9 · Consolidated evaluation

| Stage | Metric | Value |
|---|---|---|
| Sign recognition (closed-set, unseen signers) | Accuracy | **0.9293** |
| Sign recognition (closed-set, unseen signers) | Macro-F1 | **0.9281** |
| Sign recognition (closed-set, unseen signers) | Top-5 | 0.9728 |
| Sign recognition (open-set, rejection on) | Accuracy | 0.8986 |
| Unknown detection (160 known / 24 novel) | AUROC | 0.9346 |
| Unknown detection (160 known / 24 novel) | OSCR-AUC | 0.8758 |
| Unknown detection (160 known / 24 novel) | Unknown recall @ 90 % known-accept | 0.6667 |
| Sign sequence | WER | 0.0950 |
| Sign sequence | CER | 0.0627 |
| Gloss-string output | BLEU (char) | 93.47 |
| Gloss-string output | chrF | 93.26 |
| Metric-space quality | prototype nearest-neighbour accuracy | 0.9221 |
| Metric-space quality | cosine to own prototype vs. nearest impostor | 0.767 vs 0.385 |

Total notebook wall-clock: **17.7 min** end to end on one RTX 2050 (data load 2 s,
feature engineering 16 s, four model trainings, all evaluation and inference).

---

## 10 · What was built

```
modules/
  config.py               paths, landmark topology, split definition, all hyperparameters
  data_io.py              indexing, loading, signer-independent split, leakage audit
  feature_engineering.py  vectorised raw → 4-stream tensors
  dataset.py              in-RAM StreamBank + batched GPU augmentation + mixup
  models.py               BaselineGRU, MSSFNet, ArcMarginHead, soft-target CE
  train.py                AMP training loop, cosine+warm-up, early stopping
  evaluate.py             recognition / sequence (WER, CER) / translation (BLEU, chrF) / open-set metrics
  openset.py              prototypes, fused rejection score, threshold calibration, OSCR
  inference.py            SignPredictor, video→landmarks, LLM composition, TTS
  viz.py                  Thai-capable global font, skeletons, heatmaps, curves
slm_labs.ipynb            16 sections, executed end to end, 14 figures, 0 errors
artifacts/
  mssf_best.pt            the one production checkpoint (17.1 MB) — weights, standardiser,
                          prototypes, fusion stats, calibrated threshold, class names
  inference_demo.mp3      TTS output of the demonstration sentence
```

Fonts: `font/THSarabunNew*.ttf` registered as the global matplotlib family, so
every figure renders Thai and Latin. LLM and TTS credentials are read from `.env`
and never printed.

---

## 11 · Honest limitations

1. **`data_test/` was empty**, so the "new real-world data" requirement could not be exercised on real material. The code path exists, is typed and is one cell away from running.
2. **No Thai lemma table.** Gloss ids stay ids. BLEU/chrF therefore measure gloss-string fidelity, and the LLM sentence in the notebook is an explicitly-labelled demonstration, not a scored translation result.
3. **No dense facial landmarks.** "Face impression" is head-pose-and-mouth-corner level, not expression recognition. The measured contribution (+0.9 accuracy points) should be read with that in mind.
4. **Isolated signs only.** WER/CER are computed on utterances composed from held-out isolated clips. This measures the recogniser under sequence-level scoring; it does not measure continuous-signing segmentation, which the corpus cannot support.
5. **Small test set per class.** 552 test clips over 184 glosses is ~3 per class, so per-gloss F1 is coarse; the aggregate numbers are the reliable ones.
6. **One split.** Results are for one signer-independent partition, not a cross-validated mean over signer folds.

## 12 · Highest-value next steps

1. Re-extract TSL-ONE-S with **MediaPipe Face Mesh** (468 points) and re-run §10 of the notebook — the fusion path is already in place and this is where the remaining headroom is.
2. Add the **Thai lemma table** to `lexicon` — this activates the LLM and TTS stages for real and makes BLEU/chrF genuine translation metrics.
3. **Leave-one-signer-out cross-validation** over all 29 signers for a variance estimate instead of a single split.
4. **Continuous signing**: sliding-window inference plus CTC over the existing trunk, which would make WER/CER measure real segmentation.
5. Cheap accuracy wins that were not applied here: test-time augmentation and a small seed ensemble.
