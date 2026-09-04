# SLM Labs — Architecture

> v1 (Demo 1) = ส่วนที่ 1–9 ด้านล่าง (ออกแบบครั้งแรก) · **v2 = ส่วนที่ 10–14** (เพิ่มเพื่อให้ inference จริงใช้ได้: open-vocabulary, memory, SSL)
> code: [`slm_labs/`](slm_labs) · ห้องทดลอง: [`SLM_Labs_main.ipynb`](SLM_Labs_main.ipynb) · ผลเทียบเก่า/ใหม่: `prompt/result/fixed_v1.md`

## 1. Input
Namonpas/thai-sign-language-tsl51 · Datasets at Hugging Face (51 signs + null_act, 76 sentence patterns) + วิดีโอจริงใน `data_test/*.mp4` (ชื่อไฟล์ = ประโยคที่ annotate ไว้)

## 2. Data Processing
ทุก Video -> Frame Sampling (time-based → 10 fps เท่า dataset ไม่ว่ากล้อง 24/25/30/60) -> MediaPipe Holistic -> Hand + Pose + Face landmarks เก็บเป็น T x D (schema เดียวกับ CSV ใน dataset) — `slm_labs/extractor.py`

## 3. Feature Engineering — `slm_labs/features.py`
Hand: x,y,z + Δx,Δy,Δz + presence (254) · Body: x,y,z + relative + velocity (51) · Face: 6 จุด + Δ + geometry (39)
normalize ด้วยกึ่งกลางไหล่ / ความกว้างไหล่ (translation + scale invariant) → Hand H_t, Body B_t, Face F_t
**v2:** `hand_activity()` (energy ของมือ + ข้อมือยกสูงกว่าระดับพัก) · `random_view()` (global/local temporal crop + spatial aug สำหรับ SSL) · `time_reverse()` (pseudo-unknown)

## 4. Multimodal Encoder (Main Model) — `slm_labs/model.py`
Modality MLP ×3 → fusion → learned pos-emb → Transformer Encoder (pre-LN, GELU) (temporal sequence — ไม่ใช้ LSTM)
heads: `cls_head` (CE) · `ctc_head` (CTC) · `face_proj` (face cue) · **v2: `emb_head` (128-d, L2-normalised) สำหรับ prototype / open-set / memory**

## 5. Output ของ Vision Model
ไม่ output sentence โดยตรง → sign sequence + face representation
**v2:** output เป็น *slots* ต่อ segment: `{start,end,t_ms, word | "_", status: known/unknown/learned/null, conf, sim, emb}`
```
{"signs": ["_", "ข้าว", "_", "ด้วยกัน", "_"], "face_embedding": [...], "unknown": [{"slot": 0, "t": [0.0, 1.9], "emb": [...]}, ...]}
```

## 6. LLM — `slm_labs/llm.py`
Sign tokens + blanks + face cue + retrieval hints (จาก memory) → LLM (OpenAI Responses API, `gpt-5.6`; fallback rule-based) → Thai ธรรมชาติ **คง `___` ตำแหน่งคำที่ไม่รู้** (ห้ามเดาใส่ประโยค; เดาได้เฉพาะ field `guesses`)

## 7. Face / Emotion
Face encoder → face embedding + heuristic cue (คิ้ว/ปาก เทียบ baseline) → LLM ใช้เป็น contextual cue (question-like → "?")

## 8. TTS — `slm_labs/tts.py`
Thai (เฉพาะส่วนที่รู้ — `___` ถูกข้าม) → `facebook/mms-tts-tha` (open-source, uroman) หรือ OpenAI TTS → .wav

## 9. Training จริงแบ่งเป็น 2 Model — `slm_labs/train.py`
Model A (Sign Encoder): TSL-51 · Stage A isolated (CE + label smoothing) → Stage B continuous (CTC + CE เสริม)
Model B (LLM): pretrained + prompt/interface
Objective: Classification (CE) + CTC

---

## 10. (v2) SSL — SignDINO-style global/local self-distillation — `model.SignDINO`, `train.train_ssl`
ก่อน Stage A: student เห็น 2 global views (60–100% ของคลิป) + 2 local views (20–45%) พร้อม spatial aug (หมุน/ย่อขยาย/เลื่อน/mirror/hand-drop) ; teacher = EMA เห็นเฉพาะ global
loss = Σ CE(teacher(global_j), student(view_i)) + centering/sharpening → local view ของ "บางคำ" map ไปที่ representation เดียวกับ global view ของทั้งประโยค → ทนต่อกล้อง/สถานที่/ผู้ใช้ต่างกัน (ไม่ใช้ label — ใช้ทุกคลิปได้)

## 11. (v2) Segmentation ไม่พึ่ง vocab — `openset.segment_timeline`
ช่วง "กำลังทำท่า" = ข้อมือยก ∨ (มือปรากฏ ∧ energy สูง) → ตัดที่ valley ของ energy (`find_peaks`) → รู้ว่า "มีกี่คำ" แม้ไม่รู้จักคำ ; `tune_segmentation()` calibrate พารามิเตอร์จากจำนวน gloss ใน sentence-train

## 12. (v2) Open-set decision ต่อ segment — `openset.analyze_clip`
- global view: forward ทั้งคลิป → CTC tokens (+conf, +frame) + frame embeddings
- local view: forward เฉพาะ segment → cls prob + embedding ; emb = normalise(local + global-segment)
- prototype ต่อ class = mean emb ของ isolated-train ; `τ_sim` = 5th-percentile ของ cosine(correct class) บน val
- **known** ⇐ (CTC conf ≥ 0.5) ∧ (cos(emb, proto[token]) ≥ τ_sim) · **local rescue** ⇐ local cls conf ≥ 0.7 ∧ nearest proto ตรงกัน · **null** ⇐ ใกล้ null_act ที่สุด · **learned** ⇐ cos กับ learned prototype (memory) ≥ 0.8 · ไม่งั้น **unknown → `_`**

## 13. (v2) Memory — `memory.SignMemory` (Qdrant local + ไฟล์)
ทุก segment (known/unknown) → `memory/qdrant/` (cosine, 128-d) + `memory/embeddings/<clip>.npz` + `memory/segments.jsonl`
payload: clip, slot_idx, start/end frame, t_start/t_end ms, word, status, source, conf, sim, nearest_known, candidate_sentence (ชื่อไฟล์), known_hits_in_name, synonym_hints, label
`annotate(clip, {slot_idx: word})` → status=learned → `learned_prototypes()` → ครั้งหน้ารู้จักโดยไม่ retrain · `retrieve(emb)` → hints ให้ LLM

## 14. (v2) Evaluation เพิ่ม
shifted-domain test (aug แรงกว่า train บน test set) · prototype-NN accuracy · open-set AUROC / TNR@95%TPR (known vs time-reversed + real unknown segments) · segment-count MAE (motion-only vs model-assisted) · real-video known-recall / #unknown stored

## Pipeline (v2) — `pipeline.SLMPipeline.run(video)`
```
.mp4 → extractor (cache CSV) → features → analyze_clip (slots) → memory.add_clip → llm_translate (คง ___) → tts (พูดส่วนที่รู้)
     → outputs/inference/<clip>.json (+ _tts.wav, _overlay.mp4)
```
