# SLM Labs — System Flow (v3)

> เอกสารนี้อธิบาย **ระบบปัจจุบัน (v3)** · ผลการทดลองเต็ม + สิ่งที่แก้จาก v1/v2 อยู่ใน [`prompt/result/fixed_v2.md`](prompt/result/fixed_v2.md)
> code: [`slm_labs/`](slm_labs) · driver: [`run_v3.py`](run_v3.py) · lab: [`SLM_Labs_main.ipynb`](SLM_Labs_main.ipynb) · env: `conda activate hugging` (torch GPU)

## 1. Pipeline

```
                    ┌──────────────────────────────────────────────────────────────────────┐
                    │  INPUT : .mp4 (fps อะไรก็ได้ / ถ่ายที่ไหนก็ได้ / คำเดี่ยวหรือทั้งประโยค)  │
                    └───────────────────────────────┬──────────────────────────────────────┘
                                                    ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │  extractor.py  MediaPipe Holistic → pose + face + มือซ้าย/ขวา                          │
     │  features.load_landmarks(..., fps=cfg.TARGET_FPS)                                      │
     │     ★ resample ตาม t_ms ให้เป็น **15 fps เท่ากันทั้ง dataset และวิดีโอใหม่**             │
     │       (dataset TSL-51 อยู่ที่ ~24-30 fps ; v1/v2 ดึงวิดีโอใหม่ที่ 10 fps → ผิดกัน 2.5 เท่า) │
     │  features.make_features → Hand 254 | Body 51 | Face 39  (normalise ด้วยความกว้างไหล่)   │
     └───────────────────────────────┬───────────────────────────────────────────────────────┘
                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │  STAGE 0 — Segmentation (ไม่ใช้ vocab)                                                  │
     │    features.hand_activity()   ความเร็วมือ + ข้อมือยกเหนือระดับพัก                        │
     │    openset.segment_timeline() ตัดที่ valley ของ energy  (grid-search บน sentence-train)   │
     │    openset.ctc_boundaries()   ★ CTC frame-posterior เป็น "ตัวเสนอขอบเขต" (ไม่ใช่ตัว decode) │
     │    openset.split_segments()   segment ที่ยาวเกิน SEG_MAX_FRAMES → แตกที่ CTC cut          │
     └───────────────────────────────┬───────────────────────────────────────────────────────┘
                                     │  slots (start, end)  ที่ความละเอียดเวลาเดิม
                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │  SignEncoder (model.py)                                                                │
     │    Hand ─MLP─┐                                                                          │
     │    Body ─MLP─┼─ fusion ─ +pos ─ Transformer(pre-LN, 4 layer, d=256) ─ x                 │
     │    Face ─MLP─┘                                                                          │
     │        x ─ masked-mean ─ cls_head (52)          ← Stage A : CE                          │
     │        x ─ masked-mean ─ emb_head (128-d, L2)   ← Stage A : ★ ArcFace(s=24, m=0.25)      │
     │        x ─ avgpool÷4   ─ ctc_head (52+blank)    ← Stage B : CTC, **encoder ถูก freeze**   │
     └───────────────────────────────┬───────────────────────────────────────────────────────┘
                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │  ต่อ segment (openset.analyze_clip) — forward เหมือนเป็น isolated clip                   │
     │    cand = argmax(cls_head)  หรือ prototype-NN ถ้ามั่นใจกว่าชัดเจน                        │
     │      null     ⇐ cand = null_act (ท่าพัก / transition) → ไม่ output                       │
     │      known    ⇐ cos(emb, prototype[cand]) ≥ τ_sim  ∧  p(cand) ≥ SEG_CLS_CONF             │
     │      learned  ⇐ cos(emb, learned prototype จาก memory) ≥ MEMORY_SIM_THRESHOLD            │
     │      '_'      ⇐ ที่เหลือ = คำนอกคลัง → เก็บ embedding + metadata ไว้                      │
     │    τ_sim / SEG_CLS_CONF calibrate บน isolated-val + sentence-val (ไม่แตะ test)            │
     └───────────────────────────────┬───────────────────────────────────────────────────────┘
                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │  memory.SignMemory — Qdrant local (cosine, 128-d) + memory/embeddings/*.npz + jsonl      │
     │    payload: clip, slot_idx, ช่วงเฟรม/เวลา, word/status/source, conf, sim, nearest_known,  │
     │             candidate_sentence (ชื่อไฟล์), known_hits_in_name, label                      │
     │    annotate(clip, {slot: คำ}) → learned prototype → ครั้งหน้ารู้จักทันที **โดยไม่ retrain**   │
     │    retrieve(emb) → context ให้ LLM                                                       │
     └───────────────────────────────┬───────────────────────────────────────────────────────┘
                                     ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │  llm.py   gloss + '_' + face cue (คิ้ว/ปาก) + retrieval hints → ประโยคไทย (คง '___')      │
     │  tts.py   MMS-TTS-tha → .wav  (speakable() ตัด '___' ออกก่อนพูด)                          │
     └───────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. การเทรน (train.py)

| stage | ข้อมูล | loss | แตะพารามิเตอร์อะไร |
|---|---|---|---|
| **A** | isolated clips (expert 2 signer + user[calib]) | `CE(cls_head)` + `0.5 · ArcFace(emb_head)` | ทั้ง encoder |
| **B** | sentence clips + synthetic sentences | `CTC(ctc_head)` | **`ctc_head` 13.6k พารามิเตอร์เท่านั้น — encoder freeze** |

Stage B ที่ freeze encoder ทำให้ isolated accuracy หลัง Stage B **เท่ากับก่อน Stage B เป๊ะ ๆ**
(v2 เคยตกจาก 0.988 → 0.337 เพราะ Stage B fine-tune ทั้งโมเดล)

## 3. การแบ่งข้อมูล (data.py) — ไม่มี leakage

| | train | val | test |
|---|---|---|---|
| isolated | expert_primary_02/03 (signer 2 คน) + user[`subset=calib`] | user[`subset=test`] ครึ่งหนึ่ง | อีกครึ่ง |
| sentence | 44 pattern | 9 pattern | 9 pattern — **ไม่ซ้ำกับ train เลย** |

`data.split_report()` ตรวจว่า id และ pattern ไม่ทับกันระหว่าง split (ต้องเป็น 0 ทุกคู่)
โปรโตคอลเดิมของ v1/v2 เอาคลิปที่ 0/1/2 ของ pattern เดียวกันไปใส่ val/test/train → pattern ทับกัน 100%

## 4. สิ่งที่ตัดทิ้งจาก v2 (เพื่อความเรียบง่าย)

- **SignDINO / SSL** — Stage A ของ v2 ได้ 0.988 เทียบ v1 0.976 คือต่างกัน 1 คลิปใน 83 ไม่คุ้มกับ 3 module + 10 hyper-parameter
- **การเอา CTC token มา assign เข้า segment แล้วแตก/รวมซ้ำ** (`_assign_tokens`, local-rescue) — แทนด้วย
  segment → classify ตรง ๆ แล้วใช้ CTC เฉพาะเป็นตัวเสนอขอบเขต
- **การ resample ทั้งคลิปลง MAX_FRAMES_SEQ ก่อนตัด segment** — บิดเวลาของคลิปยาว
