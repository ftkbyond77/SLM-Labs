# SLM-Labs — Thai Sign Language → open-vocabulary sign extraction → LLM → Thai → Speech

| อะไร | อยู่ที่ไหน |
|---|---|
| ห้องทดลอง (รันจากบนลงล่าง) | [`SLM_Labs_main.ipynb`](SLM_Labs_main.ipynb) |
| config / architecture / pipeline (.py) | [`slm_labs/`](slm_labs) — `config.py` `vocab.py` `features.py` `data.py` `model.py` `train.py` `metrics.py` `extractor.py` `openset.py` `memory.py` `llm.py` `tts.py` `viz.py` `pipeline.py` |
| สถาปัตยกรรม (v1 → v2) | [`main_arch.md`](main_arch.md) |
| ผลเทียบเก่า/ใหม่ | `prompt/result/fixed_v1.md` |
| วิดีโอทดสอบจริง | `data_test/*.mp4` (ชื่อไฟล์ = ประโยคที่ annotate ไว้) |
| font ไทยสำหรับกราฟ | `font/*.ttf` (ตั้งเป็น global ด้วย `slm_labs.viz.setup_thai_font()`) |
| memory (Qdrant local + npz/jsonl) | `memory/` (สร้างตอน inference) |
| checkpoints / prototypes / ผล inference | `outputs/` |

## รัน
```bash
conda activate slm_lab
jupyter lab SLM_Labs_main.ipynb        # หรือ VS Code เลือก kernel "slm_lab"
```
- `SLM_FAST=1` → 1 epoch ทุก stage (ทดสอบว่า cell ทั้งหมดรันได้)
- `SLM_RETRAIN=0` → โหลด checkpoint ใน `outputs/` แทนการ train ใหม่
- ต้องมี `OPENAI_API_KEY` ใน `.env` ถ้าจะใช้ LLM (ไม่มี → rule-based fallback)

## ใช้เป็น library
```python
from slm_labs.pipeline import SLMPipeline
pipe = SLMPipeline()                                   # โหลด outputs/sign_encoder_v2.pt + prototypes.npz + memory/
res  = pipe.run("data_test/ไปทานข้าวด้วยกันมั้ย.mp4")   # → words เช่น ['ไป', '_', 'ข้าว', 'ด้วยกัน', '_'] + thai + wav
pipe.memory.table()                                    # ดู segment ที่เก็บ (unknown = '_' พร้อม embedding + เวลา + ชื่อคลิป)
pipe.memory.annotate("ไปทานข้าวด้วยกันมั้ย", {1: "ทาน", 4: "มั้ย"})   # map ทีหลัง → ครั้งหน้ารู้จักโดยไม่ retrain
```
