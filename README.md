# SLM-Labs — Thai Sign Language → open-vocabulary sign extraction → LLM → Thai → Speech

| อะไร | อยู่ที่ไหน |
|---|---|
| ห้องทดลอง (รันจากบนลงล่าง) | [`SLM_Labs_main.ipynb`](SLM_Labs_main.ipynb) |
| driver รันทุกอย่างรวดเดียว | [`run_v3.py`](run_v3.py) — `python run_v3.py` (เต็ม) / `--fast` (smoke test) |
| config / architecture / pipeline (.py) | [`slm_labs/`](slm_labs) — `config.py` `vocab.py` `features.py` `data.py` `model.py` `train.py` `metrics.py` `extractor.py` `openset.py` `memory.py` `llm.py` `tts.py` `viz.py` `pipeline.py` |
| system flow ปัจจุบัน | [`flow_system.md`](flow_system.md) |
| ผลเทียบเก่า/ใหม่ | [`prompt/result/fixed_v2.md`](prompt/result/fixed_v2.md) (v3, ล่าสุด) · `prompt/result/fixed_v1.md` (v2) |
| วิดีโอทดสอบจริง | `data_test/*.mp4` (ชื่อไฟล์ = ประโยคที่ annotate ไว้) |
| font ไทยสำหรับกราฟ | `font/*.ttf` (ตั้งเป็น global ด้วย `slm_labs.viz.setup_thai_font()`) |
| memory (Qdrant local + npz/jsonl) | `memory/` (สร้างตอน inference) |
| checkpoints / prototypes / ผล inference | `outputs/` |

## รัน
```bash
conda activate hugging                 # env ที่มี torch GPU
python scripts_build_items.py          # ครั้งเดียว: landmark CSV → feature items ที่ cfg.TARGET_FPS (~25 s)
python run_v3.py                       # train + eval ทั้งหมด (~13 นาทีบน RTX 2050) → outputs/summary_v3.json
jupyter lab SLM_Labs_main.ipynb        # หรือ VS Code เลือก kernel "hugging (GPU)"
```
- `SLM_RETRAIN=1` ใน notebook → train ใหม่แทนการโหลด checkpoint จาก `outputs/`
- ต้องมี `OPENAI_API_KEY` ใน `.env` ถ้าจะใช้ LLM (ไม่มี → rule-based fallback)

## ใช้เป็น library
```python
from slm_labs.pipeline import SLMPipeline
pipe = SLMPipeline()                                   # โหลด outputs/sign_encoder_v3.pt + prototypes.npz + memory/
res  = pipe.run("data_test/ไปทานข้าวด้วยกันมั้ย.mp4")   # → res["words"] (มี '_' สำหรับคำนอกคลัง), res["thai"], res["wav"]
pipe.memory.table()                                    # ทุก segment ที่เก็บ พร้อม embedding + ช่วงเวลา + ชื่อคลิป
pipe.memory.annotate("ไปทานข้าวด้วยกันมั้ย", {1: "ทาน", 4: "มั้ย"})   # map ทีหลัง → ครั้งหน้ารู้จักโดยไม่ retrain
```

## หมายเหตุสำคัญเรื่องตัวเลข
ตัวเลขของ v3 มาจาก split ที่ **ไม่มี leakage** (pattern ของประโยคใน test ไม่เคยอยู่ใน train, isolated แบ่งตาม
`subset` ที่ dataset กำหนด) ส่วนตัวเลขของ v1/v2 มาจาก split เดิมที่ pattern ทับกัน 100% — เทียบกันตรง ๆ ไม่ได้
`run_v3.py` จึงเทรน v3 บน split เดิมด้วยเพื่อให้เทียบแบบ apples-to-apples (ดู `fixed_v2.md` §3)
