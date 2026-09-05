# scripts_diag — สคริปต์ diagnostic ที่ใช้หา root cause ของ v1/v2

รันด้วย env `hugging` จาก root ของ repo:
```bash
python scripts_diag/diag1.py     # leakage ของ split + isolated acc ของ checkpoint แต่ละ stage + emb_head ถูกฝึกหรือไม่ + CTC blank
python scripts_diag/diag2.py     # domain gap: สถิติ feature ของ dataset เทียบวิดีโอจริง + Stage A บน segment ของคลิปจริง
python scripts_diag/diag3.py     # เทียบ landmark CSV ดิบ dataset เทียบกับที่ extractor ของเราดึงมา
python scripts_diag/diag4.py     # fps จริงของ dataset (t_ms) เทียบ cfg.TARGET_FPS  ← เจอ root cause หลัก
python scripts_diag/diag6.py     # subset ที่ dataset กำหนด, จำนวน signer, timing ของ expert clips
```

`diag1`/`diag2` อ้าง checkpoint ของ v1/v2 (`outputs/v1_stageA_best.pt` ฯลฯ) และ `outputs/items_cache.pkl`
ซึ่งเป็นของรุ่นเดิม — เก็บไว้เพื่อให้ตรวจซ้ำได้ว่าข้อสรุปใน `prompt/result/fixed_v2.md` §1 มาจากไหน
