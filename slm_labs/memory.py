"""Sign memory: เก็บ embedding + metadata ของทุก segment (known / unknown) ลง Qdrant (local) + ไฟล์ (npz / jsonl)

flow
  run_pipeline(video) → slots → memory.add_clip(...)         (unknown = '_' พร้อม embedding + ช่วงเวลา + ชื่อคลิป)
  user annotate ทีหลัง → memory.annotate("ไปทานข้าวด้วยกันมั้ย", {0: "ไป", 1: "ทาน", ...})
  ครั้งหน้า analyze_clip() ใช้ memory.nearest_learned(emb) → คำที่เคย annotate กลับมาเป็นคำที่รู้จัก (ไม่ต้อง retrain)
  retrieve(emb) → ส่ง context ให้ LLM
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import cfg
from .vocab import annotate_from_filename, UNK

NS = uuid.UUID("8d7b7e1a-4c1e-4d0f-9a7e-5c2d3f4a5b6c")


def _pid(clip: str, idx: int) -> str:
    return str(uuid.uuid5(NS, f"{clip}#{idx}"))


class SignMemory:
    def __init__(self, root: Path | None = None, collection: str = cfg.QDRANT_COLLECTION, dim: int = cfg.EMB_DIM):
        self.root = Path(root or cfg.MEMORY_DIR); self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "embeddings").mkdir(exist_ok=True); (self.root / "annotations").mkdir(exist_ok=True)
        self.jsonl = self.root / "segments.jsonl"
        self.collection, self.dim = collection, dim
        self.backend = "qdrant"
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self.client = QdrantClient(path=str(self.root / "qdrant"))
            if not self.client.collection_exists(collection):
                self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        except Exception as e:  # fallback: in-memory numpy store persisted as npz
            print("qdrant unavailable → numpy fallback:", type(e).__name__, str(e)[:100])
            self.backend = "numpy"; self.client = None
            self._np = self._load_np()

    # ---------- numpy fallback ----------
    def _np_path(self):
        return self.root / "memory_np.json"

    def _load_np(self):
        p = self._np_path()
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return {k: dict(vector=np.array(v["vector"], np.float32), payload=v["payload"]) for k, v in d.items()}
        return {}

    def _save_np(self):
        self._np_path().write_text(json.dumps({k: dict(vector=v["vector"].tolist(), payload=v["payload"]) for k, v in self._np.items()},
                                              ensure_ascii=False), encoding="utf-8")

    # ---------- core ops ----------
    def _upsert(self, points):
        if self.backend == "qdrant":
            from qdrant_client.models import PointStruct
            self.client.upsert(self.collection, points=[PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in points])
        else:
            for i, v, p in points:
                self._np[i] = dict(vector=v, payload=p)
            self._save_np()

    def _all(self, with_vectors=True):
        if self.backend == "qdrant":
            out, off = [], None
            while True:
                recs, off = self.client.scroll(self.collection, limit=512, offset=off, with_vectors=with_vectors, with_payload=True)
                out += [dict(id=str(r.id), vector=np.array(r.vector, np.float32) if with_vectors else None, payload=r.payload) for r in recs]
                if off is None:
                    break
            return out
        return [dict(id=k, vector=v["vector"], payload=v["payload"]) for k, v in self._np.items()]

    def count(self):
        return self.client.count(self.collection).count if self.backend == "qdrant" else len(self._np)

    def add_clip(self, clip_name: str, slots: list, extra: dict | None = None, store_known=True) -> int:
        """เก็บทุก slot (known+unknown, ไม่รวม null) ของคลิป → qdrant + embeddings/<clip>.npz + segments.jsonl"""
        ann = annotate_from_filename(clip_name); now = datetime.now().isoformat(timespec="seconds")
        points, rows = [], []
        for s in slots:
            if s["status"] == "null" or (s["status"] == "known" and not store_known):
                continue
            payload = dict(clip=clip_name, slot_idx=s["idx"], start_frame=s["start"], end_frame=s["end"],
                           t_start_ms=s["t_start_ms"], t_end_ms=s["t_end_ms"], n_frames=s["n_frames"],
                           word=s["word"], status=s["status"], source=s["source"], conf=round(float(s["conf"]), 4), sim=round(float(s["sim"]), 4),
                           nearest_known=s["nearest"], label=(s["word"] if s["status"] in ("known", "learned") else None),
                           candidate_sentence=ann["sentence"], candidate_tokens=ann["tokens"], known_hits_in_name=ann["known_hits"],
                           synonym_hints=ann["hints"], created_at=now, **(extra or {}))
            points.append((_pid(clip_name, s["idx"]), s["emb"], payload)); rows.append(payload)
        if not points:
            return 0
        self._upsert(points)
        np.savez(self.root / "embeddings" / f"{clip_name}.npz", emb=np.stack([p[1] for p in points]),
                 slot_idx=np.array([p[2]["slot_idx"] for p in points]), meta=json.dumps(rows, ensure_ascii=False))
        with open(self.jsonl, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(points)

    def retrieve(self, emb: np.ndarray, k=5, status: str | None = None, exclude_clip: str | None = None):
        """nearest segments (score = cosine) พร้อม payload"""
        if self.backend == "qdrant":
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            flt = Filter(must=[FieldCondition(key="status", match=MatchValue(value=status))]) if status else None
            res = self.client.query_points(self.collection, query=emb.tolist(), limit=k + (5 if exclude_clip else 0), query_filter=flt, with_payload=True).points
            hits = [dict(score=float(r.score), **r.payload) for r in res]
        else:
            hits = []
            for v in self._np.values():
                if status and v["payload"]["status"] != status:
                    continue
                hits.append(dict(score=float(v["vector"] @ emb), **v["payload"]))
            hits.sort(key=lambda h: -h["score"])
        if exclude_clip:
            hits = [h for h in hits if h["clip"] != exclude_clip]
        return hits[:k]

    def annotate(self, clip_name: str, labels: dict | list, note: str = "") -> int:
        """labels: {slot_idx: word} หรือ list ตามลำดับ slot ที่เก็บ (ข้ามด้วย None/'_') → status='learned'"""
        recs = [r for r in self._all(with_vectors=False) if r["payload"]["clip"] == clip_name]
        recs.sort(key=lambda r: r["payload"]["slot_idx"])
        if isinstance(labels, list):
            labels = {r["payload"]["slot_idx"]: w for r, w in zip(recs, labels)}
        n = 0
        for r in recs:
            w = labels.get(r["payload"]["slot_idx"])
            if not w or w == UNK:
                continue
            payload = dict(label=w, word=w, status="learned", annotated_at=datetime.now().isoformat(timespec="seconds"), note=note)
            if self.backend == "qdrant":
                self.client.set_payload(self.collection, payload=payload, points=[r["id"]])
            else:
                self._np[r["id"]]["payload"].update(payload)
            n += 1
        if self.backend == "numpy":
            self._save_np()
        (self.root / "annotations" / f"{clip_name}.json").write_text(json.dumps(dict(clip=clip_name, labels={str(k): v for k, v in labels.items()}, note=note), ensure_ascii=False, indent=1), encoding="utf-8")
        return n

    def learned_prototypes(self) -> dict:
        """word → dict(vec (E,), n)  จาก segments ที่ status == learned (annotate แล้ว)"""
        acc = {}
        for r in self._all():
            p = r["payload"]
            if p.get("status") == "learned" and p.get("label"):
                acc.setdefault(p["label"], []).append(r["vector"])
        out = {}
        for w, vs in acc.items():
            v = np.mean(vs, 0); out[w] = dict(vec=v / (np.linalg.norm(v) + 1e-8), n=len(vs))
        return out

    def nearest_learned(self, emb: np.ndarray):
        protos = self.learned_prototypes()
        if not protos:
            return None
        best = max(protos.items(), key=lambda kv: float(kv[1]["vec"] @ emb))
        return dict(word=best[0], sim=float(best[1]["vec"] @ emb), n=best[1]["n"])

    def table(self) -> pd.DataFrame:
        rows = [dict(id=r["id"], **{k: v for k, v in r["payload"].items() if k not in ("nearest_known", "candidate_tokens", "synonym_hints")})
                for r in self._all(with_vectors=False)]
        cols = ["clip", "slot_idx", "start_frame", "end_frame", "t_start_ms", "t_end_ms", "word", "status", "source", "conf", "sim", "label", "candidate_sentence", "known_hits_in_name", "created_at"]
        df = pd.DataFrame(rows)
        return df[[c for c in cols if c in df.columns]].sort_values(["clip", "slot_idx"]) if len(df) else df

    def clear(self):
        if self.backend == "qdrant":
            from qdrant_client.models import Distance, VectorParams
            self.client.delete_collection(self.collection)
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
        else:
            self._np = {}; self._save_np()
        if self.jsonl.exists():
            self.jsonl.unlink()
        for p in (self.root / "embeddings").glob("*.npz"):
            p.unlink()

    def close(self):
        if self.backend == "qdrant":
            try:
                self.client.close()
            except Exception:
                pass
