"""LLM: sign slots (known words + '_' blanks) + face cue + retrieval hints → natural Thai (คง '___' สำหรับคำที่ไม่รู้)"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .config import cfg
from .vocab import UNK

# ---------------- face cue ----------------
_FACE_BASE = {"mean": None, "std": None}


def set_face_baseline(iso_train_items, save_path=None):
    geo = np.concatenate([it["feat"]["face"][:, -3:] for it in iso_train_items], 0)
    _FACE_BASE["mean"], _FACE_BASE["std"] = geo.mean(0), geo.std(0) + 1e-6
    if save_path:
        np.savez(save_path, mean=_FACE_BASE["mean"], std=_FACE_BASE["std"])


def load_face_baseline(path):
    z = np.load(path); _FACE_BASE["mean"], _FACE_BASE["std"] = z["mean"], z["std"]


def face_cue(face_feat: np.ndarray) -> dict:
    if _FACE_BASE["mean"] is None:
        return dict(text="neutral", z_peak=[0, 0, 0])
    z = (face_feat[:, -3:] - _FACE_BASE["mean"]) / _FACE_BASE["std"]
    peak = np.percentile(z, 80, axis=0); cue = []
    if peak[0] > 1.0: cue.append("eyebrows raised (question-like / emphasis)")
    if peak[0] < -1.0: cue.append("eyebrows lowered / furrowed (negative or wh-question)")
    if peak[1] > 1.0: cue.append("mouth widened (smile / positive)")
    if np.abs(z).max() < 0.7: cue.append("neutral")
    return dict(text=", ".join(cue) or "neutral", z_peak=peak.round(2).tolist())


# ---------------- rule-based fallback ----------------
SUBJ = {"แม่", "คุณ", "แฟน", "พี่", "น้อง", "ฉัน", "ยาย", "แมว"}
OBJ = {"ภาษามือ", "ขนมปัง", "ข้าว", "ไข่"}
PLACE = {"กรุงเทพ", "โรงเรียน", "บ้าน", "ตลาด"}
GREET = {"สวัสดี", "ขอโทษ", "ขอบคุณ"}
TIME = {"วันนี้", "เช้า", "วันหยุด", "พรุ่งนี้"}
QWORD = {"ทำไม", "อะไร", "ที่ไหน"}
STATE = {"เหงา", "โสด", "โกรธ", "กลัว", "เหนื่อย", "สบายดี", "หูหนวก", "ง่วง", "ร้อน", "ดี", "อยู่บ้าน", "ทำงาน"}


def rule_gloss_to_thai(words):
    ws = list(dict.fromkeys(w for w in words if w not in ("<null>", UNK)))
    greet = [w for w in ws if w in GREET]; tm = [w for w in ws if w in TIME]; subj = [w for w in ws if w in SUBJ]
    obj = [w for w in ws if w in OBJ]; place = [w for w in ws if w in PLACE]; q = [w for w in ws if w in QWORD]
    state = [w for w in ws if w in STATE]
    verbs = [w for w in ws if w not in GREET | TIME | SUBJ | OBJ | PLACE | QWORD | STATE]
    S = subj[0] if subj else ""; O2 = subj[1] if len(subj) > 1 else ""
    out = "".join(greet) + ("  " if greet else "") + "".join(tm) + S
    if O2 and "แต่งงาน" in verbs: return out + "แต่งงานกับ" + O2
    if O2 and not verbs and not state: return out + "เป็นแฟนกับ" + O2 if "แฟน" in (S, O2) else out + O2
    if "เกิด" in verbs and "หูหนวก" in state: return out + "เกิดมาหูหนวก"
    core = ""
    if "ชอบ" in verbs: core += "ชอบ"
    if q and "ที่ไหน" in q and tm: core += "จะ"
    if "ไป" in verbs: core += "ไป"
    if "เที่ยว" in verbs: core += "เที่ยว"
    core += "".join(v for v in verbs if v not in ("ชอบ", "ไป", "เที่ยว"))
    core += "".join(obj) + "".join(place) + "".join(state)
    if q: core += "".join(q) + "?"
    return out + core


def rule_slots_to_thai(words):
    """คงลำดับ + blank: ['_','ข้าว','_','ด้วยกัน'] → '___ ข้าว ___ ด้วยกัน' (ถ้าไม่มี blank ใช้ rule เดิม)"""
    if UNK not in words:
        return rule_gloss_to_thai(words)
    return " ".join("___" if w == UNK else w for w in words if w != "<null>")


# ---------------- OpenAI ----------------
LLM_SYSTEM = (
    "You are a Thai Sign Language (TSL) to Thai interpreter. Input is a gloss sequence in TSL word order "
    "(typically topic/object first, verb last, no function words) plus a facial-expression cue. "
    "Some slots are '_' = a sign the recogniser does NOT know (out of vocabulary). "
    "Rewrite as ONE natural Thai sentence with the same meaning. Rules: do NOT invent facts for unknown slots — keep each unknown "
    "slot as '___' in the sentence at the position that reads most naturally; you MAY add function words (กับ, ที่, จะ, ไป, มา, เป็น, ไหม) "
    "and reorder known words; if the face cue is question-like or a question word exists, end with '?'. "
    "If retrieval hints are given for an unknown slot, you may propose a guess for it in the 'guesses' field only (never in the sentence). "
    "Respond with JSON: {\"thai\": \"<sentence with ___ for unknown>\", \"guesses\": {\"<slot index>\": \"<guess or null>\"}, \"note\": \"<short>\"}"
)


def _hints_text(slots, memory=None, k=3):
    lines = []
    for s in slots:
        if s["word"] != UNK:
            continue
        near = ", ".join(f"{w}({sm:.2f})" for w, sm in s.get("nearest", [])[:k])
        mem = ""
        if memory is not None:
            hits = memory.retrieve(s["emb"], k=k, exclude_clip=s.get("clip"))
            hits = [h for h in hits if h.get("label")]
            if hits:
                mem = " | memory: " + ", ".join(f"{h['label']}({h['score']:.2f})" for h in hits)
        lines.append(f"slot {s['idx']} (t={s['t_start_ms'] / 1000:.1f}-{s['t_end_ms'] / 1000:.1f}s): nearest known signs: {near or 'none'}{mem}")
    return "\n".join(lines)


def llm_translate(slots_or_words, cue_text="neutral", memory=None, provider=None, model=None):
    """slots (จาก analyze_clip) หรือ list ของคำ → (thai, provider_used, extra)"""
    if slots_or_words and isinstance(slots_or_words[0], dict):
        slots = [s for s in slots_or_words if s["status"] != "null"]; words = [s["word"] for s in slots]
    else:
        slots = [dict(idx=i, word=w, nearest=[], t_start_ms=0, t_end_ms=0, emb=None) for i, w in enumerate(slots_or_words)]
        words = list(slots_or_words)
    words = [w for w in words if w != "<null>"]
    if not words:
        return "", "none", {}
    if not any(w != UNK for w in words):
        return " ".join("___" for _ in words), "none", {}
    provider = provider or cfg.LLM_PROVIDER; model = model or cfg.LLM_MODEL
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            hints = _hints_text(slots, memory) if any(w == UNK for w in words) else ""
            user = f"GLOSSES (in order): {' | '.join(words)}\nFACE CUE: {cue_text}"
            if hints:
                user += "\nRETRIEVAL HINTS FOR UNKNOWN SLOTS:\n" + hints
            r = client.responses.create(model=model, instructions=LLM_SYSTEM, input=user)
            txt = r.output_text.strip()
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0)) if m else {"thai": txt}
            return d.get("thai", txt).strip(), model, {k: v for k, v in d.items() if k != "thai"}
        except Exception as e:
            print("LLM API error → rule fallback:", type(e).__name__, str(e)[:120])
    return rule_slots_to_thai(words), "rule", {}
