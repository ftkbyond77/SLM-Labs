"""Vocabulary (52 classes ของ TSL-51) + column schema ของ landmark CSV + helper สำหรับชื่อไฟล์"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# 52 classes (51 signs + null_act) จาก README ของ dataset
SIGN_CLASSES = [
    "แม่_var_1", "คุณ_var_1", "แฟน_var_1", "พี่_var_1", "น้อง_var_1", "ฉัน_var_2", "ฉัน_var_1", "ยาย_var_1", "แมว_var_1",
    "ภาษามือ_var_1", "ขนมปัง_var_1", "ข้าว_var_1", "ไข่_var_2",
    "ถาม_ทำท่ามือถามไปยังผู้นั้น", "เรียก_var_1", "ชอบ_var_1", "เที่ยว_var_1", "ไป_var_2",
    "เหงา_var_1", "แต่งงาน_var_1", "ร้อน_สองมือ", "โสด_var_1", "โกรธ_var_1", "กลัว_var_1", "เหนื่อย_var_1", "สบายดี_var_1",
    "ทำงาน_var_2", "เกิด_var_2", "ง่วง_var_1", "อยู่บ้าน_var_1",
    "กิน_var_1", "เรียน_var_1", "ชื่อ_var_1", "ดี_var_1", "ด้วยกัน_var_1", "หูหนวก_var_2",
    "สวัสดี_อายุเท่ากันหรือน้อยกว่า", "ขอโทษ_อายุเท่ากันหรือน้อยกว่า", "ขอบคุณ_เปิดมือสองข้าง",
    "วันนี้_var_1", "เช้า_var_1", "วันหยุด_var_1", "พรุ่งนี้_var_1", "อย่า_var_1", "ทำไม_var_1", "อะไร_var_1", "ที่ไหน_var_1",
    "กรุงเทพ_var_1", "โรงเรียน_var_1", "บ้าน_var_1", "ตลาด_var_4", "null_act",
]
assert len(SIGN_CLASSES) == 52
NULL_CLASS = "null_act"
N_CLASSES = len(SIGN_CLASSES)
CLS2ID = {c: i for i, c in enumerate(SIGN_CLASSES)}
BLANK = 0                       # CTC blank ; class i → token i+1
UNK = "_"                       # สัญลักษณ์คำที่ไม่รู้จัก (blank slot)


def class_to_word(c: str) -> str:
    """'ฉัน_var_1' -> 'ฉัน', 'ขอบคุณ_เปิดมือสองข้าง' -> 'ขอบคุณ'"""
    if c == NULL_CLASS:
        return "<null>"
    return re.split(r"_(var_\d+|สอง|อายุ|เปิด|ทำท่า)", c)[0]


CLASS2WORD = {c: class_to_word(c) for c in SIGN_CLASSES}
WORD2CLASSES: dict[str, list[str]] = defaultdict(list)
for _c, _w in CLASS2WORD.items():
    WORD2CLASSES[_w].append(_c)
KNOWN_WORDS = sorted({w for w in CLASS2WORD.values() if w != "<null>"}, key=len, reverse=True)

# คำพ้อง/คำใกล้เคียงที่พบบ่อยในชื่อไฟล์ → คำใน vocab (ใช้เป็น "hint" ตอน map ชื่อคลิป เท่านั้น ไม่ใช่ label)
SYNONYMS = {"ทาน": "กิน", "รับประทาน": "กิน", "ผม": "ฉัน", "หนู": "ฉัน", "เธอ": "คุณ", "มั้ย": "?", "ไหม": "?", "มั๊ย": "?"}


def ids_to_words(ids):
    return [CLASS2WORD[SIGN_CLASSES[i]] for i in ids]


# ---------------- landmark CSV schema ----------------
LH_COLS = [f"lh_{a}{i}" for i in range(21) for a in "xyz"]
RH_COLS = [f"rh_{a}{i}" for i in range(21) for a in "xyz"]
POSE_PTS = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]
POSE_COLS = [f"{p}_{a}" for p in POSE_PTS for a in "xyz"]
FACE_PTS = ["lbrow_outer", "lbrow_inner", "rbrow_inner", "rbrow_outer", "mouth_right", "mouth_left"]
FACE_COLS = [f"{p}_{a}" for p in FACE_PTS for a in "xyz"]
ALL_COLS = ["frame", "t_ms"] + LH_COLS + RH_COLS + \
           [f"{p}_{a}" for p in POSE_PTS for a in ["x", "y", "z", "vis", "pres"]] + FACE_COLS

# ---------------- sentence file names ----------------
SUFFIX_RE = re.compile(r"_(no_space|with_space)_\d+$")


def parse_sentence_glosses(stem: str) -> list[str]:
    """'กรุงเทพ_var_1__ฉัน_var_1__เที่ยว_var_1__ชอบ_var_1_no_space_0' -> ['กรุงเทพ_var_1', 'ฉัน_var_1', ...]"""
    stem = SUFFIX_RE.sub("", Path(stem).stem)
    toks = [t for t in stem.split("__") if t]
    out = []
    for t in toks:
        if t in SIGN_CLASSES:
            out.append(t); continue
        t2 = re.sub(r"_\d+$", "", t)
        if t2 in SIGN_CLASSES:
            out.append(t2); continue
        w = class_to_word(t)
        out.append(WORD2CLASSES[w][0] if w in WORD2CLASSES else t)
    return out


def tokenize_thai(text: str) -> list[str]:
    """ตัดคำไทย (pythainlp newmm) — fallback: คืน text ทั้งก้อน"""
    try:
        from pythainlp.tokenize import word_tokenize
        return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]
    except Exception:
        return [text]


def annotate_from_filename(stem: str) -> dict:
    """ชื่อคลิป (ที่ user annotate ไว้ เช่น 'ไปทานข้าวด้วยกันมั้ย') → candidate words

    คืน dict(sentence, tokens, known_hits, hints) — ใช้เป็น metadata เท่านั้น (ไม่ใช่ ground-truth ของ segment)
    """
    stem = Path(stem).stem
    if "__" in stem:  # รูปแบบ gloss ของ dataset
        glosses = parse_sentence_glosses(stem)
        words = [CLASS2WORD.get(g, g) for g in glosses]
        return dict(sentence=" ".join(words), tokens=words, known_hits=[w for w in words if w in KNOWN_WORDS],
                    hints={}, is_gloss=True)
    toks = tokenize_thai(stem)
    known, hints = [], {}
    for t in toks:
        if t in KNOWN_WORDS:
            known.append(t)
        elif t in SYNONYMS and SYNONYMS[t] in KNOWN_WORDS:
            hints[t] = SYNONYMS[t]
    # substring fallback (tokenizer อาจตัดไม่ตรง vocab)
    for w in KNOWN_WORDS:
        if w in stem and w not in known:
            known.append(w)
    return dict(sentence=stem, tokens=toks, known_hits=known, hints=hints, is_gloss=False)
