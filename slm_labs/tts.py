"""TTS: natural Thai (เฉพาะคำที่รู้ / ประโยคที่ LLM เรียบเรียง; '___' จะถูกข้าม) → .wav"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch

from .config import cfg

_mms = {}


def speakable(text: str) -> str:
    """ตัด '___' / '_' ออกก่อนพูด (คำที่ไม่รู้ = เว้นไว้)"""
    t = re.sub(r"_{1,}", " ", text)
    return re.sub(r"\s+", " ", t).strip()


def tts_mms(text, out_path):
    from scipy.io import wavfile
    if not _mms:
        from transformers import VitsModel, AutoTokenizer
        _mms["tok"] = AutoTokenizer.from_pretrained("facebook/mms-tts-tha")
        _mms["m"] = VitsModel.from_pretrained("facebook/mms-tts-tha").eval()
    tok, m = _mms["tok"], _mms["m"]
    if getattr(tok, "is_uroman", False):
        import uroman as ur
        text = ur.Uroman().romanize_string(text)
    with torch.no_grad():
        wav = m(**tok(text, return_tensors="pt")).waveform[0].numpy()
    wavfile.write(out_path, m.config.sampling_rate, (wav * 32767).astype(np.int16))
    return out_path


def tts_openai(text, out_path):
    from openai import OpenAI
    r = OpenAI().audio.speech.create(model=cfg.OPENAI_TTS_MODEL, voice=cfg.OPENAI_TTS_VOICE, input=text, response_format="wav")
    Path(out_path).write_bytes(r.content)
    return out_path


def tts(text, out_path, backend=None):
    """→ path หรือ None (ถ้า backend off / ไม่มีอะไรจะพูด / error)"""
    backend = backend or cfg.TTS_BACKEND
    text = speakable(text or "")
    if backend == "off" or not text:
        return None
    try:
        if backend == "openai" and os.environ.get("OPENAI_API_KEY"):
            return tts_openai(text, out_path)
        return tts_mms(text, out_path)
    except Exception as e:
        print("TTS error (skipped):", type(e).__name__, str(e)[:120])
        return None
