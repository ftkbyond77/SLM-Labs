"""End-to-end: .mp4 → landmarks → features → open-set extraction (known | '_') → memory (qdrant) → LLM → TTS"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import cfg, DEVICE
from .vocab import N_CLASSES, annotate_from_filename, UNK
from .features import load_landmarks, make_features
from .model import build_model
from .train import load_checkpoint
from .extractor import HolisticExtractor, check_video_quality, extract_cached
from .openset import Prototypes, analyze_clip, slots_to_words
from .memory import SignMemory
from .llm import face_cue, llm_translate, load_face_baseline
from .tts import tts
from .metrics import wer_cer


class SLMPipeline:
    def __init__(self, model=None, protos: Prototypes | None = None, memory: SignMemory | None = None,
                 ckpt: str | Path | None = None, extractor: HolisticExtractor | None = None, verbose=True):
        if model is None:
            model = load_checkpoint(build_model(N_CLASSES, DEVICE), ckpt or cfg.OUT_DIR / "sign_encoder_v2.pt")
        self.model = model.eval()
        self.protos = protos or Prototypes.load(cfg.OUT_DIR / "prototypes.npz")
        self.memory = memory if memory is not None else SignMemory()
        self._extractor = extractor
        fb = cfg.OUT_DIR / "face_baseline.npz"
        if fb.exists():
            load_face_baseline(fb)
        self.verbose = verbose
        (cfg.OUT_DIR / "inference").mkdir(parents=True, exist_ok=True)

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = HolisticExtractor(verbose=self.verbose)
        return self._extractor

    @torch.no_grad()
    def run(self, video_path, expected_words=None, save_memory=True, translate=True, speak=True, show=True, use_cache=True):
        video_path = Path(video_path); clip = video_path.stem; lat = {}
        t = time.perf_counter(); df = extract_cached(self.extractor, video_path) if use_cache else self.extractor.extract(video_path)
        lat["1_mediapipe_s"] = time.perf_counter() - t
        quality = check_video_quality(df, verbose=show)
        t = time.perf_counter(); lm = load_landmarks(df); feat = make_features(lm)
        analysis = analyze_clip(self.model, feat, self.protos, memory=self.memory, t_ms=lm["t_ms"])
        for s in analysis["slots"]:
            s["clip"] = clip
        lat["2_extract_s"] = time.perf_counter() - t
        words = slots_to_words(analysis["slots"])
        cue = face_cue(feat["face"])
        ann = annotate_from_filename(clip)

        thai, prov, extra = "", "none", {}
        if translate:
            t = time.perf_counter(); thai, prov, extra = llm_translate(analysis["slots"], cue["text"], memory=self.memory); lat["3_llm_s"] = time.perf_counter() - t
        wav = None
        if speak and thai:
            t = time.perf_counter(); wav = tts(thai, cfg.OUT_DIR / "inference" / f"{clip}_tts.wav"); lat["4_tts_s"] = time.perf_counter() - t
        n_mem = self.memory.add_clip(clip, analysis["slots"], extra=dict(video=str(video_path), face_cue=cue["text"])) if save_memory else 0
        lat["total_s"] = sum(v for k, v in lat.items() if k != "total_s")

        res = dict(video=str(video_path), clip=clip, n_frames=int(analysis["T"]), words=words,
                   n_known=sum(w != UNK for w in words), n_unknown=sum(w == UNK for w in words),
                   slots=[{k: v for k, v in s.items() if k != "emb"} for s in analysis["slots"]],
                   ctc_tokens=analysis["tokens"], face_cue=cue["text"], thai=thai, llm=prov, llm_extra=extra,
                   wav=str(wav) if wav else None, filename_annotation=ann, memory_stored=n_mem, quality=quality, latency=lat)
        if expected_words:
            res["wer"], res["cer"] = wer_cer([expected_words], [words])
            res["known_recall"] = _known_recall(expected_words, words)
        (cfg.OUT_DIR / "inference" / f"{clip}.json").write_text(json.dumps(_jsonable(res), ensure_ascii=False, indent=1), encoding="utf-8")
        res["_analysis"] = analysis
        if show:
            print(f"\n=== {clip} ===")
            print("  SIGNS :", " ".join(words))
            for s in analysis["slots"]:
                if s["status"] == "null":
                    continue
                print(f"    [{s['idx']}] {s['t_start_ms'] / 1000:5.2f}-{s['t_end_ms'] / 1000:5.2f}s  {s['word']:<8} {s['status']:<8} src={s['source']:<6} conf={s['conf']:.2f} sim={s['sim']:.2f}  nearest={s['nearest'][:2]}")
            print("  FACE  :", cue["text"])
            print("  THAI  :", thai, f"({prov})")
            if extra.get("guesses"):
                print("  LLM guesses for '_':", extra["guesses"])
            print("  file-name annotation:", ann["sentence"], "| known in name:", ann["known_hits"], "| hints:", ann["hints"])
            print("  memory: stored", n_mem, "segments | latency:", {k: round(v, 2) for k, v in lat.items()})
            if expected_words:
                print(f"  WER={res['wer']:.3f} known-recall={res['known_recall']:.2f}")
        return res

    def run_dir(self, folder, expected: dict | None = None, **kw):
        rows = []
        for v in sorted(Path(folder).glob("*.mp4")):
            exp = (expected or {}).get(v.stem)
            r = self.run(v, expected_words=exp, **kw)
            rows.append(dict(clip=v.stem, words=" ".join(r["words"]), n_known=r["n_known"], n_unknown=r["n_unknown"],
                             thai=r["thai"], wer=r.get("wer"), known_recall=r.get("known_recall"), memory_stored=r["memory_stored"],
                             latency_s=round(r["latency"]["total_s"], 2)))
        import pandas as pd
        return pd.DataFrame(rows)


def _known_recall(expected, words):
    exp_known = [w for w in expected if w != UNK]
    if not exp_known:
        return float("nan")
    got = [w for w in words if w != UNK]
    return sum(w in got for w in exp_known) / len(exp_known)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
