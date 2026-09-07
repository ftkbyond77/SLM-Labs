#!/usr/bin/env python
"""Read a Thai Sign Language recording from the command line.

    python read_video.py data_test/*.mp4
    python read_video.py clip.mp4 --head frame --speak out.mp3 --json out.json

This is the deployment path with nothing else attached: video in, a gloss
sequence with frame spans out, plus a Thai sentence and speech when the lexicon
can supply lemmas. It loads the two checkpoints the notebook produces and does
not need the corpus, the notebook, or a GPU (though it will use one).

Everything it prints is inspectable: for every sign it found it reports where in
the video it is, what it thinks it is, and - when it declines to name it - which
of the two gates refused and why.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.config import ARTIFACT_DIR, ModelConfig, get_device      # noqa: E402
from modules.inference import (VIDEO_EXT, body_aspect_ratio, load_npy_clip,  # noqa: E402
                               synthesize_speech)
from modules.lexicon import Lexicon, compose_sentence                  # noqa: E402
from modules.models import MSSFNet                                     # noqa: E402
from modules.reader import ContinuousReader                            # noqa: E402
from modules.seq_model import CSRNet                                   # noqa: E402


def build_reader(device=None, lexicon: Lexicon | None = None) -> ContinuousReader:
    device = device or get_device()
    csr_ck = ARTIFACT_DIR / "csr_best.pt"
    word_ck = ARTIFACT_DIR / "mssf_best.pt"
    for p in (csr_ck, word_ck):
        if not p.exists():
            raise SystemExit(f"missing checkpoint {p} - run slm_labs.ipynb first")

    ck = torch.load(csr_ck, map_location="cpu", weights_only=False)
    wk = torch.load(word_ck, map_location="cpu", weights_only=False)
    names = ck["class_names"]
    mcfg = ModelConfig(**ck["model_cfg"])

    csr = CSRNet(ck["dims"], len(names), mcfg, use_unk=ck.get("use_unk", True))
    csr.load_state_dict(ck["state_dict"])
    word = MSSFNet(ck["dims"], len(names), mcfg)
    word.load_state_dict(wk["state_dict"])
    csr.to(device).eval()
    word.to(device).eval()

    stats = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32))
             for k, v in ck["stats"].items()}
    return ContinuousReader(
        csr=csr, word=word, stats=stats, class_names=names,
        prototypes=np.asarray(ck["prototypes"]), fuse_stats=ck["fuse_stats"],
        device=device, word_threshold=float(ck.get("word_threshold", -1.47)),
        seq_threshold=float(ck.get("seq_threshold", 0.0)),
        min_run=int(ck.get("min_run", 3)),
        lexicon=lexicon if (lexicon and lexicon.n_filled) else None)


def report(reader: ContinuousReader, path: Path, head: str, lex: Lexicon) -> dict:
    ext = path.suffix.lower()
    if ext == ".npy":
        # a .npy carries no frame dimensions, so the aspect of the recording it
        # came from is unknowable and no correction can be applied
        clip = load_npy_clip(path)
        note = "(.npy input - frame aspect unknown, no correction applied)"
    elif ext in VIDEO_EXT:
        from modules.inference import extract_from_video
        clip = extract_from_video(path, max_frames=900, square_pixels=True)
        note = "(square-pixel corrected)"
    else:
        raise SystemExit(f"unsupported file type: {path}")
    out = reader.read(clip, head=head)

    print(f"\n=== {path.name} ===")
    print(f"{len(clip)} frames {note};  body aspect ratio "
          f"{body_aspect_ratio(clip):.3f} (anatomical ~0.80)")
    print(f"{out['n_signs_seen']} sign(s) localised, read-out '{out['head']}'")
    if not out["segments"]:
        print("  nothing above the gates - the reader saw no sign it could commit to")
    for i, s in enumerate(out["segments"]):
        mark = "OK " if s["is_known"] else "-- "
        print(f"  {mark}[{i}] frames {s['start']:>4}-{s['end']:<4} "
              f"({s['frames']:>3}f)  {s['thai']:<12} "
              f"p(seq)={s['confidence']:.3f} open-set={s['openset_score']:+.2f} "
              f"proto={s['proto_similarity']:.3f}")
        if not s["is_known"]:
            print(f"          declined: {s['reason']}  "
                  f"(best guess {s['best_guess']}, word model says "
                  f"{s['word_model_says']})")
    print("  gloss sequence:", " ".join(out["sequence"]) or "-")

    comp = compose_sentence(out["sequence"], lex)
    print(f"  Thai sentence ({comp['source']}): {comp['sentence'] or '-'}")
    out["sentence"] = comp
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video (.mp4/.mov/.avi/.webm/.mkv) or .npy")
    ap.add_argument("--head", choices=("ctc", "frame", "energy"), default="energy",
                    help="energy = motion-energy boundaries + model labels "
                         "(default; the most robust on recordings the model has "
                         "not seen); frame = the frame head's contiguous runs, "
                         "best on corpus-like material; ctc = the sequence head")
    ap.add_argument("--speak", metavar="OUT.mp3",
                    help="synthesise the composed Thai sentence to this file")
    ap.add_argument("--json", metavar="OUT.json", help="write the full result")
    a = ap.parse_args()

    lex = Lexicon.load()
    if lex.n_filled == 0:
        print("note: artifacts/lexicon.json has no Thai lemmas yet, so glosses stay "
              "as ids and the sentence stage is skipped.\n"
              "      Fill it using artifacts/gloss_cards/ to turn the language "
              "layer on.", file=sys.stderr)
    reader = build_reader(lexicon=lex)

    results = []
    for pattern in a.inputs:
        paths = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") \
            else [Path(pattern)]
        for p in paths:
            if not p.exists():
                print(f"skip {p}: not found", file=sys.stderr)
                continue
            r = report(reader, p, a.head, lex)
            results.append({"file": str(p), "n_signs": r["n_signs_seen"],
                            "sequence": r["sequence"],
                            "sentence": r["sentence"]["sentence"],
                            "segments": [{k: v for k, v in s.items()} for s in r["segments"]]})
            if a.speak and r["sentence"]["sentence"]:
                out = synthesize_speech(r["sentence"]["sentence"], Path(a.speak))
                print(f"  speech -> {out}" if out
                      else "  speech skipped (no OPENAI_API_KEY in .env)")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1, ensure_ascii=False,
                                           default=float), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
