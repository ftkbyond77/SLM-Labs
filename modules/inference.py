"""End-to-end inference: clip -> gloss (or unknown) -> Thai sentence -> speech.

Two input routes are supported:
  1. ``.npy`` skeleton clips in the same (T, 150) TSL-ONE-S layout;
  2. raw video (``.mp4/.mov/.avi/.webm``), landmarked on the fly with the
     MediaPipe Tasks API so the pipeline works on genuinely new recordings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CFG, TEST_DIR, load_openai_key
from .feature_engineering import apply_standardiser, clip_to_streams

VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
UNKNOWN = "_"


def aspect_correct(clip: np.ndarray, aspect: float) -> np.ndarray:
    """Undo the anisotropy that MediaPipe's normalised coordinates introduce.

    MediaPipe divides x by the frame width and y by the frame height
    *independently*, so the same body recorded at 16:9 and at 1:1 produces
    different geometry: every horizontal distance is squashed by the aspect
    ratio relative to every vertical one. Nothing downstream removes this - the
    body-frame normalisation in `feature_engineering` divides both axes by a
    single scalar (shoulder width), which cannot undo an axis-dependent scale.

    Measured on this workspace: shoulder-width / torso-height is 0.79 across all
    29 corpus signers - which is also the anatomical value for a real person -
    and 0.39 on the two 16:9 recordings in `data_test/`. The corpus is therefore
    effectively square-pixel and the new recordings are not, and every geometric
    feature on them was distorted by ~2x along x.

    Multiplying x by width/height restores square pixels. Scaling is about the
    frame centre so the skeleton stays roughly in frame for plotting; the
    features are neck-centred anyway, so the choice of centre is cosmetic.
    """
    out = np.asarray(clip, dtype=np.float32).copy()
    miss = (out == 0).all(-1)
    out[..., 0] = 0.5 + (out[..., 0] - 0.5) * float(aspect)
    out[miss] = 0.0
    return out


def video_aspect(path: Path) -> float:
    """Frame width / height of a video file, or NaN if it cannot be read."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        return float(w / h) if w and h else float("nan")
    except Exception:
        return float("nan")


def body_aspect_ratio(clip: np.ndarray) -> float:
    """shoulder width / torso height, median over the clip.

    A pure geometry statistic with a known anatomical value (~0.8), so it works
    as an aspect-distortion detector on any recording, corpus or not.
    """
    p = clip[:, :33]
    sw = np.linalg.norm(p[:, 11] - p[:, 12], axis=-1)
    th = np.linalg.norm(0.5 * (p[:, 11] + p[:, 12]) - 0.5 * (p[:, 23] + p[:, 24]),
                        axis=-1)
    ok = (sw > 1e-3) & (th > 1e-3)
    return float(np.median(sw[ok] / th[ok])) if ok.any() else float("nan")


# --------------------------------------------------------- gloss lexicon ---
def gloss_display(gloss_id: str, lexicon: dict | None = None) -> str:
    """Surface form for a gloss id.

    TSL-ONE-S ships numeric gloss ids only; the Thai lemma table is not part of
    the released .npy bundle.  Drop ``{"0133": "สวัสดี", ...}`` into `lexicon`
    (or modules/lexicon.py) and every downstream stage - display, sentence
    composition, TTS - switches to real Thai with no other change.
    """
    if lexicon and gloss_id in lexicon:
        return lexicon[gloss_id]
    return f"GLOSS_{gloss_id}"


# --------------------------------------------------------------- runtime ---
@dataclass
class SignPredictor:
    model: object
    stats: dict
    class_names: list
    prototypes: np.ndarray
    fuse_stats: dict
    threshold: float
    device: object
    lexicon: dict | None = None

    def _streams(self, clip: np.ndarray) -> dict:
        s = clip_to_streams(clip, CFG.feature)
        s = apply_standardiser({k: v[None] for k, v in s.items()}, self.stats)
        return s

    def predict_batch(self, clips: list, topk: int = 5) -> list:
        import torch
        from .openset import fuse, scores
        banks = [self._streams(c) for c in clips]
        xb = {k: torch.from_numpy(np.concatenate([b[k] for b in banks])).to(self.device)
              for k in ("lh", "rh", "pose", "face")}
        self.model.eval()
        with torch.no_grad():
            out = self.model(xb)
        logits = out["logits"].float().cpu().numpy()
        embeds = out["embed"].float().cpu().numpy()
        attn = out["attn"].float().cpu().numpy() if out.get("attn") is not None else None

        sc = scores(logits, embeds, self.prototypes)
        fused, _ = fuse(sc, self.fuse_stats)
        prob = np.exp(logits - logits.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)

        results = []
        for i in range(len(clips)):
            order = np.argsort(-prob[i])[:topk]
            accepted = bool(fused[i] >= self.threshold)
            gid = self.class_names[int(order[0])]
            results.append({
                "gloss_id": gid if accepted else UNKNOWN,
                "text": gloss_display(gid, self.lexicon) if accepted else UNKNOWN,
                "is_known": accepted,
                "confidence": float(prob[i, order[0]]),
                "openset_score": float(fused[i]),
                "proto_similarity": float(sc["proto"][i]),
                "topk": [(self.class_names[int(j)], float(prob[i, j])) for j in order],
                "attention": attn[i] if attn is not None else None,
            })
        return results

    def predict(self, clip: np.ndarray, topk: int = 5) -> dict:
        return self.predict_batch([clip], topk)[0]


# ----------------------------------------------------------- test intake ---
def scan_test_dir(test_dir: Path = TEST_DIR) -> list:
    test_dir = Path(test_dir)
    if not test_dir.exists():
        return []
    items = []
    for f in sorted(test_dir.rglob("*")):
        if f.is_dir():
            continue
        if f.suffix.lower() == ".npy":
            items.append({"path": f, "kind": "npy"})
        elif f.suffix.lower() in VIDEO_EXT:
            items.append({"path": f, "kind": "video"})
    return items


def load_npy_clip(path: Path) -> np.ndarray:
    a = np.load(str(path)).astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 150:
        return a.reshape(len(a), 75, 2)
    if a.ndim == 3 and a.shape[1:] == (75, 2):
        return a
    raise ValueError(f"{path.name}: expected (T,150) or (T,75,2), got {a.shape}")


_TASK_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
}


def extract_from_video(path: Path, cache_dir: Path | None = None,
                       max_frames: int = 300, square_pixels: bool = False):
    """Video -> (T, 75, 2) in the exact TSL-ONE-S landmark order.

    Requires the two MediaPipe Tasks bundles; they are fetched once into
    `cache_dir` (defaults to a temp folder) because mediapipe>=1.0 no longer
    ships the legacy ``mp.solutions`` graphs.
    """
    import urllib.request
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cache_dir = Path(cache_dir or (Path.home() / ".cache" / "mediapipe_tasks"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, url in _TASK_URLS.items():
        p = cache_dir / Path(url).name
        if not p.exists():
            urllib.request.urlretrieve(url, p)
        paths[k] = str(p)

    pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=paths["pose"]),
        running_mode=vision.RunningMode.VIDEO))
    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=paths["hand"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    frames, t = [], 0
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(t / fps * 1000)
        f = np.zeros((75, 2), dtype=np.float32)
        pr = pose_lm.detect_for_video(img, ts)
        if pr.pose_landmarks:
            f[:33] = [[lm.x, lm.y] for lm in pr.pose_landmarks[0]]
        hr = hand_lm.detect_for_video(img, ts)
        for hd, lms in zip(hr.handedness, hr.hand_landmarks):
            # MediaPipe reports the anatomical side; slot 33.. is the pose-left hand
            off = 33 if hd[0].category_name == "Left" else 54
            f[off:off + 21] = [[lm.x, lm.y] for lm in lms]
        frames.append(f)
        t += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    out = np.stack(frames)
    if square_pixels:
        out = aspect_correct(out, W / max(H, 1.0))
    return out


# ------------------------------------------------------- LLM  +  speech ----
def compose_sentence(glosses: list, model: str = "gpt-4o-mini",
                     lexicon: dict | None = None) -> dict:
    """Gloss sequence -> fluent Thai sentence (gloss order is not Thai syntax).

    The LLM is only invoked when a lexicon maps the glosses to real Thai lemmas.
    Numeric gloss ids carry no lexical content, so asking a language model to
    "translate" ``GLOSS_0272 GLOSS_0282`` would produce a confident hallucination;
    in that case the raw gloss string is returned unchanged and reported as such.
    """
    surface = [g if g == UNKNOWN else gloss_display(g, lexicon) for g in glosses]
    naive = " ".join(surface)
    lexical = [s for s in surface if not s.startswith("GLOSS_") and s != UNKNOWN]
    if not lexical:
        return {"sentence": naive, "glosses": surface,
                "source": "no-lexicon (LLM skipped: gloss ids carry no lexical content)"}
    key = load_openai_key()
    if not key:
        return {"sentence": naive, "source": "fallback-join", "glosses": surface}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        r = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=200,
            messages=[
                {"role": "system", "content":
                 "You rewrite Thai Sign Language gloss sequences as one natural "
                 "Thai sentence. The glosses are in signing order, which differs "
                 "from Thai word order, and they omit particles and inflection - "
                 "add them. '_' marks a sign the recogniser could not identify: "
                 "keep '_' in place rather than inventing a word. A token of the "
                 "form GLOSS_nnnn is an unmapped gloss id: keep it verbatim. "
                 "Reply with the Thai sentence only, no explanation."},
                {"role": "user", "content": " ".join(surface)},
            ])
        return {"sentence": r.choices[0].message.content.strip(),
                "source": model, "glosses": surface}
    except Exception as e:
        return {"sentence": naive, "source": f"fallback ({type(e).__name__})",
                "glosses": surface}


def synthesize_speech(text: str, out_path: Path, model: str = "gpt-4o-mini-tts",
                      voice: str = "alloy") -> Path | None:
    """Thai text -> spoken audio. Returns None when no API key is available."""
    key = load_openai_key()
    if not key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=key)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=text) as resp:
        resp.stream_to_file(str(out_path))
    return out_path
