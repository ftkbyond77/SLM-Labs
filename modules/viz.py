"""Visualisation helpers - Thai-capable global font, skeletons, heatmaps, curves."""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .config import FONT_DIR, LHAND_SLICE, POSE_SLICE, RHAND_SLICE

# MediaPipe connection lists (upper body + hands) for skeleton drawing
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
              (3, 7), (6, 8), (9, 10), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13),
              (13, 17)]

PALETTE = {"pose": "#3F6FB5", "lh": "#D9553B", "rh": "#2E9E7C",
           "face": "#9B59B6", "grid": "#DDDDDD"}


def use_thai_font(size: int = 11) -> str:
    """Register font/THSarabunNew*.ttf globally so Thai and English both render."""
    registered = []
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
        registered.append(font_manager.FontProperties(fname=str(f)).get_name())
    name = registered[0] if registered else "DejaVu Sans"
    matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["font.size"] = size
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.25
    return name


# ------------------------------------------------------------- skeleton ----
def draw_skeleton(ax, frame: np.ndarray, title: str = "", show_missing: bool = True,
                  bbox: tuple | None = None):
    """frame: (75, 2) raw MediaPipe coords. Image coords -> y is flipped.

    `bbox = (x0, x1, y0, y1)` crops to a region instead of the whole frame -
    used by the gloss identity cards, where the signer occupies a third of the
    image and a full-frame view is too small to read a handshape from.
    """
    pose, lh, rh = frame[POSE_SLICE], frame[LHAND_SLICE], frame[RHAND_SLICE]

    def seg(pts, edges, color, lw=1.6, ms=9):
        ok = ~(pts == 0).all(axis=1)
        for a, b in edges:
            if a < len(pts) and b < len(pts) and ok[a] and ok[b]:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                        color=color, lw=lw, solid_capstyle="round", zorder=2)
        ax.scatter(pts[ok, 0], pts[ok, 1], s=ms, color=color, zorder=3)

    seg(pose, POSE_EDGES, PALETTE["pose"], 2.0, 14)
    if (lh != 0).any():
        seg(lh, HAND_EDGES, PALETTE["lh"], 1.3, 7)
    elif show_missing:
        ax.text(0.02, 0.06, "left hand: not detected", transform=ax.transAxes,
                color=PALETTE["lh"], fontsize=8)
    if (rh != 0).any():
        seg(rh, HAND_EDGES, PALETTE["rh"], 1.3, 7)
    elif show_missing:
        ax.text(0.02, 0.02, "right hand: not detected", transform=ax.transAxes,
                color=PALETTE["rh"], fontsize=8)

    if bbox is None:
        ax.set_xlim(0, 1.05); ax.set_ylim(1.05, 0)
    else:
        ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[3], bbox[2])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def frame_strip(clip: np.ndarray, n: int = 6, title: str = "", label: str = ""):
    idx = np.linspace(0, len(clip) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.6))
    for ax, i in zip(np.atleast_1d(axes), idx):
        draw_skeleton(ax, clip[i], f"t={i}", show_missing=False)
    if label:
        axes[0].text(0.03, 0.97, label, transform=axes[0].transAxes, va="top",
                     fontsize=12, color="#111", bbox=dict(fc="#FFE9A8", ec="none",
                                                          alpha=.9, pad=2))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def motion_heatmap(clips: list, bins: int = 64, which: str = "hands", ax=None):
    """Spatial occupancy of the moving landmarks - where signs actually live."""
    xs, ys = [], []
    for c in clips:
        pts = c[33:75] if which == "hands" else c[:33]
        pts = pts.reshape(-1, 2)
        pts = pts[~(pts == 0).all(axis=1)]
        xs.append(pts[:, 0]); ys.append(pts[:, 1])
    x, y = np.concatenate(xs), np.concatenate(ys)
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[0, 1.05], [0, 1.05]])
    ax = ax or plt.subplots(figsize=(4.6, 4.6))[1]
    ax.imshow(np.log1p(H.T), origin="upper", extent=[0, 1.05, 1.05, 0],
              cmap="magma", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return ax


def training_curves(hists: dict, figsize=(12, 3.4)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for name, h in hists.items():
        axes[0].plot(h["epoch"], h["train_loss"], label=f"{name} train")
        axes[0].plot(h["epoch"], h["val_loss"], "--", label=f"{name} val")
        axes[1].plot(h["epoch"], h["val_acc"], label=name)
        axes[2].plot(h["epoch"], h["val_f1"], label=name)
    for ax, t in zip(axes, ["loss", "val accuracy", "val macro-F1"]):
        ax.set_xlabel("epoch"); ax.set_title(t); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def confusion_heatmap(cm: np.ndarray, ax=None, title: str = ""):
    ax = ax or plt.subplots(figsize=(6.4, 5.6))[1]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("predicted gloss id"); ax.set_ylabel("true gloss id")
    ax.set_title(title); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, label="row-normalised rate")
    return ax


def attention_over_time(attn: np.ndarray, ax=None, title: str = ""):
    ax = ax or plt.subplots(figsize=(7, 2.6))[1]
    im = ax.imshow(attn, aspect="auto", cmap="magma")
    ax.set_xlabel("frame (resampled)"); ax.set_ylabel("sample")
    ax.set_title(title); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.03)
    return ax


def embedding_scatter(emb2d: np.ndarray, labels: np.ndarray, ax=None,
                      title: str = "", n_show: int = 30):
    ax = ax or plt.subplots(figsize=(5.6, 5.2))[1]
    keep = np.isin(labels, np.unique(labels)[:n_show])
    ax.scatter(emb2d[~keep, 0], emb2d[~keep, 1], s=4, c="#DDDDDD", zorder=1)
    sc = ax.scatter(emb2d[keep, 0], emb2d[keep, 1], s=12, c=labels[keep],
                    cmap="tab20", zorder=2)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return ax


def score_hist(known: np.ndarray, unknown: np.ndarray, thr: float, ax=None,
               title: str = ""):
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    ax.hist(known, bins=40, alpha=.65, label="known gloss", color="#2E9E7C", density=True)
    ax.hist(unknown, bins=40, alpha=.65, label="novel gloss", color="#D9553B", density=True)
    ax.axvline(thr, color="#111", ls="--", lw=1.6, label=f"threshold {thr:.2f}")
    ax.set_xlabel("open-set score"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(fontsize=8)
    return ax


# ------------------------------------------------ continuous utterances ----
def utterance_map(clip, spans=None, labels=None, ax=None, energy=None,
                  activity=None, decoded=None, class_names=None, title=""):
    """One utterance on one axis: motion energy, where the signs really are, and
    where the reader thinks they are.

    Truth is drawn as a band along the top, the decode as a band along the
    bottom, so a boundary that drifts is visible as a horizontal offset rather
    than having to be read out of two separate plots.
    """
    from .continuous import motion_energy
    ax = ax or plt.subplots(figsize=(12, 2.8))[1]
    e = energy if energy is not None else motion_energy(clip)
    ax.plot(e / (e.max() + 1e-9), color="#3F6FB5", lw=1.2, label="wrist motion energy")
    if activity is not None:
        f = len(clip) / max(len(activity), 1)
        ax.plot(np.arange(len(activity)) * f, activity, color="#9B59B6", lw=1.4,
                alpha=.85, label="model: sign being articulated")

    def band(items, y0, h, colours, tag):
        for it in items:
            s, t = it["start"], it["end"]
            ax.add_patch(plt.Rectangle((s, y0), t - s, h, color=colours(it),
                                       alpha=.55, lw=0))
            ax.text((s + t) / 2, y0 + h / 2, it["label"], ha="center", va="center",
                    fontsize=8, color="#111")
        ax.text(0, y0 + h / 2, tag, ha="right", va="center", fontsize=8, color="#555")

    if spans is not None and labels is not None:
        items = [{"start": s, "end": t,
                  "label": (class_names[l] if class_names else str(l))}
                 for (s, t), l in zip(spans, labels)]
        band(items, 1.06, 0.12, lambda _: "#7F8C8D", "truth ")
    if decoded is not None:
        items = [{"start": d["start"], "end": d["end"], "label": d["gloss_id"]}
                 for d in decoded]
        band(items, -0.20, 0.12,
             lambda d: "#2E9E7C" if not d["label"].startswith("_") else "#D9553B",
             "decode ")
    ax.set_ylim(-0.24, 1.24); ax.set_xlim(0, len(clip))
    ax.set_yticks([]); ax.set_xlabel("frame")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title(title, fontsize=11)
    return ax


def posteriorgram(prob, blank_id, class_names=None, top_k=8, ax=None, title="",
                  unk_id=None):
    """The sequence head's output over time, restricted to the classes it
    actually used. Plotting all 186 rows would be a black rectangle; the handful
    that carry probability mass is what tells you whether the model committed to
    a sign or hedged between two."""
    ax = ax or plt.subplots(figsize=(12, 2.6))[1]
    mass = prob.max(0).copy()
    mass[blank_id] = 0
    rows = np.argsort(-mass)[:top_k]
    ax.imshow(prob[:, rows].T, aspect="auto", cmap="magma", vmin=0, vmax=1,
              interpolation="nearest")
    lab = []
    for r in rows:
        if unk_id is not None and r == unk_id:
            lab.append("<unk>")
        else:
            lab.append(class_names[r] if class_names else str(r))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("model step (4 video frames each)")
    ax.set_title(title, fontsize=11); ax.grid(False)
    return ax


def segment_strip(clip, segments, title="", key="gloss_id", max_cols=8):
    """A mid-stroke skeleton per decoded sign - the visual check that the reader
    cut the utterance where a person would."""
    segs = segments[:max_cols]
    if not segs:
        fig, ax = plt.subplots(figsize=(4, 2.4))
        ax.text(.5, .5, "no segments decoded", ha="center"); ax.axis("off")
        return fig
    fig, axes = plt.subplots(1, len(segs), figsize=(2.5 * len(segs), 3.0),
                             squeeze=False)
    for ax, r in zip(axes[0], segs):
        draw_skeleton(ax, clip[min((r["start"] + r["end"]) // 2, len(clip) - 1)],
                      show_missing=False)
        known = r.get("is_known", True)
        sub = r.get("thai") or r.get("best_guess") or ""
        ax.set_title(f"{r[key]}\n{sub}  f{r['start']}-{r['end']}", fontsize=9,
                     color="#1B7F5A" if known else "#C0392B")
    fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout()
    return fig


def curriculum_curves(hists: dict, figsize=(12, 3.2)):
    """Loss and validation WER across curriculum stages, laid end to end."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    off = 0
    for name, h in hists.items():
        x = np.array(h["epoch"]) + off
        axes[0].plot(x, h["ctc"], label=f"{name} CTC")
        axes[0].plot(x, h["frame"], "--", label=f"{name} frame")
        axes[1].plot(x, h["val_wer"], label=f"{name} CTC read-out")
        if "val_wer_frame" in h:
            axes[1].plot(x, h["val_wer_frame"], "--", label=f"{name} frame read-out")
        axes[2].plot(x, h["val_exact"], label=name)
        off += len(x)
        for ax in axes:
            ax.axvline(off - .5, color="#BBB", ls=":", lw=1)
    for ax, t in zip(axes, ["training loss", "validation WER",
                            "validation exact-match"]):
        ax.set_xlabel("epoch (stages laid end to end)"); ax.set_title(t)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
