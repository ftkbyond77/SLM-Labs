"""Visualisation: Thai font (font/*.ttf) เป็น global ของ matplotlib + timeline ของ slots + overlay บนวิดีโอ"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import cfg
from .vocab import UNK

_FONT = {"name": None, "path": None}


def setup_thai_font(font_dir: Path | None = None, prefer="TH Charmonman", size=14):
    """ลงทะเบียน .ttf ทุกไฟล์ใน font/ แล้วตั้งเป็น global font ของ matplotlib → คืน (font name, path)"""
    import matplotlib
    import matplotlib.font_manager as fm
    font_dir = Path(font_dir or cfg.FONT_DIR)
    names = []
    for p in sorted(font_dir.glob("*.ttf")):
        fm.fontManager.addfont(str(p)); n = fm.FontProperties(fname=str(p)).get_name(); names.append((n, p))
    if not names:
        print("no .ttf in", font_dir); return None, None
    pick = next(((n, p) for n, p in names if n == prefer and "Bold" not in p.name), names[0])
    matplotlib.rcParams["font.family"] = pick[0]
    matplotlib.rcParams["font.size"] = size
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT["name"], _FONT["path"] = pick[0], str(pick[1])
    return pick


def font_path():
    return _FONT["path"] or str(next(Path(cfg.FONT_DIR).glob("*.ttf")))


STATUS_COLOR = dict(known="#2a9d8f", learned="#4361ee", unknown="#e76f51", null="#bbbbbb")


def plot_timeline(analysis: dict, title: str = "", fps=cfg.TARGET_FPS, ax=None, show_tokens=True):
    """energy curve + segments (สี = known/unknown/null) + คำ/blank ต่อ segment + CTC tokens"""
    import matplotlib.pyplot as plt
    energy = analysis["energy"]; idx_map = analysis["idx_map"]; T = analysis["T"]
    t = idx_map / fps
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.plot(t, energy, color="#333", lw=1, label="hand energy")
    ymax = float(energy.max() + 1e-6) * 1.15
    for s in analysis["slots"]:
        c = STATUS_COLOR[s["status"]]
        ax.axvspan(s["start"] / fps, s["end"] / fps, color=c, alpha=0.25)
        lab = s["word"] if s["status"] != "null" else "·"
        ax.text((s["start"] + s["end"]) / 2 / fps, ymax * 0.9, lab, ha="center", va="top", fontsize=15, color=c, fontweight="bold")
        if s["status"] == "unknown" and s.get("nearest"):
            ax.text((s["start"] + s["end"]) / 2 / fps, ymax * 0.68, "≈" + s["nearest"][0][0] + f" {s['nearest'][0][1]:.2f}", ha="center", fontsize=9, color="#777")
    if show_tokens:                                    # ผลดิบของ classifier ต่อ segment (ก่อนตัดสิน known/unknown)
        for s in analysis["slots"]:
            if s["status"] == "null" or not s.get("cls"):
                continue
            xs = (s["start"] + s["end"]) / 2 / fps
            ax.text(xs, -ymax * 0.08, f"{s['cls'][0]}\n{s['cls'][1]:.2f}", ha="center", va="top", fontsize=9, color="#e63946")
    ax.set_xlim(0, T / fps); ax.set_ylim(-ymax * 0.35, ymax)
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.set_title(title, fontsize=15)
    return ax


def draw_thai(img_bgr, text, xy=(10, 10), size=28, color=(255, 255, 255), bg=(0, 0, 0)):
    """วาดข้อความไทยบนเฟรม OpenCV (BGR) ด้วย PIL + font ไทย"""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img_bgr[:, :, ::-1]); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font_path(), size)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    d.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), fill=bg); d.text(xy, text, font=f, fill=color)
    return np.array(im)[:, :, ::-1].copy()


def render_overlay_video(video_path, analysis: dict, out_path, fps_target=cfg.TARGET_FPS, max_w=640):
    """เขียนวิดีโอใหม่ที่มีคำ/blank ของ segment ปัจจุบัน + ประโยคทั้งหมด (debug ว่า segment ตรงกับท่าจริงไหม)"""
    import cv2
    cap = cv2.VideoCapture(str(video_path)); src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = min(1.0, max_w / max(W, 1)); W2, H2 = int(W * sc), int(H * sc)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W2, H2))
    words = " ".join(s["word"] for s in analysis["slots"] if s["status"] != "null")
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t_ms = 1000.0 * fi / src_fps; fr = cv2.resize(fr, (W2, H2))
        cur = next((s for s in analysis["slots"] if s["t_start_ms"] <= t_ms <= s["t_end_ms"] + 1000 / fps_target), None)
        if cur is not None and cur["status"] != "null":
            col = (80, 200, 120) if cur["status"] in ("known", "learned") else (60, 90, 240)
            fr = draw_thai(fr, f"{cur['word']}  ({cur['status']})", (10, 10), 30, color=col)
        fr = draw_thai(fr, words, (10, H2 - 45), 24)
        out.write(fr); fi += 1
    cap.release(); out.release()
    return out_path
