"""Paper-style result figures, following learned-compression conventions.

  docs/static/fig/rd_kodak.{svg,pdf,png}            PSNR vs bpp on Kodak, HyRES highlighted
  docs/static/fig/visual_comparison.{png,pdf}       crops: original / JPEG at equal bitrate / HyRES

Fairness rules for the visual comparison:
  - JPEG uses the same TurboJPEG settings as the HyRES base layer, at the lowest
    quality whose bpp is >= the HyRES bpp on that image (JPEG never gets fewer bits).
  - The crop is the most textured window of the ORIGINAL image (gradient energy),
    chosen without looking at either reconstruction.

Usage (from the repo root, after make_figures.py has written docs/static/data/results.json):
    python tools/figures/make_plots.py --eval-dir output/phase1
"""
import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.getcwd())
from models.utils.turbo_jpeg_compression import TurboJPEGCompression  # noqa: E402

OUT = os.path.join("docs", "static", "fig")
BLUE, ORANGE, INK, GRID = "#2a78d6", "#d9622b", "#333333", "#e6e6e6"
SHOWN = ["kodim01", "kodim05", "kodim23"]
CROP_W, CROP_H = 192, 128

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "svg.fonttype": "path", "pdf.fonttype": 42,
})


def save(fig, stem, exts):
    os.makedirs(OUT, exist_ok=True)
    for ext in exts:
        path = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------- RD curve
def rd_plot(results):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {  # published codecs: recessive grays, told apart by dash pattern and marker
        "ELIC (2022)": ("ELIC (CVPR22)", "#4d4d4d", "-", "s"),
        "Cheng 2020": ("Cheng (CVPR20)", "#7a7a7a", "-", "^"),
        "Balle 2018": ("Ball\u00e9 (ICLR18)", "#9a9a9a", "--", "v"),
        "VTM (VVC)": ("VVC (VTM)", "#5f5f5f", ":", None),
    }
    for b in results["baselines"]:
        if b["name"] not in styles:
            print(f"warning: no style for baseline {b['name']}, skipped")
            continue
        label, color, ls, mk = styles[b["name"]]
        pts = np.array(b["points"])
        ax.plot(pts[:, 0], pts[:, 1], ls, color=color, lw=1.3, marker=mk, ms=3.5, label=label)

    jp = np.array(results["jpeg"]["points"])
    ax.plot(jp[:, 0], jp[:, 1], "-", color=ORANGE, lw=1.6, marker="o", ms=3.5, label="JPEG (TurboJPEG)")

    est = sorted((p["bpp_est"], p["psnr"]) for p in results["reported_phases"])
    est = np.array(est)
    ax.plot(est[:, 0], est[:, 1], ":", color=BLUE, lw=1.2, marker="o", ms=5, mfc="white", mec=BLUE, mew=1.3,
            label="HyRES phases (training-time estimate)")

    a = results["hyres_phase1"]["average"]
    ax.plot([a["bpp"]], [a["psnr"]], "*", color=BLUE, ms=15, mec="white", mew=0.8, zorder=5,
            label="HyRES (real bitstream)")
    j_at = float(np.interp(a["bpp"], jp[:, 0], jp[:, 1]))
    ax.annotate("", xy=(a["bpp"], a["psnr"] - 0.35), xytext=(a["bpp"], j_at + 0.1),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    ax.text(a["bpp"] + 0.04, (a["psnr"] + j_at) / 2 + 0.35, f"+{a['psnr'] - j_at:.1f} dB", color=BLUE,
            fontsize=9, va="center", fontweight="bold")

    ax.set_xlim(0, 2.0); ax.set_ylim(22, 41)
    ax.set_xlabel("Bitrate (bpp)"); ax.set_ylabel("PSNR (dB)")
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_title("Kodak (24 images, RGB PSNR)", fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=1, edgecolor="#cccccc")
    save(fig, "rd_kodak", ["svg", "pdf", "png"])


# ---------------------------------------------------------------- visual comparison
def load(path):
    if not os.path.isfile(path):
        sys.exit(f"error: missing {path}; run scripts/evaluate.sh first (see module docstring)")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    return 10 * math.log10(255.0 ** 2 / np.mean((a - b) ** 2))


def ms_ssim(a, b):
    try:
        from pytorch_msssim import ms_ssim as f
    except ImportError:
        return None
    t = lambda x: torch.from_numpy(x / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    return f(t(a), t(b), data_range=1.0).item()


def jpeg_at_least(orig, target_bpp):
    """Lowest TurboJPEG quality with bpp >= target. Returns (quality, bpp, decoded array)."""
    x = torch.from_numpy(orig / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    npx = orig.shape[0] * orig.shape[1]
    for q in range(1, 101):
        codec = TurboJPEGCompression(quality=q)
        bufs = codec.compress(x)
        bpp = sum(len(b.getvalue()) for b in bufs) * 8 / npx
        if bpp >= target_bpp or q == 100:
            dec = codec.decompress(bufs, torch.device("cpu"))[0].permute(1, 2, 0).numpy() * 255.0
            return q, bpp, np.clip(np.round(dec), 0, 255)


def textured_crop(orig):
    """Window with the highest gradient energy in the original image."""
    g = np.mean(orig, axis=2)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    integ = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, xy = -1, (0, 0)
    for y in range(0, e.shape[0] - CROP_H, 16):
        for x in range(0, e.shape[1] - CROP_W, 16):
            s = integ[y + CROP_H, x + CROP_W] - integ[y, x + CROP_W] - integ[y + CROP_H, x] + integ[y, x]
            if s > best:
                best, xy = s, (x, y)
    return xy


def visual(eval_dir, kodak):
    hy = {}
    with open(os.path.join(eval_dir, "metrics.csv")) as f:
        for row in csv.DictReader(f):
            hy[os.path.splitext(row["filename"])[0]] = float(row["total_bpp"])

    fig, axes = plt.subplots(len(SHOWN), 4, figsize=(10, 2.25 * len(SHOWN)),
                             gridspec_kw={"width_ratios": [1.5, 1, 1, 1], "wspace": 0.04, "hspace": 0.32})
    for r, stem in enumerate(SHOWN):
        orig = load(os.path.join(kodak, f"{stem}.png"))
        rec = load(os.path.join(eval_dir, f"{stem}_recon.png"))
        if orig.shape[0] > orig.shape[1]:
            sys.exit(f"error: {stem} is portrait; pick landscape images for this layout")
        q, jbpp, jdec = jpeg_at_least(orig, hy[stem])
        x, y = textured_crop(orig)
        c = lambda a: a[y:y + CROP_H, x:x + CROP_W].astype(np.uint8)

        def label(name, bpp, img):
            m = ms_ssim(orig, img)
            ms = f" / {m:.3f}" if m is not None else ""
            return f"{name}\n[{bpp:.3f} bpp / {psnr(orig, img):.2f} dB{ms}]"

        ax = axes[r, 0]
        ax.imshow(orig.astype(np.uint8))
        ax.add_patch(mpatches.Rectangle((x, y), CROP_W, CROP_H, fill=False, ec=BLUE, lw=1.6))
        ax.set_title(f"{stem}", fontsize=10.5, pad=4)
        panels = [("Original", c(orig), None),
                  (label(f"JPEG q{q}", jbpp, jdec), c(jdec), ORANGE),
                  (label("HyRES (ours)", hy[stem], rec), c(rec), BLUE)]
        for k, (title, img, col) in enumerate(panels, start=1):
            a = axes[r, k]
            a.imshow(img, interpolation="nearest")
            a.set_xlabel(title, fontsize=10, labelpad=4, fontweight="bold" if col == BLUE else "normal")
            for s in a.spines.values():
                s.set_edgecolor(col or INK); s.set_linewidth(1.6 if col else 0.8)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
        axes[r, 0].set_xlabel("Full image (crop in blue box)", fontsize=10, labelpad=4)
        print(f"{stem}: HyRES {hy[stem]:.3f} bpp vs JPEG q{q} {jbpp:.3f} bpp, crop at {(x, y)}")
    fig.text(0.5, 0.005, "Labels: [bitrate / PSNR / MS-SSIM], whole image. JPEG gets at least as many bits as HyRES.",
             ha="center", fontsize=9.5, color="#555555")
    save(fig, "visual_comparison", ["png", "pdf"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="output/phase1")
    ap.add_argument("--kodak", default="data/test")
    ap.add_argument("--results", default=os.path.join("docs", "static", "data", "results.json"))
    args = ap.parse_args()
    if not os.path.isfile(args.results):
        sys.exit(f"error: {args.results} not found; run tools/figures/make_figures.py first")
    rd_plot(json.load(open(args.results)))
    visual(args.eval_dir, args.kodak)


if __name__ == "__main__":
    main()
