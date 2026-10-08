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
# CARI palette: blue = HyRES / learned, ochre = JPEG, navy titles, slate secondary text, gray for other codecs.
BLUE, OCHRE, NAVY, SLATE = "#1677ff", "#b7791f", "#172033", "#475467"
MUTED, GRID, FRAME, GRAY = "#667085", "#eceef2", "#98a2b3", "#8b93a3"
SHOWN = ["kodim01", "kodim05", "kodim23"]
CROP_W, CROP_H = 192, 128

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10, "text.color": NAVY, "axes.labelcolor": NAVY,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#e4e7ec", "axes.linewidth": 0.8,
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
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    styles = {  # published codecs: recessive grays told apart by marker and dash, keyed in a quiet legend
        "ELIC (2022)": ("ELIC (CVPR22)", "#4b5565", "-", "s"),
        "Cheng 2020": ("Cheng (CVPR20)", "#7a8394", "-", "^"),
        "Balle 2018": ("Ball\u00e9 (ICLR18)", "#98a2b3", "--", "v"),
        "VTM (VVC)": ("VVC (VTM)", "#667085", ":", None),
    }
    for b in results["baselines"]:
        if b["name"] not in styles:
            print(f"warning: no style for baseline {b['name']}, skipped")
            continue
        text, color, ls, mk = styles[b["name"]]
        pts = np.array(b["points"])
        ax.plot(pts[:, 0], pts[:, 1], ls, color=color, lw=1.5, marker=mk, ms=4.6, mec="white", mew=0.7,
                zorder=2, label=text)

    jp = np.array(results["jpeg"]["points"])
    ax.plot(jp[:, 0], jp[:, 1], "-", color=OCHRE, lw=1.9, zorder=3)
    ax.plot(jp[:, 0], jp[:, 1], "o", color=OCHRE, ms=4.8, mec="white", mew=0.9, zorder=4)
    ax.text(2.07, float(np.interp(2.07, jp[:, 0], jp[:, 1])) - 1.1, "JPEG", color=OCHRE, fontsize=10.5,
            fontweight="bold", ha="right", va="top")

    est = np.array(sorted((p["bpp_est"], p["psnr"]) for p in results["reported_phases"]))
    ax.plot(est[:, 0], est[:, 1], "o", color="white", ms=7, mec=BLUE, mew=1.7, zorder=4)
    ax.text(est[0, 0] + 0.04, est[0, 1] - 1.0, "HyRES phases 6 to 1:\ntraining-time estimates", color=BLUE,
            fontsize=9, va="top", ha="left", linespacing=1.35)

    a = results["hyres_phase1"]["average"]
    j_at = float(np.interp(a["bpp"], jp[:, 0], jp[:, 1]))
    ax.annotate("", xy=(a["bpp"], a["psnr"] - 0.45), xytext=(a["bpp"], j_at + 0.2),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.8, shrinkA=0, shrinkB=0))
    ax.text(a["bpp"] + 0.05, (a["psnr"] + j_at) / 2 + 0.45, f"+{a['psnr'] - j_at:.1f} dB", color=BLUE,
            fontsize=10.5, fontweight="bold", va="center")
    ax.plot([a["bpp"]], [a["psnr"]], "o", color=BLUE, ms=15, mec="white", mew=2.2, alpha=0.35, zorder=5)
    ax.plot([a["bpp"]], [a["psnr"]], "o", color=BLUE, ms=8.5, mec="white", mew=1.6, zorder=6)
    ax.text(a["bpp"] + 0.07, a["psnr"] + 0.95, "HyRES\n(real bitstream)", color=NAVY, fontsize=10,
            fontweight="bold", ha="left", va="center", linespacing=1.3)

    ax.annotate("", xy=(1.74, 26.9), xytext=(1.92, 24.3),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
    ax.text(1.93, 24.2, "better", color=MUTED, fontsize=10, style="italic", va="center")

    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.80, 0.02), frameon=False, fontsize=9, labelcolor=SLATE,
                    title="Published codecs", handlelength=2.4, labelspacing=0.55, borderaxespad=0)
    leg.get_title().set_fontsize(9); leg.get_title().set_color(MUTED); leg._legend_box.align = "left"
    ax.set_xlim(0, 2.1); ax.set_ylim(22, 41.5)
    ax.set_xlabel("Bitrate (bpp)  \u2193", fontsize=11.5, labelpad=8)
    ax.set_ylabel("PSNR (dB)  \u2191", fontsize=11.5, labelpad=8)
    ax.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#e4e7ec")
    ax.tick_params(length=3, color="#d0d5dd")
    ax.set_title("Kodak, 24 images, RGB PSNR  \u00b7  JPEG uses the same TurboJPEG settings as the HyRES base layer",
                 fontsize=9.5, color=MUTED, loc="left", pad=10)
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

    fig, axes = plt.subplots(len(SHOWN), 4, figsize=(10.6, 2.55 * len(SHOWN)),
                             gridspec_kw={"width_ratios": [1.45, 1, 1, 1], "wspace": 0.045, "hspace": 0.5})
    heads = [("Full image", NAVY), ("Original", NAVY), ("JPEG, equal bitrate", OCHRE), ("HyRES (ours)", BLUE)]

    for r, stem in enumerate(SHOWN):
        orig = load(os.path.join(kodak, f"{stem}.png"))
        rec = load(os.path.join(eval_dir, f"{stem}_recon.png"))
        if orig.shape[0] > orig.shape[1]:
            sys.exit(f"error: {stem} is portrait; pick landscape images for this layout")
        q, jbpp, jdec = jpeg_at_least(orig, hy[stem])
        x, y = textured_crop(orig)
        c = lambda a: a[y:y + CROP_H, x:x + CROP_W].astype(np.uint8)

        def note(bpp, img, quality=None):
            m = ms_ssim(orig, img)
            ms = f" / {m:.3f}" if m is not None else ""
            first = f"q{quality} \u00b7 {bpp:.3f} bpp" if quality else f"{bpp:.3f} bpp"
            return f"{first}\n{psnr(orig, img):.2f} dB{ms}"

        ax = axes[r, 0]
        ax.imshow(orig.astype(np.uint8))
        ax.add_patch(mpatches.Rectangle((x, y), CROP_W, CROP_H, fill=False, ec=NAVY, lw=2.0))
        ax.text(-0.05, 0.5, stem, transform=ax.transAxes, ha="right", va="center", fontsize=11.5,
                fontweight="bold", color=NAVY)
        panels = [(c(orig), FRAME, "crop of the original, lossless"),
                  (c(jdec), OCHRE, note(jbpp, jdec, q)),
                  (c(rec), BLUE, note(hy[stem], rec))]
        for k, (img, col, caption) in enumerate(panels, start=1):
            a = axes[r, k]
            a.imshow(img, interpolation="nearest")
            a.set_xlabel(caption, fontsize=9.5, labelpad=6, color=SLATE, linespacing=1.45)
            for sp in a.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.6)
        axes[r, 0].set_xlabel("navy box = crop shown on the right", fontsize=9.5, labelpad=6, color=SLATE)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
        axes[r, 0].spines[:].set_edgecolor(FRAME); axes[r, 0].spines[:].set_linewidth(1.7)
        print(f"{stem}: HyRES {hy[stem]:.3f} bpp vs JPEG q{q} {jbpp:.3f} bpp, crop at {(x, y)}")
    fig.canvas.draw()  # axes resize to keep image aspect; read final positions, then align headers on one line
    top = max(axes[0, k].get_position().y1 for k in range(4))
    for k, (txt, col) in enumerate(heads):
        pos = axes[0, k].get_position()
        fig.text((pos.x0 + pos.x1) / 2, top + 0.012, txt, ha="center", va="bottom", fontsize=12, fontweight="bold", color=col)
    fig.text(0.5, 0.0, "Under each crop: bitrate, then PSNR / MS-SSIM of the whole image. JPEG gets at least as many bits "
             "as HyRES; crops are the most textured region of the original.", ha="center", va="top", fontsize=9.5, color=MUTED)
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
