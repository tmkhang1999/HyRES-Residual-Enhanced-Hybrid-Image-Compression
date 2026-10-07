"""Build the images and data behind the project page (docs/).

Inputs (all produced by this repo):
  --eval-dir   output of scripts/evaluate.sh (kodimXX_{original,jpeg,recon,residual,residual_hat}.png + metrics.csv)
  --phase-dir  saved per-phase reconstructions (Original/original_i.png, JPEG/jpeg_i.png, Phase_k/recon_i.png)
  --kodak      Kodak PNGs (data/test)
  tools/figures/data/*.json  baseline curves (CompressAI) and our JPEG curve (jpeg_curve.py)

Outputs:
  docs/static/img/...       web images (WebP)
  docs/static/data/results.json

Usage (from the repo root):
    bash scripts/evaluate.sh checkpoint/inference/phase1_lambda0.045.pth.tar data/test output/phase1
    python tools/figures/jpeg_curve.py
    python tools/figures/make_figures.py --eval-dir output/phase1 --phase-dir checkpoint/1B
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

DATA = os.path.join("tools", "figures", "data")
OUT_IMG = os.path.join("docs", "static", "img")
OUT_DATA = os.path.join("docs", "static", "data")

# Images shown on the page (the six whose per-phase reconstructions were saved).
SHOWCASE = ["kodim23", "kodim05", "kodim20", "kodim01", "kodim19", "kodim18"]
# Index of each showcase image inside the saved phase folders (matched by pixel equality).
PHASE_INDEX = {"kodim05": 0, "kodim18": 1, "kodim20": 2, "kodim23": 3, "kodim01": 4, "kodim19": 5}
CROP = 160  # px, square crop for zoomed comparisons

# Published codecs from CompressAI's Kodak results (BSD 3-Clause Clear, see data/compressai_kodak/LICENSE).
BASELINES = [
    ("paper-elic2022_mse", "ELIC (2022)"),
    ("compressai-cheng2020-attn_mse_cuda", "Cheng 2020"),
    ("compressai-bmshj2018-hyperprior_mse_cuda", "Balle 2018"),
    ("vtm", "VTM (VVC)"),
]

# Numbers reported during the project (final presentation); not re-measurable
# because only the Phase 1 weights survive. bpp is the old training-time estimate.
REPORTED_PHASES = [
    {"phase": 1, "lmbda": 0.045, "best_epoch": 400, "bpp_est": 1.465, "mse": 11.76},
    {"phase": 2, "lmbda": 0.032, "best_epoch": 41, "bpp_est": 1.109, "mse": 15.72},
    {"phase": 3, "lmbda": 0.016, "best_epoch": 6, "bpp_est": 0.852, "mse": 22.89},
    {"phase": 4, "lmbda": 0.008, "best_epoch": 51, "bpp_est": 0.604, "mse": 41.28},
    {"phase": 5, "lmbda": 0.004, "best_epoch": 13, "bpp_est": 0.460, "mse": 63.79},
    {"phase": 6, "lmbda": 0.002, "best_epoch": 17, "bpp_est": 0.380, "mse": 91.61},
]
REPORTED_TIMING = [  # seconds per Kodak image, all on the same A40 VM, measured during the project
    {"model": "Balle 2018 (hyperprior)", "params_m": None, "enc": 0.22, "dec": 0.24},
    {"model": "Minnen 2018 (joint AR)", "params_m": 14.13, "enc": 2.85, "dec": 3.74},
    {"model": "Cheng 2020", "params_m": 13.18, "enc": 3.57, "dec": 6.56},
    {"model": "ELIC 2022", "params_m": 33.79, "enc": 4.31, "dec": 4.54},
    {"model": "HyRES (ours)", "params_m": 10.14, "enc": 0.476, "dec": 0.286},
]


def load_rgb(path):
    if not os.path.isfile(path):
        sys.exit(f"error: missing input image {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * math.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")


def save_webp(arr, path, lossless=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))
    if lossless:
        img.save(path, "WEBP", lossless=True, quality=100, method=6)
    else:
        img.save(path, "WEBP", quality=95, method=6)


def pick_crop(orig, jpeg, size=CROP, stride=16):
    """Window where JPEG error is largest on a textured region (most to see)."""
    err = np.mean((orig - jpeg) ** 2, axis=2)
    h, w = err.shape
    integral = np.pad(err, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, best_xy = -1.0, (0, 0)
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            s = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
            if s > best:
                best, best_xy = s, (x, y)
    return best_xy


def read_curve(name):
    path = os.path.join(DATA, "compressai_kodak", name + ".json")
    if not os.path.isfile(path):
        print(f"warning: baseline file not found, skipped: {path}")
        return None
    r = json.load(open(path))["results"]
    pts = sorted(zip(r["bpp"], r["psnr-rgb"]))
    return [[round(b, 4), round(p, 3)] for b, p in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="output")
    ap.add_argument("--phase-dir", default="checkpoint/1B")
    ap.add_argument("--kodak", default="data/test")
    args = ap.parse_args()

    # --- per-image verified metrics (real bitstream) ---
    csv_path = os.path.join(args.eval_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found; run scripts/evaluate.sh first")
    per_image = []
    for row in csv.DictReader(open(csv_path)):
        stem = os.path.splitext(row["filename"])[0]
        orig = load_rgb(os.path.join(args.kodak, row["filename"]))
        jpeg = load_rgb(os.path.join(args.eval_dir, f"{stem}_jpeg.png"))
        per_image.append({
            "id": stem,
            "bpp": round(float(row["total_bpp"]), 4),
            "jpeg_bpp": round(float(row["jpeg_bpp"]), 4),
            "psnr": round(float(row["psnr"]), 3),
            "jpeg_psnr": round(psnr(orig, jpeg), 3),
            "ms_ssim": round(float(row["ms_ssim"]), 4),
        })
    per_image.sort(key=lambda r: r["id"])
    n = len(per_image)
    avg = {k: round(sum(r[k] for r in per_image) / n, 4) for k in ("bpp", "jpeg_bpp", "psnr", "jpeg_psnr", "ms_ssim")}
    print(f"{n} images, average: {avg}")

    # --- showcase images: full-size comparisons + residuals ---
    for stem in SHOWCASE:
        orig = load_rgb(os.path.join(args.kodak, f"{stem}.png"))
        jpeg = load_rgb(os.path.join(args.eval_dir, f"{stem}_jpeg.png"))
        hyres = load_rgb(os.path.join(args.eval_dir, f"{stem}_recon.png"))
        save_webp(orig, os.path.join(OUT_IMG, "compare", f"{stem}_original.webp"))
        save_webp(jpeg, os.path.join(OUT_IMG, "compare", f"{stem}_jpeg.webp"))
        save_webp(hyres, os.path.join(OUT_IMG, "compare", f"{stem}_hyres.webp"))
        # Residual true (x - JPEG) and decoded, amplified 2x around mid-gray.
        save_webp(127.5 + 2 * (orig - jpeg), os.path.join(OUT_IMG, "residual", f"{stem}_residual.webp"))
        save_webp(127.5 + 2 * (hyres - jpeg), os.path.join(OUT_IMG, "residual", f"{stem}_residual_hat.webp"))

    # --- phase strip: zoomed crops per phase, distortion measured on the saved images ---
    phases_measured = {}
    for stem in SHOWCASE:
        i = PHASE_INDEX[stem]
        orig = load_rgb(os.path.join(args.phase_dir, "Original", f"original_{i}.png"))
        jpeg = load_rgb(os.path.join(args.phase_dir, "JPEG", f"jpeg_{i}.png"))
        kod = load_rgb(os.path.join(args.kodak, f"{stem}.png"))
        if orig.shape != kod.shape or np.any(orig != kod):
            sys.exit(f"error: {args.phase_dir}/Original/original_{i}.png is not {stem}")
        x, y = pick_crop(orig, jpeg)
        crop = lambda a: a[y:y + CROP, x:x + CROP]
        save_webp(crop(orig), os.path.join(OUT_IMG, "phases", f"{stem}_original.webp"), lossless=True)
        save_webp(crop(jpeg), os.path.join(OUT_IMG, "phases", f"{stem}_jpeg.webp"), lossless=True)
        entry = {"crop": [x, y, CROP], "jpeg_psnr": round(psnr(orig, jpeg), 3), "phases": []}
        for k in range(1, 7):
            rec = load_rgb(os.path.join(args.phase_dir, f"Phase_{k}", f"recon_{i}.png"))
            save_webp(crop(rec), os.path.join(OUT_IMG, "phases", f"{stem}_p{k}.webp"), lossless=True)
            entry["phases"].append(round(psnr(orig, rec), 3))
        phases_measured[stem] = entry
        print(stem, "crop", (x, y), "phase PSNR", entry["phases"])

    # --- rate-distortion curves ---
    curves = [{"name": label, "points": pts} for key, label in BASELINES if (pts := read_curve(key))]
    jpeg_ours = json.load(open(os.path.join(DATA, "jpeg_turbo_kodak.json")))
    jr = jpeg_ours["results"]
    jpeg_curve = {"name": "JPEG (TurboJPEG)", "quality": jpeg_ours["quality"],
                  "points": [[round(b, 4), round(p, 3)] for b, p in zip(jr["bpp"], jr["psnr-rgb"])]}

    reported = [dict(r, psnr=round(10 * math.log10(255.0 ** 2 / r["mse"]), 3)) for r in REPORTED_PHASES]

    results = {
        "dataset": "Kodak (24 images, 768x512)",
        "hyres_phase1": {"lmbda": 0.045, "epoch": 400, "params_m": 10.14, "average": avg,
                         "y_bpp": None, "per_image": per_image},
        "reported_phases": reported,
        "phases_measured": phases_measured,
        "jpeg": jpeg_curve,
        "baselines": curves,
        "timing": REPORTED_TIMING,
        "showcase": SHOWCASE,
    }
    os.makedirs(OUT_DATA, exist_ok=True)
    out = os.path.join(OUT_DATA, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
