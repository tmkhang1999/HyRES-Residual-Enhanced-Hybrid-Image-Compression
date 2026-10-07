"""Measure the JPEG rate-distortion curve on Kodak with the exact TurboJPEG
settings HyRES uses for its base layer, so the JPEG baseline on the project
page is like-for-like with HyRES.

Usage (from the repo root):
    python tools/figures/jpeg_curve.py [--kodak data/test] [--out tools/figures/data/jpeg_turbo_kodak.json]
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.getcwd())
from models.utils.turbo_jpeg_compression import TurboJPEGCompression  # noqa: E402

QUALITIES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak", default="data/test")
    ap.add_argument("--out", default="tools/figures/data/jpeg_turbo_kodak.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.kodak, "*.png")))
    if not files:
        sys.exit(f"error: no PNG images found in {args.kodak}")
    images = [transforms.ToTensor()(Image.open(f).convert("RGB")).unsqueeze(0) for f in files]

    bpp, psnr = [], []
    for q in QUALITIES:
        codec = TurboJPEGCompression(quality=q)
        b_sum = p_sum = 0.0
        for x in images:
            bufs = codec.compress(x)
            x_hat = codec.decompress(bufs, torch.device("cpu"))
            mse = torch.mean((x - x_hat) ** 2).item()
            b_sum += sum(len(b.getvalue()) for b in bufs) * 8 / (x.shape[2] * x.shape[3])
            p_sum += 10 * math.log10(1.0 / mse)
        bpp.append(b_sum / len(images))
        psnr.append(p_sum / len(images))
        print(f"q={q:3d}  bpp={bpp[-1]:.4f}  psnr={psnr[-1]:.2f} dB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"name": "JPEG (TurboJPEG, HyRES settings)", "quality": QUALITIES,
                   "results": {"bpp": bpp, "psnr-rgb": psnr}}, f, indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
