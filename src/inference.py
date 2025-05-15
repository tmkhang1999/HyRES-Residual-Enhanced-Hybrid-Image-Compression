"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import List

import compressai
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from compressai.zoo import load_state_dict
from pytorch_msssim import ms_ssim
from torchvision import transforms
from models import ResidualJPEGCompression, LightWeightCheckerboard

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, f, outputpath, patch):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/result.csv'
    print('decoding img: {}'.format(f))

    # Padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    _, _, height, width = x_padded.size()

    # Compression
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    # Decompression
    start = time.time()
    out_dec = model.decompress(out_enc)
    dec_time = time.time() - start

    # Remove padding
    out_dec["x_hat"] = torch.nn.functional.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    # Calculate bits per pixel
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = 0

    # Handle structured strings from checkerboard model
    for string_list in out_enc["strings"][0]:  # First element is y strings (anchor + non-anchor)
        for s in string_list:  # Each string_list has anchor strings and non-anchor strings
            bpp += len(s)

    # Add z strings
    for s in out_enc["strings"][1]:  # Second element is z strings
        bpp += len(s)

    bpp *= 8.0 / num_pixels

    # Calculate anchor/non-anchor bpp portions (if needed)
    z_bpp = sum(len(s) for s in out_enc["strings"][1]) * 8.0 / num_pixels
    y_bpp = bpp - z_bpp

    # Save reconstructed image
    torchvision.utils.save_image(out_dec["x_hat"], imgPath, nrow=1)

    # Calculate quality metrics
    mse = torch.nn.functional.mse_loss(x, out_dec["x_hat"]).item()
    mse_db = mse * 255 ** 2
    PSNR = psnr(x, out_dec["x_hat"])
    msssim = ms_ssim(x, out_dec["x_hat"], data_range=1.0).item()

    # Write results to CSV
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, y_bpp, z_bpp,
               mse_db, PSNR, msssim,
               enc_time, dec_time]

        # Add detailed timing - handle both dictionary and float time values
        if "time" in out_enc:
            if isinstance(out_enc["time"], dict):
                # ELIC style - dictionary with detailed timings
                row.extend([
                    out_enc["time"].get('y_enc', 0) * 1000,
                    out_dec["time"].get('y_dec', 0) * 1000 if "time" in out_dec else 0,
                    out_enc["time"].get('z_enc', 0) * 1000,
                    out_enc["time"].get('z_dec', 0) * 1000,
                    out_enc["time"].get('params', 0) * 1000
                ])
            else:
                # Checkerboard style - just total compression time
                total_time = out_enc["time"] * 1000
                row.extend([total_time / 2, total_time / 2, 0, 0, 0])
        else:
            row.extend([0, 0, 0, 0, 0])  # Add zeros if time not available

        write = csv.writer(f)
        write.writerow(row)

    print(f'bpp: {bpp:.4f}, MSE: {mse_db:.4f}, PSNR: {PSNR:.4f}, MS-SSIM: {msssim:.4f}, encoding time: {enc_time:.4f}s, decoding time: {dec_time:.4f}s')

    return {
        "psnr": PSNR,
        "mse": mse_db,
        "msssim": msssim,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, f, outputpath, patch):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/result.csv'
    print('decoding img: {}'.format(f))

    # Padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    # Inference with entropy estimation (no actual coding)
    start = time.time()
    out_net = model.inference(x_padded)
    elapsed_time = time.time() - start

    # Remove padding
    out_net["x_hat"] = torch.nn.functional.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    # Calculate bits per pixel
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )  # Missing closing parenthesis was here

    # Split bpp between y and z
    y_bpp = (torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))
    z_bpp = (torch.log(out_net["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels))

    # Save reconstructed image
    torchvision.utils.save_image(out_net["x_hat"], imgPath, nrow=1)

    # Calculate quality metrics
    PSNR = psnr(x, out_net["x_hat"])

    # Write results to CSV
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp.item() * num_pixels, num_pixels, bpp.item(), y_bpp.item(), z_bpp.item(),
               torch.nn.functional.mse_loss(x, out_net["x_hat"]).item() * 255 ** 2, PSNR,
               ms_ssim(x, out_net["x_hat"], data_range=1.0).item(),
               elapsed_time / 2.0, elapsed_time / 2.0]

        # Add detailed timing if available
        if "time" in out_net:
            row.extend([
                out_net["time"].get('y_enc', 0) * 1000,
                out_net["time"].get('y_dec', 0) * 1000,
                out_net["time"].get('z_enc', 0) * 1000,
                out_net["time"].get('z_dec', 0) * 1000,
                out_net["time"].get('params', 0) * 1000
            ])
        else:
            row.extend([0, 0, 0, 0, 0])  # Add zeros if detailed timing not available

        write = csv.writer(f)
        write.writerow(row)

    return {
        "psnr": PSNR,
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, entropy_estimation=False, half=False, outputpath='Recon', patch=256):
    device = next(model.parameters()).device
    metrics = defaultdict(float)

    # Create output directory
    imgdir = filepaths[0].split('/')
    imgdir[-2] = outputpath
    imgDir = '/'.join(imgdir[:-1])
    if not os.path.isdir(imgDir):
        os.makedirs(imgDir)

    # Create/reset CSV file
    csvfile = imgDir + '/result.csv'
    if os.path.isfile(csvfile):
        os.remove(csvfile)
    with open(csvfile, 'w') as f:
        row = ['name', 'bits', 'pixels', 'bpp', 'y_bpp', 'z_bpp', 'mse', 'psnr(dB)', 'ms-ssim', 'enc_time(s)',
               'dec_time(s)', 'y_enc(ms)', 'y_dec(ms)', 'z_enc(ms)', 'z_dec(ms)', 'param(ms)']
        write = csv.writer(f)
        write.writerow(row)

    # Process each image
    for f in filepaths:
        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, f, outputpath, patch)
        else:
            rv = inference_entropy_estimation(model, x, f, outputpath, patch)
        for k, v in rv.items():
            metrics[k] += v

    # Calculate average metrics
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument(
        "--output_path",
        help="result output path",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        required=True,
        help="checkpoint path",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=256,
        help="padding patch size (default: %(default)s)",
    )
    parser.add_argument("--N", type=int, default=128, help="Number of channels")
    parser.add_argument("--M", type=int, default=192, help="Number of latent channels")
    parser.add_argument(
        "--jpeg-quality",
        default=1,
        type=int,
        help="JPEG quality factor (default: %(default)s)",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    filepaths = sorted(filepaths)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    state_dict = load_state_dict(torch.load(args.paths))
    base_model = LightWeightCheckerboard(N=args.N, M=args.M)
    model_cls = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=args.jpeg_quality,
    )
    model = model_cls.from_state_dict(state_dict, jpeg_quality=args.jpeg_quality).eval()

    results = defaultdict(list)

    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")

    metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.output_path, args.patch)
    for k, v in metrics.items():
        results[k].append(v)

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])