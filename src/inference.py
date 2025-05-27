import argparse
import sys
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from models import ResidualJPEGCompression, LightWeightCheckerboard
from src.utils import load_checkpoint

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Inference script for ResidualJPEGCompression model.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint model"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory of images"
    )
    parser.add_argument(
        "--output", type=str, default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Number of channels (default: %(default)s)"
    )
    parser.add_argument(
        "--M", type=int, default=192, help="Number of latent channels (default: %(default)s)"
    )
    parser.add_argument(
        "--jpeg-quality",
        default=1,
        type=int,
        help="JPEG quality factor (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda", type=lambda x: str(x).lower() == 'true',
        default=True, help="Use cuda if available (default: %(default)s)"
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Save JPEG and residual components"
    )

    return parser.parse_args(argv)


def process_image(model, img_path, output_dir, device, save_components=False):
    """Process a single image through the model and save the output."""
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    transform = transforms.ToTensor()
    x = transform(img).unsqueeze(0)  # Add batch dimension

    # Get original dimensions for metrics calculation
    num_pixels = x.size(0) * x.size(2) * x.size(3)

    # Move to appropriate device
    x = x.to(device)

    # Processing
    out_enc = model.compress(x)
    enc_time = out_enc["time"]
    out_dec = model.decompress(out_enc)
    dec_time = out_dec["time"]

    # Create output filename
    img_name = os.path.basename(img_path)
    base_name, ext = os.path.splitext(img_name)

    # Save reconstructed image
    recon_path = os.path.join(output_dir, f"{base_name}_recon{ext}")
    save_image(out_dec["x_hat"], recon_path)

    # Save JPEG decoded image if requested
    with torch.no_grad():
        out_net = model(x)

    # Save component images if requested
    if save_components:
        # Save original image
        original_path = os.path.join(output_dir, f"{base_name}_original{ext}")
        save_image(x, original_path)

        jpeg_path = os.path.join(output_dir, f"{base_name}_jpeg{ext}")
        save_image(out_net["jpeg_decoded"], jpeg_path)

        # Save residual images
        residual_path = os.path.join(output_dir, f"{base_name}_residual{ext}")
        residual_vis = out_net["residual"] * 0.5 + 0.5  # Normalize for better visualization
        save_image(residual_vis, residual_path)

        residual_hat_path = os.path.join(output_dir, f"{base_name}_residual_hat{ext}")
        residual_hat_vis = out_net["residual_hat"] * 0.5 + 0.5
        save_image(residual_hat_vis, residual_hat_path)

    # Calculate y bpp
    y_bpp = 0
    for string_list in out_enc["strings"][0]:  # First element is y strings (anchor + non-anchor)
        for s in string_list:
            y_bpp += len(s) * 8
    y_bpp /= num_pixels

    # Calculate z bpp
    z_bpp = 0
    for s in out_enc["strings"][1]:
        z_bpp += len(s) * 8
    z_bpp /= num_pixels

    # Get JPEG bpp
    jpeg_bpp = out_net["jpeg_bpp_loss"].item()

    # Calculate total bpp
    total_bpp = jpeg_bpp + y_bpp + z_bpp

    # Quality metrics
    mse = torch.nn.functional.mse_loss(x, out_dec["x_hat"]).item()
    mse_db = mse * 255 ** 2
    psnr_val = -10 * math.log10(mse_db)

    # Calculate MS-SSIM if available
    try:
        from pytorch_msssim import ms_ssim
        msssim_val = ms_ssim(x, out_dec["x_hat"], data_range=1.0).item()
    except ImportError:
        msssim_val = 0

    print(f"Processed {img_path}")
    print(f"Total bpp: {total_bpp:.4f} (JPEG: {jpeg_bpp:.4f}, Y: {y_bpp:.5f}, Z: {z_bpp:.5f})")
    print(f"MSE: {mse_db:.4f})")
    print(f"PSNR: {psnr_val:.2f} dB, MS-SSIM: {msssim_val:.4f}")
    print(f"Encoding time: {enc_time:.4f}s, Decoding time: {dec_time:.4f}s")

    return {
        "filename": img_name,
        "total_bpp": total_bpp,
        "jpeg_bpp": jpeg_bpp,
        "y_bpp": y_bpp,
        "z_bpp": z_bpp,
        "mse": mse_db,
        "psnr": psnr_val,
        "ms_ssim": msssim_val,
        "enc_time": enc_time,
        "dec_time": dec_time,
    }


def main(argv):
    args = parse_args(argv)

    # Set up device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load model from checkpoint
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f'"{checkpoint_path}" is not a valid file.')

    print(f"Loading model from {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path)

    # Initialize model
    base_model = LightWeightCheckerboard(N=args.N, M=args.M)
    model = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=args.jpeg_quality
    )

    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Process input (single image or directory)
    input_path = Path(args.input).resolve()

    metrics = []

    if input_path.is_file():
        # Process single image
        result = process_image(model, str(input_path), args.output, device, args.save_components)
        metrics.append(result)
    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for img_path in input_path.glob('*'):
            if img_path.suffix.lower() in image_extensions:
                result = process_image(model, str(img_path), args.output, device, args.save_components)
                metrics.append(result)
    else:
        raise RuntimeError(f'"{input_path}" is neither a file nor a directory.')

    # Print average metrics
    if metrics:
        avg_metrics = {key: 0 for key in metrics[0].keys() if key != 'filename'}
        for m in metrics:
            for key in avg_metrics:
                avg_metrics[key] += m[key]

        for key in avg_metrics:
            avg_metrics[key] /= len(metrics)

        print("\nAverage metrics:")
        print(f"Total bpp: {avg_metrics['total_bpp']:.4f} (JPEG: {avg_metrics['jpeg_bpp']:.4f}, "
              f"Y: {avg_metrics['y_bpp']:.5f}, Z: {avg_metrics['z_bpp']:.5f})")
        print(f"MSE: {avg_metrics['mse']:.4f} dB")
        print(f"PSNR: {avg_metrics['psnr']:.2f} dB, MS-SSIM: {avg_metrics['ms_ssim']:.4f}")
        print(f"Encoding time: {avg_metrics['enc_time']:.4f}s, Decoding time: {avg_metrics['dec_time']:.4f}s")

    # Save metrics as CSV
    if metrics:
        import csv
        csv_path = os.path.join(args.output, "metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'total_bpp', 'jpeg_bpp', 'y_bpp', 'z_bpp',
                             'mse', 'psnr', 'ms_ssim', 'enc_time(s)', 'dec_time(s)'])
            for m in metrics:
                writer.writerow([
                    m['filename'],
                    m['total_bpp'],
                    m['jpeg_bpp'],
                    m['y_bpp'],
                    m['z_bpp'],
                    m['mse'],
                    m['psnr'],
                    m['ms_ssim'],
                    m['enc_time'],
                    m['dec_time']
                ])
        print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    import math  # For log calculations

    main(sys.argv[1:])