import argparse
import sys
from pathlib import Path

import torch
from models import ResidualJPEGCompression, LightWeightCheckerboard
from .utils import load_checkpoint
import os


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath", type=str, required=True, help="Path to the checkpoint model to be exported."
    )
    parser.add_argument("-n", "--name", type=str, help="Exported model name.")
    parser.add_argument("-d", "--dir", type=str, help="Exported model directory.")
    parser.add_argument(
        "--no-update",
        action="store_true",
        default=False,
        help="Do not update the model CDFs parameters.",
    )
    parser.add_argument("--N", type=int, default=128, help="Number of channels")
    parser.add_argument("--M", type=int, default=192, help="Number of latent channels")
    parser.add_argument(
        "--jpeg-quality",
        default=1,
        type=int,
        help="JPEG quality factor (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force update the model CDFs parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to perform update on (default: cpu)",
    )
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    filepath = Path(args.filepath).resolve()
    if not filepath.is_file():
        raise RuntimeError(f'"{filepath}" is not a valid file.')

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load checkpoint properly
    checkpoint = load_checkpoint(filepath)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Create base model with specified parameters
    base_model = LightWeightCheckerboard(N=args.N, M=args.M)
    model = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=args.jpeg_quality
    )

    # Move model to device before loading state dict
    model = model.to(device)

    # Load state dict
    model.load_state_dict(state_dict)

    # Update entropy parameters if needed
    if not args.no_update:
        print("Updating the model's entropy parameters...")
        updated = model.update(force=args.force)
        if updated:
            print("Successfully updated the model's entropy parameters.")
        else:
            print("No update needed or update not successful.")

    # Get the updated state dict. Drop the untrained refinement weights when the
    # source checkpoint had none, so inference keeps refinement disabled.
    updated_state_dict = model.state_dict()
    if not model.use_refine:
        updated_state_dict = {k: v for k, v in updated_state_dict.items() if not k.startswith("refine.")}
        print("Exported without refinement weights (source checkpoint has none).")

    # Determine output filename
    if not args.name:
        filename = filepath.stem
        while Path(filename).suffixes:
            filename = Path(filename).stem
    else:
        filename = args.name

    ext = "".join(filepath.suffixes)

    # Handle output directory
    if args.dir is not None:
        output_dir = Path(args.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    # Save the updated model
    output_filepath = output_dir / f"{filename}{ext}"
    print(f"Saving updated model to: {output_filepath}")
    torch.save(updated_state_dict, output_filepath)
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])