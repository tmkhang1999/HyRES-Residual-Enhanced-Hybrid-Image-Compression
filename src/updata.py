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
        "--filepath", type=str, help="Path to the checkpoint model to be exported."
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
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    filepath = Path(args.filepath).resolve()
    if not filepath.is_file():
        raise RuntimeError(f'"{filepath}" is not a valid file.')

    state_dict = load_checkpoint(filepath)

    base_model = LightWeightCheckerboard(N=args.N, M=args.M)
    model_cls = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=args.jpeg_quality
    )
    net = model_cls.from_state_dict(state_dict)

    if not args.no_update:
        net.update(force=True)
        print("Updated the model CDFs parameters.")
    state_dict = net.state_dict()

    if not args.name:
        filename = filepath
        while filename.suffixes:
            filename = Path(filename.stem)
    else:
        filename = args.name

    ext = "".join(filepath.suffixes)

    if args.dir is not None:
        output_dir = Path(args.dir)
        if not os.path.exists(args.dir):
            try:
                os.mkdir(args.dir)
            except:
                os.makedirs(args.dir)
        Path(output_dir).mkdir(exist_ok=True)
    else:
        output_dir = Path.cwd()

    filepath = output_dir / f"{filename}{ext}"
    torch.save(state_dict, filepath)


if __name__ == "__main__":
    main(sys.argv[1:])
