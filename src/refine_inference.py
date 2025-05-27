import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.hyres import ResidualJPEGCompression
from models.checkerboard import LightWeightCheckerboard


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with optional post-processing")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--N", default=128, type=int, help="Number of channels")
    parser.add_argument("--M", default=192, type=int, help="Number of latent channels")
    parser.add_argument("--jpeg-quality", default=1, type=int, help="JPEG quality factor")
    parser.add_argument("--se-reduction", default=16, type=int, help="SE block reduction factor")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--postprocess-checkpoint", type=str, help="Path to post-processing checkpoint")
    parser.add_argument("--skip-postprocessing", action="store_true", help="Skip post-processing blocks")
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == 'true', default=True, help="Use CUDA")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create model
    base_model = LightWeightCheckerboard(N=args.N, M=args.M)
    model = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=args.jpeg_quality,
        se_reduction=args.se_reduction
    )

    # Load base model weights
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Load post-processing weights if provided
    if args.postprocess_checkpoint and not args.skip_postprocessing:
        pp_checkpoint = torch.load(args.postprocess_checkpoint, map_location=device)
        # Update model's state dict with just the post-processing weights
        model_state_dict = model.state_dict()
        model_state_dict.update(pp_checkpoint["state_dict"])
        model.load_state_dict(model_state_dict)

    model = model.to(device)
    model.eval()

    # Create a custom forward function if skipping post-processing
    if args.skip_postprocessing:
        original_forward = model.forward

        def forward_no_postprocess(x, noisequant=False):
            results = original_forward(x, noisequant)
            # Use x_hat before post-processing
            x_hat_initial = results['jpeg_decoded'] + results['residual_hat']
            results['x_hat'] = torch.clamp(x_hat_initial, 0, 1)
            return results

        model.forward = forward_no_postprocess

    # Load and preprocess input image
    input_image = Image.open(args.input).convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        results = model(input_tensor)

    # Save output image
    output_tensor = results['x_hat'].squeeze().cpu()
    output_image = transforms.ToPILImage()(output_tensor)
    output_image.save(args.output)

    print(f"Processed image saved to {args.output}")


if __name__ == "__main__":
    main()