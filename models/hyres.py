import torch
from compressai.models import CompressionModel

from . import LightWeightELICWithCheckerboard
from .utils.gpu_jpeg_compression import NvJPEGCompression

class ResidualJPEGCompression(CompressionModel):
    """
    Combined JPEG and neural compression model that compresses residuals.
    Optimized for GPU-only data flow.
    """

    def __init__(self, base_model=None, jpeg_quality=25, **kwargs):
        super().__init__()
        # Use GPU JPEG if available
        self.jpeg = NvJPEGCompression(quality=jpeg_quality)
        self.residual_model = base_model if base_model is not None else LightWeightELICWithCheckerboard(**kwargs)

    def forward(self, x, noisequant=False):
        """
        Forward pass for combined JPEG + residual compression.
        Optimized for GPU-only data flow.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W] with values in [0, 1]
            noisequant (bool): Whether to use noise-based quantization

        Returns:
            dict: Results including reconstructed image and likelihood information
        """
        # JPEG operations now happen directly on the device
        jpeg_decoded = self.jpeg(x)

        # Calculate residual (all on device)
        residual = x - jpeg_decoded

        # Process residual with neural model
        residual_results = self.residual_model(residual, noisequant=noisequant)

        # Get reconstructed residual
        residual_hat = residual_results['x_hat']

        # Final reconstruction
        x_hat = jpeg_decoded + residual_hat

        # Clamp to valid range
        x_hat = torch.clamp(x_hat, 0, 1)

        return {
            'x_hat': x_hat,
            'likelihoods': residual_results['likelihoods'],
            'jpeg_decoded': jpeg_decoded,
            'residual': residual,
            'residual_hat': residual_hat
        }


# Create an example to check if the model works in only in gpu
if __name__ == "__main__":
    import torch
    from torchvision import transforms
    from PIL import Image

    # Load an example image
    img = Image.open("data/kodak/kodim01.png").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Create the model
    model = ResidualJPEGCompression().cuda()  # Move model to GPU

    # Forward pass
    output = model(img_tensor)

    print(output['x_hat'].shape)  # Should print the shape of the reconstructed image tensor