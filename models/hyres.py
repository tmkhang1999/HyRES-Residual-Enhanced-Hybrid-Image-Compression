import torch
import torch.nn as nn
from compressai.models import CompressionModel
from .utils.jpeg_compression import JPEGCompression
from .elic import LightWeightELIC


class ResidualJPEGCompression(CompressionModel):
    """
    Combined JPEG and neural compression model that compresses residuals.
    """

    def __init__(self, base_model=None, jpeg_quality=25, **kwargs):
        """
        Initialize the residual compression model.

        Args:
            base_model (nn.Module): Base neural compression model (TestModel)
            jpeg_quality (int): JPEG quality factor (0-100)
        """
        super().__init__()

        # JPEG compression module
        self.jpeg = JPEGCompression(quality=jpeg_quality)

        # Neural compression model for residuals
        self.residual_model = base_model if base_model is not None else LightWeightELIC(**kwargs)

    def forward(self, x, noisequant=False):
        """
        Forward pass for combined JPEG + residual compression.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W] with values in [0, 1]
            noisequant (bool): Whether to use noise-based quantization

        Returns:
            dict: Results including reconstructed image and likelihood information
        """
        # JPEG compression and decompression
        jpeg_decoded = self.jpeg(x)

        # Calculate residual
        residual = x - jpeg_decoded

        # Compress residual with neural model
        residual_results = self.residual_model(residual, noisequant=noisequant)

        # Get reconstructed residual
        residual_hat = residual_results['x_hat']

        # Final reconstruction: JPEG decoded + residual reconstruction
        x_hat = jpeg_decoded + residual_hat

        # Clamp to valid range
        x_hat = torch.clamp(x_hat, 0, 1)

        # Return results including likelihoods from residual model
        return {
            'x_hat': x_hat,
            'likelihoods': residual_results['likelihoods'],
            'jpeg_decoded': jpeg_decoded,
            'residual': residual,
            'residual_hat': residual_hat
        }

    def compress(self, x):
        """
        Compress the input image using both JPEG and neural compression.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W]

        Returns:
            dict: Compressed representation including JPEG buffers and neural model output
        """
        # JPEG compression stage
        jpeg_buffers = self.jpeg.compress(x)

        # JPEG decompression for residual calculation
        jpeg_decoded = self.jpeg.decompress(jpeg_buffers, x.device)

        # Calculate residual
        residual = x - jpeg_decoded

        # Compress residual with neural model
        # Note: Assuming residual_model has a compress method similar to CompressAI models
        residual_compressed = self.residual_model.compress(residual)
        residual_compressed['jpeg_buffers'] = jpeg_buffers
        return residual_compressed

    def decompress(self, compressed_representation):
        """
        Decompress the input representation.

        Args:
            compressed_representation (dict): Compressed representation from compress()

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        # Extract components
        jpeg_buffers = compressed_representation['jpeg_buffers']
        strings = compressed_representation['strings']
        shape = compressed_representation['shape']

        # JPEG decompression
        device = next(self.parameters()).device
        jpeg_decoded = self.jpeg.decompress(jpeg_buffers, device)

        # Residual decompression
        decompress_result = self.residual_model.decompress(strings, shape)

        # Final reconstruction
        x_hat = jpeg_decoded + decompress_result["x_hat"]

        # Clamp to valid range
        x_hat = torch.clamp(x_hat, 0, 1)
        decompress_result['x_hat'] = x_hat
        return decompress_result

    def load_state_dict(self, state_dict, **kwargs):
        self.residual_model.load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == "__main__":
    # Create a base TestModel
    base_model = LightWeightELIC(N=192, M=320, num_slices=5)

    # Create the residual compression model
    model = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=50
    )

    # x = torch.rand(10, 3, 512, 512)
    # out = model(x)
    print(model.state_dict())
