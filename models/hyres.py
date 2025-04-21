import torch
from compressai.models import CompressionModel

from .checkerboard import LightWeightCheckerboard
from .utils.turbo_jpeg_compression import TurboJPEGCompression


class ResidualJPEGCompression(CompressionModel):
    """
    Combined JPEG and neural compression model that compresses residuals.
    Optimized for CPU→CPU→GPU data flow.
    """

    def __init__(self, base_model=None, jpeg_quality=25, **kwargs):
        super().__init__()
        self.jpeg = TurboJPEGCompression(quality=jpeg_quality)
        self.residual_model = base_model if base_model is not None else LightWeightCheckerboard(**kwargs)

    def forward(self, x, noisequant=False):
        """
        Forward pass for combined JPEG + residual compression.
        Optimized for CPU→CPU→GPU data flow.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W] with values in [0, 1]
            noisequant (bool): Whether to use noise-based quantization

        Returns:
            dict: Results including reconstructed image and likelihood information
        """
        # Record the original device for later use
        device = next(self.parameters()).device

        # JPEG operations should happen on CPU
        if x.device == 'cuda':
            x_cpu = x.cpu()
        else:
            x_cpu = x

        # Perform JPEG compression/decompression on CPU
        jpeg_decoded_cpu = self.jpeg(x_cpu)

        # Calculate residual on CPU
        residual_cpu = x_cpu - jpeg_decoded_cpu

        # Now move the processed data to the original device (GPU if applicable)
        # for neural network processing
        jpeg_decoded = jpeg_decoded_cpu.to(device)
        residual = residual_cpu.to(device)

        # Process residual with neural model (on GPU if available)
        residual_results = self.residual_model(residual, noisequant=noisequant)

        # Get reconstructed residual
        residual_hat = residual_results['x_hat']

        # Final reconstruction on original device
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
        residual_model_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith('residual_model.'):
                # Remove the 'residual_model.' prefix
                new_key = key[len('residual_model.'):]
                residual_model_state_dict[new_key] = value

        self.residual_model.load_state_dict(residual_model_state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == "__main__":
    # Create a base TestModel
    base_model = LightWeightCheckerboard(N=192, M=320)

    # Create the residual compression model
    model = ResidualJPEGCompression(
        base_model=base_model,
        jpeg_quality=50
    )

    # x = torch.rand(10, 3, 512, 512)
    # out = model(x)
    print(model.state_dict())
