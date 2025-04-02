import io
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class JPEGCompression(nn.Module):
    """
    JPEG compression module that can be integrated with neural compression pipelines.
    Handles compression, decompression, and tensor/numpy conversions efficiently.
    """

    def __init__(self, quality=25):
        """
        Initialize JPEG compression module with specified quality.

        Args:
            quality (int): JPEG quality factor (0-100)
        """
        super().__init__()
        self.quality = quality
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def compress(self, x):
        """
        Compress tensor images using JPEG and return compressed data.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W] with values in [0, 1]

        Returns:
            list: List of compressed image buffers
        """
        batch_size = x.size(0)
        compressed_buffers = []

        for i in range(batch_size):
            # Convert to PIL image
            pil_img = self.to_pil(torch.clamp(x[i], 0, 1))

            # Compress with JPEG
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)

            compressed_buffers.append(buffer)

        return compressed_buffers

    def decompress(self, compressed_buffers, device):
        """
        Decompress JPEG data to tensor images.

        Args:
            compressed_buffers (list): List of compressed image buffers
            device (torch.device): Device to place tensors on

        Returns:
            torch.Tensor: Decompressed image tensor [B, 3, H, W]
        """
        batch_size = len(compressed_buffers)
        decompressed_images = []

        for i in range(batch_size):
            # Reset buffer position
            compressed_buffers[i].seek(0)

            # Decompress JPEG
            pil_img = Image.open(compressed_buffers[i])
            tensor_img = self.to_tensor(pil_img).to(device)

            decompressed_images.append(tensor_img)

        return torch.stack(decompressed_images, dim=0)

    def forward(self, x):
        """
        Forward pass through JPEG compression and decompression.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W] with values in [0, 1]

        Returns:
            torch.Tensor: Reconstructed image tensor [B, 3, H, W]
        """
        compressed = self.compress(x)
        decompressed = self.decompress(compressed, x.device)
        return decompressed
