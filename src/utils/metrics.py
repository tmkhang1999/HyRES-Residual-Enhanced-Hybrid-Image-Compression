import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import lpips


class CompressionMetrics:
    """Common metrics for image compression evaluation."""

    def __init__(self, device='cuda'):
        """Initialize metrics calculator."""
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)

    def psnr(self, x, y):
        """
        Calculate Peak Signal-to-Noise Ratio.
        Args:
            x: Original image tensor [B,3,H,W] in range [0,1]
            y: Compressed image tensor [B,3,H,W] in range [0,1]
        """
        mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
        psnr = -10 * torch.log10(mse)
        return psnr.mean().item()

    def msssim(self, x, y):
        """
        Calculate MS-SSIM (Multi-Scale Structural Similarity).
        Args:
            x: Original image tensor [B,3,H,W] in range [0,1]
            y: Compressed image tensor [B,3,H,W] in range [0,1]
        """
        return ms_ssim(x, y, data_range=1.0).item()

    def lpips(self, x, y):
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity).
        Args:
            x: Original image tensor [B,3,H,W] in range [0,1]
            y: Compressed image tensor [B,3,H,W] in range [0,1]
        """
        return self.lpips_fn(x, y).mean().item()

    def compute_all(self, x, y):
        """
        Compute all metrics at once.
        Args:
            x: Original image tensor [B,3,H,W] in range [0,1]
            y: Compressed image tensor [B,3,H,W] in range [0,1]
        """
        return {
            'psnr': self.psnr(x, y),
            'ms-ssim': self.msssim(x, y),
            'lpips': self.lpips(x, y)
        }