import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class VGGLoss(nn.Module):
    """VGG-based perceptual loss for image compression."""

    def __init__(self, layer_ids=None):
        """
        Initialize VGG loss module.
        Args:
            layer_ids: List of VGG16 layer indices to extract features from
        """
        super().__init__()

        # Load pretrained VGG16 model
        if layer_ids is None:
            layer_ids = [2, 7, 14, 21, 28]
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Create sequential blocks up to each chosen layer
        self.slices = nn.ModuleList()
        start = 0
        for layer_id in layer_ids:
            self.slices.append(vgg[start:layer_id + 1])
            start = layer_id + 1

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x, y):
        """
        Compute VGG perceptual loss between original and reconstructed images.
        Args:
            x: Reconstructed image tensor [B,3,H,W] in range [0,1]
            y: Original image tensor [B,3,H,W] in range [0,1]
        Returns:
            Total perceptual loss (weighted sum of L1 distances between feature maps)
        """
        # Normalize inputs
        x = self.normalize(x)
        y = self.normalize(y)

        # Compute feature maps and L1 distances
        loss = 0
        device = x.device
        for slice in self.slices:
            slice = slice.to(device)
            x = slice(x)
            y = slice(y)
            loss += torch.abs(x - y).mean()

        return loss