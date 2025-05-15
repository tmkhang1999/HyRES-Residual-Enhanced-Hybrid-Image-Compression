import math

import torch
import torch.nn as nn
from src.losses.vgg16 import VGGLoss


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.004, alpha=0.001):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg = VGGLoss()
        self.lmbda = lmbda
        self.alpha = alpha

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["residual_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # Total bpp is the sum of both components
        out["bpp_loss"] = out["residual_bpp_loss"] + output["jpeg_bpp_loss"]

        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["vgg_loss"] = self.vgg(output["x_hat"], target) * 255 ** 2

        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"] + self.alpha * out["vgg_loss"]

        return out

