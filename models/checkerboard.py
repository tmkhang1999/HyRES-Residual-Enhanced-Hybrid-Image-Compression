import math
import time

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.base import CompressionModel
from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import quantize_ste

from models.layers import AttentionBlock, conv3x3, conv1x1, CheckboardMaskedConv2d
from models.utils.quantization import Quantizer


SCALES_MIN, SCALES_MAX, SCALES_LEVELS = 0.11, 256, 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class LightWeightCheckerboard(CompressionModel):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N, self.M = N, M

        # Entropy bottleneck and quantizer
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.quantizer = Quantizer()

        # Analysis transform
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N),
            GDN(N),
            ResidualBottleneckBlock(N, N),
            conv(N, M),
            AttentionBlock(M)
        )

        # Synthesis transform
        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneckBlock(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            GDN(N, inverse=True),
            deconv(N, 3)
        )

        # Hyper transforms
        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N)
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M)
        )

        # Context and parameter networks
        self.context_prediction = CheckboardMaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.param_aggregation = nn.Sequential(
            conv1x1(4 * M, 640),
            nn.ReLU(inplace=True),
            conv1x1(640, 512),
            nn.ReLU(inplace=True),
            conv1x1(512, 2 * M)
        )

    def forward(self, x, noisequant=False):
        # Analysis and hyperprior
        y = self.g_a(x)
        device = x.device

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        latent_params = self.h_s(z_hat)

        # Split into anchor/non-anchor
        y_anchor = torch.zeros_like(y, device=device)
        y_non_anchor = torch.zeros_like(y, device=device)

        y_anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        y_anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        y_non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        y_non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        # Process anchor pixels
        anchor_params = self.param_aggregation(
            torch.cat([latent_params, torch.zeros_like(latent_params)], dim=1)
        )
        scales_anchor, means_anchor = anchor_params.chunk(2, 1)

        # Quantize anchor pixels
        y_anchor_hat = (self.quantizer.quantize(y_anchor, "noise") if noisequant
                        else self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor)

        # Process non-anchor pixels using anchor context
        ctx_params = self.context_prediction(y_anchor_hat)
        non_anchor_params = self.param_aggregation(
            torch.cat([latent_params, ctx_params], dim=1)
        )
        scales_non_anchor, means_non_anchor = non_anchor_params.chunk(2, 1)

        # Quantize non-anchor pixels
        y_non_anchor_hat = (self.quantizer.quantize(y_non_anchor, "noise") if noisequant
                            else self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste") + means_non_anchor)

        # Combine and reconstruct
        y_hat = y_anchor_hat + y_non_anchor_hat
        x_hat = self.g_s(y_hat)

        # Calculate likelihoods
        scales = scales_anchor + scales_non_anchor
        means = means_anchor + means_non_anchor
        _, y_likelihoods = self.gaussian_conditional(y, scales, means=means)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def _split_tensor(self, x, mode):
        split = torch.zeros_like(x)
        if mode == "anchor":
            split[:, :, 0::2, 0::2] = x[:, :, 0::2, 0::2]
            split[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2]
        else:
            split[:, :, 0::2, 1::2] = x[:, :, 0::2, 1::2]
            split[:, :, 1::2, 0::2] = x[:, :, 1::2, 0::2]
        return split

    def _compress_part(self, x, scales, means):
        indexes = self.gaussian_conditional.build_indexes(scales)
        return self.gaussian_conditional.compress(x, indexes, means=means)

    def _decompress_part(self, strings, scales, means):
        indexes = self.gaussian_conditional.build_indexes(scales)
        return self.gaussian_conditional.decompress(strings, indexes, means=means)

    def compress(self, x):
        # Analysis
        start_time = time.time()
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_params = self.h_s(z_hat)

        # Split and compress anchor pixels
        y_anchor = self._split_tensor(y, "anchor")
        anchor_params = self.param_aggregation(
            torch.cat([latent_params, torch.zeros_like(latent_params)], dim=1)
        )
        scales_anchor, means_anchor = anchor_params.chunk(2, 1)
        anchor_strings = self._compress_part(y_anchor, scales_anchor, means_anchor)

        # Split and compress non-anchor pixels
        y_anchor_hat = self._decompress_part(anchor_strings, scales_anchor, means_anchor)
        ctx_params = self.context_prediction(y_anchor_hat)
        non_anchor_params = self.param_aggregation(
            torch.cat([latent_params, ctx_params], dim=1)
        )
        scales_non_anchor, means_non_anchor = non_anchor_params.chunk(2, 1)
        y_non_anchor = self._split_tensor(y, "non_anchor")
        non_anchor_strings = self._compress_part(y_non_anchor, scales_non_anchor, means_non_anchor)

        return {
            "strings": [[anchor_strings, non_anchor_strings], z_strings],
            "shape": z.size()[-2:],
            "time": time.time() - start_time
        }

    def decompress(self, strings, shape):
        """Decompress from entropy-coded strings."""
        # Start timing
        start_time = time.time()

        # Decompress hyperprior
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_params = self.h_s(z_hat)

        # Get dimensions for context parameters
        B = z_hat.size(0)
        H, W = shape[0] * 4, shape[1] * 4  # 4x upsampling
        device = z_hat.device

        # Process anchor pixels
        anchor_params = self.param_aggregation(
            torch.cat([latent_params, torch.zeros_like(latent_params)], dim=1)
        )
        scales_anchor, means_anchor = anchor_params.chunk(2, 1)

        # Decompress anchor pixels
        y_anchor_hat = self._decompress_part(strings[0][0], scales_anchor, means_anchor)

        # Process non-anchor pixels using context
        ctx_params = self.context_prediction(y_anchor_hat)
        non_anchor_params = self.param_aggregation(
            torch.cat([latent_params, ctx_params], dim=1)
        )
        scales_non_anchor, means_non_anchor = non_anchor_params.chunk(2, 1)

        # Decompress non-anchor pixels
        y_non_anchor_hat = self._decompress_part(strings[0][1], scales_non_anchor, means_non_anchor)

        # Combine and reconstruct
        y_hat = y_anchor_hat + y_non_anchor_hat
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat,
            "time": time.time() - start_time
        }

    def inference(self, x):
        """Perform inference (compress + decompress) and measure time."""
        # Compress
        compress_result = self.compress(x)
        compression_time = compress_result["time"]

        # Decompress
        decompress_result = self.decompress(compress_result["strings"], compress_result["shape"])
        decompression_time = decompress_result["time"]

        return {
            "x_hat": decompress_result["x_hat"],
            "time": {
                "compression": compression_time,
                "decompression": decompression_time,
                "total": compression_time + decompression_time
            }
        }

    def update(self, scale_table=None, force=False, **kwargs):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict, **kwargs):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == "__main__":
    model = LightWeightCheckerboard(N=192, M=320)
    input = torch.Tensor(1, 3, 256, 256)
    output = model(input)
    print(output["x_hat"].shape)  # Should be (1, 3, 256, 256)