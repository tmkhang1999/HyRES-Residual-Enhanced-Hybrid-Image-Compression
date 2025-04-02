import math
import time

import torch
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import Cheng2020Anchor
from compressai.models.utils import update_registered_buffers
from compressai.ops import quantize_ste

from models.layers import CheckboardMaskedConv2d

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Cheng2020withCheckerboard(Cheng2020Anchor):
    """Modified Cheng2020 model with checkerboard masking for improved compression.

    This model uses a shared entropy parameters network for both anchor and non-anchor pixels,
    along with checkerboard context masking for more efficient compression.
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N, **kwargs)
        self.context_prediction = CheckboardMaskedConv2d(
            in_channels=N, out_channels=N * 2, kernel_size=5, stride=1, padding=2
        )

    # -------------------------- Core Model Methods -------------------------- #

    def forward(self, x):
        """Forward pass used during training with noise-based quantization."""
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        # Generate parameters
        hyper_params = self.h_s(z_hat)
        ctx_params = self.context_prediction(y_hat)
        # Mask anchor positions
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        # Calculate gaussian parameters and likelihoods
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        # Reconstruct
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def validate(self, x):
        """Validation pass using STE-based quantization for more accurate distortion estimation."""
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Quantize latents
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        # Generate anchor parameters
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros_like(y.repeat(1, 2, 1, 1))
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, hyper_params], dim=1)
        )

        # Mask and split parameters
        gaussian_params_anchor[:, :, 0::2, 0::2] = 0
        gaussian_params_anchor[:, :, 1::2, 1::2] = 0
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        # Generate non-anchor parameters using quantized anchors
        y_hat_anchor = quantize_ste(y - means_anchor) + means_anchor
        ctx_params = self.context_prediction(y_hat_anchor)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        # Final parameters and reconstruction
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = quantize_ste(y - means_hat) + means_hat
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # -------------------------- Compression Methods -------------------------- #

    def compress(self, x):
        """Compress input tensor using two-pass encoding with STE quantization."""
        torch.backends.cudnn.deterministic = True

        # Setup entropy coding
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Generate and compress latents
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # First pass: Compress anchor pixels
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self._compress_anchor_pixels(y, scales_anchor, means_anchor, symbols_list, indexes_list)

        # Second pass: Compress non-anchor pixels
        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        _ = self._compress_nonanchor_pixels(y, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)

        # Finalize compression
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()

        return {
            "strings": [[y_string], z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        """Decompress latent representations back into image."""
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        # Setup entropy coding
        y_string = strings[0][0]
        z_strings = strings[1]
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Decompress latents
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)

        # First pass: Decompress anchor pixels
        ctx_params_anchor = torch.zeros(
            [z_hat.size(0), self.M * 2, z_hat.size(2) * 4, z_hat.size(3) * 4],
            device=z_hat.device
        )
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self._decompress_anchor_pixels(scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)

        # Second pass: Decompress non-anchor pixels
        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self._decompress_nonanchor_pixels(
            scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets
        )

        # Reconstruct final image
        y_hat = anchor_hat + nonanchor_hat
        x_hat = self.g_s(y_hat)

        torch.cuda.synchronize()
        cost_time = time.process_time() - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    # -------------------------- Helper Methods -------------------------- #

    def _compress_anchor_pixels(self, anchor, scales, means, symbols_list, indexes_list):
        """Compress anchor pixels using entropy coding."""
        anchor_squeeze = self._checkerboard_squeeze(anchor, "anchor")
        scales_squeeze = self._checkerboard_squeeze(scales, "anchor")
        means_squeeze = self._checkerboard_squeeze(means, "anchor")

        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(anchor_squeeze, "symbols", means_squeeze)

        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())

        return self._checkerboard_unsqueeze(anchor_hat + means_squeeze, "anchor")

    def _compress_nonanchor_pixels(self, nonanchor, scales, means, symbols_list, indexes_list):
        """Compress non-anchor pixels using entropy coding."""
        nonanchor_squeeze = self._checkerboard_squeeze(nonanchor, "nonanchor")
        scales_squeeze = self._checkerboard_squeeze(scales, "nonanchor")
        means_squeeze = self._checkerboard_squeeze(means, "nonanchor")

        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_squeeze)

        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())

        return self._checkerboard_unsqueeze(nonanchor_hat + means_squeeze, "nonanchor")

    def _decompress_anchor_pixels(self, scales, means, decoder, cdf, cdf_lengths, offsets):
        """Decompress anchor pixels from entropy coded representation."""
        scales_squeeze = self._checkerboard_squeeze(scales, "anchor")
        means_squeeze = self._checkerboard_squeeze(means, "anchor")

        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        anchor_hat = torch.Tensor(anchor_hat).reshape(scales_squeeze.shape).to(scales.device) + means_squeeze

        return self._checkerboard_unsqueeze(anchor_hat, "anchor")

    def _decompress_nonanchor_pixels(self, scales, means, decoder, cdf, cdf_lengths, offsets):
        """Decompress non-anchor pixels from entropy coded representation."""
        scales_squeeze = self._checkerboard_squeeze(scales, "nonanchor")
        means_squeeze = self._checkerboard_squeeze(means, "nonanchor")

        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_squeeze.shape).to(scales.device) + means_squeeze

        return self._checkerboard_unsqueeze(nonanchor_hat, "nonanchor")

    def _checkerboard_squeeze(self, x, mode):
        """Squeeze tensor according to checkerboard pattern."""
        B, C, H, W = x.shape
        squeezed = torch.zeros([B, C, H, W // 2], device=x.device)

        if mode == "anchor":
            squeezed[:, :, 0::2, :] = x[:, :, 0::2, 1::2]
            squeezed[:, :, 1::2, :] = x[:, :, 1::2, 0::2]
        else:  # nonanchor
            squeezed[:, :, 0::2, :] = x[:, :, 0::2, 0::2]
            squeezed[:, :, 1::2, :] = x[:, :, 1::2, 1::2]

        return squeezed

    def _checkerboard_unsqueeze(self, x, mode):
        """Unsqueeze tensor according to checkerboard pattern."""
        B, C, H, W = x.shape
        unsqueezed = torch.zeros([B, C, H, W * 2], device=x.device)

        if mode == "anchor":
            unsqueezed[:, :, 0::2, 1::2] = x[:, :, 0::2, :]
            unsqueezed[:, :, 1::2, 0::2] = x[:, :, 1::2, :]
        else:  # nonanchor
            unsqueezed[:, :, 0::2, 0::2] = x[:, :, 0::2, :]
            unsqueezed[:, :, 1::2, 1::2] = x[:, :, 1::2, :]

        return unsqueezed

    # -------------------------- State Management -------------------------- #

    def load_state_dict(self, state_dict, **kwargs):
        """Load model state while handling special gaussian conditional buffers."""
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False, **kwargs):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated