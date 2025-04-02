import math

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN
from compressai.models.base import CompressionModel
from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import quantize_ste
from timm.models.layers import trunc_normal_

from models.layers import AttentionBlock, conv3x3, conv1x1, CheckboardMaskedConv2d
from models.utils.quantization import Quantizer

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class LightWeightELIC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices

        # Channel groups for different slices
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth

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
            AttentionBlock(M),
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
            deconv(N, 3),
        )

        # Hyper-analysis transform
        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        # Hyper-synthesis transform
        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M),
        )

        # Create transforms for context conditioning
        self._create_transforms(num_slices)

        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)

    def _create_transforms(self, num_slices):
        # Context channel transforms
        self.cc_transforms = nn.ModuleList([
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 224, stride=1,
                     kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            ) for i in range(1, num_slices)
        ])

        # Context prediction modules
        self.context_prediction = nn.ModuleList([
            CheckboardMaskedConv2d(
                self.groups[i + 1], 2 * self.groups[i + 1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        ])

        # Parameter aggregation networks
        self.ParamAggregation = nn.ModuleList([
            nn.Sequential(
                conv1x1(640 + self.groups[i + 1 if i > 0 else 0] * 2 + self.groups[i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1] * 2),
            ) for i in range(num_slices)
        ])

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def load_state_dict(self, state_dict, **kwargs):
        # Update Gaussian conditional buffers
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

    def update(self, scale_table=None, force=False, **kwargs):
        # Use default scale table if none provided
        if scale_table is None:
            scale_table = get_scale_table()

        # Update scale table and other components
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def _split_channels(self, tensor):
        """Split tensor into anchor and non-anchor parts using checkerboard pattern"""
        anchor = torch.zeros_like(tensor)
        non_anchor = torch.zeros_like(tensor)

        anchor[:, :, 0::2, 0::2] = tensor[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = tensor[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = tensor[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = tensor[:, :, 1::2, 0::2]

        return anchor, non_anchor

    def _process_slice(self, slice_index, y_slice, anchor_split, non_anchor_split,
                       ctx_params_anchor_split, y_hat_slices, latent_means, latent_scales,
                       noisequant, device):
        """Process a single slice with checkerboard context model"""
        # Get support info from previous slices
        if slice_index == 0:
            support = torch.cat([latent_means, latent_scales], dim=1)
        elif slice_index == 1:
            support_slices = y_hat_slices[0]
            support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
            support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            support = torch.cat([support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
        else:
            support_slices = torch.cat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
            support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
            support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            support = torch.cat([support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)

        # Process anchor pixels
        y_anchor = anchor_split[slice_index]
        means_anchor, scales_anchor = self.ParamAggregation[slice_index](
            torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

        # Initialize full tensors for means and scales
        scales_hat_split = torch.zeros_like(y_anchor, device=device)
        means_hat_split = torch.zeros_like(y_anchor, device=device)

        # Fill anchor positions
        scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
        scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
        means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
        means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

        # Quantize anchor values
        if noisequant:
            y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
            y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
        else:
            y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
            y_anchor_quantilized_for_gs = y_anchor_quantilized.clone()

        # Zero out non-anchor positions
        y_anchor_quantilized[:, :, 0::2, 1::2] = 0
        y_anchor_quantilized[:, :, 1::2, 0::2] = 0
        y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
        y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

        # Process non-anchor pixels using context from anchor pixels
        masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
        means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
            torch.cat([masked_context, support], dim=1)).chunk(2, 1)

        # Fill non-anchor positions
        scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
        scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
        means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
        means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

        # Entropy estimation
        _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

        # Quantize non-anchor values
        y_non_anchor = non_anchor_split[slice_index]
        if noisequant:
            y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
            y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
        else:
            y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                               "ste") + means_non_anchor
            y_non_anchor_quantilized_for_gs = y_non_anchor_quantilized.clone()

        # Zero out anchor positions
        y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
        y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
        y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
        y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

        # Combine anchor and non-anchor parts
        y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
        y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs

        return y_hat_slice, y_hat_slice_for_gs, y_slice_likelihood

    def forward(self, x, noisequant=False):
        # Analysis transform
        y = self.g_a(x)
        B, C, H, W = y.size()
        device = x.device

        # Hyperprior branch
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        # Get latent parameters
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # Split y into anchor and non-anchor parts
        anchor, non_anchor = self._split_channels(y)

        # Split into slices based on channel groups
        y_slices = torch.split(y, self.groups[1:], 1)
        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W, device=device),
                                              [2 * i for i in self.groups[1:]], 1)

        # Process each slice
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            y_hat_slice, y_hat_slice_for_gs, y_slice_likelihood = self._process_slice(
                slice_index, y_slice, anchor_split, non_anchor_split,
                ctx_params_anchor_split, y_hat_slices, latent_means, latent_scales,
                noisequant, device
            )

            y_hat_slices.append(y_hat_slice)
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        # Combine results
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)

        # Synthesis transform
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def _get_support_for_slice(self, slice_index, y_hat_slices, latent_means, latent_scales):
        """Get support tensor for a specific slice based on previous slices and hyperprior"""
        if slice_index == 0:
            return torch.cat([latent_means, latent_scales], dim=1)
        elif slice_index == 1:
            support_slices = y_hat_slices[0]
            support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
            support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            return torch.cat([support_slices_ch_mean, support_slices_ch_scale,
                              latent_means, latent_scales], dim=1)
        else:
            support_slices = torch.cat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
            support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
            support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            return torch.cat([support_slices_ch_mean, support_slices_ch_scale,
                              latent_means, latent_scales], dim=1)

    def _process_anchor_checkerboard(self, y_slice, slice_index, ctx_params, support, device):
        """Process anchor pixels in checkerboard pattern for compression"""
        # Calculate anchor parameters
        params_concat = torch.cat([ctx_params, support], dim=1)
        means_anchor, scales_anchor = self.ParamAggregation[slice_index](params_concat).chunk(2, 1)

        # Get dimensions
        B, C, H, W = y_slice.size()

        # Create encoding tensors that are half the width (checkerboard pattern)
        y_anchor_encode = torch.zeros(B, C, H, W // 2, device=device)
        means_anchor_encode = torch.zeros(B, C, H, W // 2, device=device)
        scales_anchor_encode = torch.zeros(B, C, H, W // 2, device=device)

        # Map checkerboard pattern to encoding tensors
        y_anchor_encode[:, :, 0::2, :] = y_slice[:, :, 0::2, 0::2]
        y_anchor_encode[:, :, 1::2, :] = y_slice[:, :, 1::2, 1::2]
        means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
        means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
        scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
        scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

        # Compress anchor pixels
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
        anchor_strings = self.gaussian_conditional.compress(
            y_anchor_encode, indexes_anchor, means=means_anchor_encode)

        # Decompress anchor pixels for context model
        anchor_quantized = self.gaussian_conditional.decompress(
            anchor_strings, indexes_anchor, means=means_anchor_encode)

        # Create decoded anchor tensor
        y_anchor_decode = torch.zeros(B, C, H, W, device=device)
        y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
        y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

        return anchor_strings, y_anchor_decode, means_anchor, scales_anchor

    def _process_nonanchor_checkerboard(self, y_slice, y_anchor_decode, slice_index, support, device):
        """Process non-anchor pixels in checkerboard pattern for compression"""
        # Get context from anchor pixels
        masked_context = self.context_prediction[slice_index](y_anchor_decode)

        # Calculate non-anchor parameters
        params_concat = torch.cat([masked_context, support], dim=1)
        means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](params_concat).chunk(2, 1)

        # Get dimensions
        B, C, H, W = y_slice.size()

        # Create encoding tensors for non-anchor pixels
        y_non_anchor_encode = torch.zeros(B, C, H, W // 2, device=device)
        means_non_anchor_encode = torch.zeros(B, C, H, W // 2, device=device)
        scales_non_anchor_encode = torch.zeros(B, C, H, W // 2, device=device)

        # Map checkerboard pattern to encoding tensors
        y_non_anchor_encode[:, :, 0::2, :] = y_slice[:, :, 0::2, 1::2]
        y_non_anchor_encode[:, :, 1::2, :] = y_slice[:, :, 1::2, 0::2]
        means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
        means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
        scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
        scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

        # Compress non-anchor pixels
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
        non_anchor_strings = self.gaussian_conditional.compress(
            y_non_anchor_encode, indexes_non_anchor, means=means_non_anchor_encode)

        # Decompress non-anchor pixels
        non_anchor_quantized = self.gaussian_conditional.decompress(
            non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode)

        # Create decoded non-anchor tensor
        y_non_anchor_decoded = torch.zeros_like(means_non_anchor)
        y_non_anchor_decoded[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
        y_non_anchor_decoded[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

        return non_anchor_strings, y_non_anchor_decoded

    def compress(self, x):
        import time

        # Analysis transform
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()
        device = x.device

        # Hyperprior branch
        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        _, z_likelihoods = self.entropy_bottleneck(z)  # Calculate z likelihoods
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Get latent parameters
        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        # Split y into slices based on channel groups
        y_slices = torch.split(y, self.groups[1:], 1)

        # Create empty context parameters for the first iteration
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W, device=device),
            [2 * i for i in self.groups[1:]], 1
        )

        # Process all slices
        params_start = time.time()
        y_strings = []
        y_hat_slices = []
        y_likelihoods_list = []

        for slice_index, y_slice in enumerate(y_slices):
            # Get support information from previous slices and hyperprior
            support = self._get_support_for_slice(slice_index, y_hat_slices, latent_means, latent_scales)

            # Process anchor pixels
            anchor_strings, y_anchor_decode, means_anchor, scales_anchor = self._process_anchor_checkerboard(
                y_slice, slice_index, ctx_params_anchor_split[slice_index], support, device)

            # Process non-anchor pixels
            non_anchor_strings, y_non_anchor_decoded = self._process_nonanchor_checkerboard(
                y_slice, y_anchor_decode, slice_index, support, device)

            # Calculate likelihoods for this slice
            slice_likelihood = torch.zeros_like(y_slice)
            anchor, non_anchor = self._split_channels(y_slice)

            # Fill anchor likelihoods
            slice_likelihood[:, :, 0::2, 0::2] = self.gaussian_conditional.likelihood(
                anchor[:, :, 0::2, 0::2], scales_anchor[:, :, 0::2, 0::2], means_anchor[:, :, 0::2, 0::2])
            slice_likelihood[:, :, 1::2, 1::2] = self.gaussian_conditional.likelihood(
                anchor[:, :, 1::2, 1::2], scales_anchor[:, :, 1::2, 1::2], means_anchor[:, :, 1::2, 1::2])

            # Fill non-anchor likelihoods using the last calculated parameters
            slice_likelihood[:, :, 0::2, 1::2] = self.gaussian_conditional.likelihood(
                non_anchor[:, :, 0::2, 1::2],
                y_non_anchor_decoded[:, :, 0::2, 1::2],
                means=y_anchor_decode[:, :, 0::2, 1::2])
            slice_likelihood[:, :, 1::2, 0::2] = self.gaussian_conditional.likelihood(
                non_anchor[:, :, 1::2, 0::2],
                y_non_anchor_decoded[:, :, 1::2, 0::2],
                means=y_anchor_decode[:, :, 1::2, 0::2])

            y_likelihoods_list.append(slice_likelihood)

            # Combine anchor and non-anchor parts
            y_slice_hat = y_anchor_decode + y_non_anchor_decoded
            y_hat_slices.append(y_slice_hat)
            y_strings.append([anchor_strings, non_anchor_strings])

        params_time = time.time() - params_start
        y_likelihoods = torch.cat(y_likelihoods_list, dim=1)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "params": params_time}
        }

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # Decompress hyperprior
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B = z_hat.size(0)
        device = z_hat.device

        # Get latent parameters
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # Get y shape based on z shape (4x upsampling in each dimension)
        y_height, y_width = shape[0] * 4, shape[1] * 4
        y_strings = strings[0]

        # Create empty context parameters for the first iteration
        ctx_params_anchor = torch.zeros(
            (B, self.M * 2, y_height, y_width), device=device
        )
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor, [2 * i for i in self.groups[1:]], 1
        )

        # Decompress all slices
        y_hat_slices = []

        for slice_index in range(len(self.groups) - 1):
            # Get support information from previous slices and hyperprior
            support = self._get_support_for_slice(slice_index, y_hat_slices, latent_means, latent_scales)

            # Calculate anchor parameters
            params_concat = torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)
            means_anchor, scales_anchor = self.ParamAggregation[slice_index](params_concat).chunk(2, 1)

            # Prepare tensors for anchor decoding
            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2, device=device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2, device=device)

            # Map anchor means and scales to encoding format
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            # Decompress anchor pixels
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            # Map decompressed anchor values back to full resolution
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor, device=device)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # Calculate non-anchor parameters using context from anchor pixels
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            params_concat = torch.cat([masked_context, support], dim=1)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](params_concat).chunk(2, 1)

            # Prepare tensors for non-anchor decoding
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2, device=device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2, device=device)

            # Map non-anchor means and scales to encoding format
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            # Decompress non-anchor pixels
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode
            )

            # Map decompressed non-anchor values back to full resolution
            y_non_anchor_decoded = torch.zeros_like(means_anchor)
            y_non_anchor_decoded[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_decoded[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            # Combine anchor and non-anchor parts
            y_slice_hat = y_anchor_decode + y_non_anchor_decoded
            y_hat_slices.append(y_slice_hat)

        # Combine all slices
        y_hat = torch.cat(y_hat_slices, dim=1)

        # Synthesis transform
        import time
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start

        return {"x_hat": x_hat, "time": {"y_dec": y_dec}}

    def inference(self, x):
        # Get compression results
        compress_result = self.compress(x)
        time_record = compress_result["time"]

        # Get decompression results
        decompress_result = self.decompress(compress_result["strings"], compress_result["shape"])
        time_record["y_dec"] = decompress_result["time"]["y_dec"]

        return {
            "x_hat": decompress_result["x_hat"],
            "likelihoods": compress_result["likelihoods"],
            "time": time_record
        }
