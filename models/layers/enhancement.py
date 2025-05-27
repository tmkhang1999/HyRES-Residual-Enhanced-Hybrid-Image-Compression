import torch
import torch.nn as nn
import torch.nn.functional as F


# ——— Spatial Attention (CBAM’s SA) ———
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # we pool channel-wise and run a conv on the 2-channel map
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        _max, _ = x.max(dim=1, keepdim=True)  # (B,1,H,W)
        attn = torch.cat([avg, _max], dim=1)  # (B,2,H,W)
        attn = self.conv(attn)  # (B,1,H,W)
        return self.sigmoid(attn)  # (B,1,H,W)


# ——— (Optional) Re-use your SEBlock here if you like ———
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ——— Helper for a 3×3 conv with arbitrary dilation ———
def dilated_conv(ch_in, ch_out, dilation):
    return nn.Conv2d(
        ch_in, ch_out,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        bias=True
    )


# ——— MultiScaleRefine with dilations + spatial attention ———
class MultiScaleRefine(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64):
        super().__init__()

        # initial conv + SE
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.act_in = nn.PReLU()
        self.se_block = SEBlock(mid_channels, reduction=16)

        # each scale block uses one standard conv + one dilated conv
        def make_block():
            return nn.Sequential(
                dilated_conv(mid_channels, mid_channels, dilation=1),  # local
                nn.PReLU(),
                dilated_conv(mid_channels, mid_channels, dilation=2),  # wider RF
                nn.PReLU()
            )

        self.scale1 = make_block()
        self.scale2 = make_block()
        self.scale3 = make_block()

        # spatial attention right before fusion
        self.spatial_att = SpatialAttention(kernel_size=7)

        # fusion back to RGB
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # — initial features with channel attention —
        feat = self.act_in(self.conv_in(x))
        feat = self.se_block(feat)

        # — scale 1 (orig) —
        feat1 = self.scale1(feat)

        # — scale 2 (½) —
        feat2 = F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False)
        feat2 = self.scale2(feat2)
        feat2 = F.interpolate(feat2, size=feat.shape[2:], mode='bilinear', align_corners=False)

        # — scale 3 (¼) —
        feat3 = F.interpolate(feat, scale_factor=0.25, mode='bilinear', align_corners=False)
        feat3 = self.scale3(feat3)
        feat3 = F.interpolate(feat3, size=feat.shape[2:], mode='bilinear', align_corners=False)

        # — concat & spatial‐attention —
        multi = torch.cat([feat1, feat2, feat3], dim=1)
        attn = self.spatial_att(multi)  # (B,1,H,W)
        multi = multi * attn

        # — fuse back & residual —
        out = self.fusion(multi)
        return out
