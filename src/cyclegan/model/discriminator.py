#!/usr/bin/env python
"""Discriminators: multi-scale PatchGAN (one score per patch) or U-Net (one score per pixel)."""

import logging                                   # structured logging

import torch                                     # tensors
from torch import nn                             # layers
from torch.nn import functional as F             # pooling and interpolation
from torch.nn.utils import spectral_norm         # keeps the discriminator from changing too fast

logger = logging.getLogger(__name__)

SLOPE = 0.2                                      # leaky relu negative slope
BLUR_SIZE = 5                                    # box blur used by the high-pass input


# =============================================================================
# Helpers
# =============================================================================
def sn_conv(in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 1) -> nn.Module:
    """Conv with spectral norm; kernel 4 with padding 1 halves the size at stride 2."""
    padding = 1 if kernel_size == 4 else kernel_size // 2
    return spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))


def high_pass(x: torch.Tensor) -> torch.Tensor:
    """Image minus its blurred copy: keeps edges and textures, removes colours and levels."""
    pad = BLUR_SIZE // 2
    blurred = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), BLUR_SIZE, stride=1)
    return x - blurred


def score_masks(mask: torch.Tensor, scores: list[torch.Tensor]) -> list[torch.Tensor]:
    """True where a score only covers good pixels; mask is b 1 h w, True = good pixel."""
    bad = (~mask).float()
    return [F.adaptive_max_pool2d(bad, s.shape[-2:]) == 0 for s in scores]


def count_params(module: nn.Module) -> str:
    """Number of weights, readable."""
    return f"{sum(p.numel() for p in module.parameters()):,}"


# =============================================================================
# PatchGAN
# =============================================================================
class PatchDiscriminator(nn.Module):
    """b bands h w -> b 1 h' w': one score per patch of about 70 x 70 pixels."""

    def __init__(self, bands: int = 6, channels: int = 64, nb_down: int = 3):
        """nb_down stride-2 convs doubling the channels, one stride-1 conv, one conv to the score."""
        super().__init__()
        layers = [sn_conv(bands, channels, stride=2), nn.LeakyReLU(SLOPE, inplace=True)]
        width = channels
        for _ in range(nb_down - 1):                              # 64 -> 128 -> 256
            layers += [sn_conv(width, 2 * width, stride=2), nn.LeakyReLU(SLOPE, inplace=True)]
            width *= 2
        layers += [sn_conv(width, 2 * width), nn.LeakyReLU(SLOPE, inplace=True)]   # 256 -> 512
        layers += [sn_conv(2 * width, 1)]                                          # 512 -> score
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score grid, no activation (the loss compares it with real or fake)."""
        return self.layers(x)


class MultiScaleDiscriminator(nn.Module):
    """One patch discriminator per scale: full image, then halved again and again."""

    def __init__(self, bands: int = 6, channels: int = 64, nb_down: int = 3, nb_scales: int = 2,
                 highpass: bool = False):
        """nb_scales independent patch discriminators; highpass = judge details only."""
        super().__init__()
        self.highpass = highpass
        self.scales = nn.ModuleList(PatchDiscriminator(bands, channels, nb_down) for _ in range(nb_scales))
        logger.info("MultiScaleDiscriminator: %d scales, %d down, highpass %s, %s params",
                    nb_scales, nb_down, highpass, count_params(self))

    @staticmethod
    def halve(x: torch.Tensor) -> torch.Tensor:
        """Average pooling that halves the size."""
        return F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """One score grid per scale, finest first."""
        if self.highpass:
            x = high_pass(x)
        scores = []
        for discriminator in self.scales:
            scores.append(discriminator(x))
            x = self.halve(x)
        return scores


# =============================================================================
# U-Net
# =============================================================================
class UNetDiscriminator(nn.Module):
    """b bands h w -> one b 1 h w score map: encoder down by 8, decoder back up with skips (real-esrgan style).

    h and w must be multiples of 8.
    """

    def __init__(self, bands: int = 6, channels: int = 64, highpass: bool = False):
        """Three halvings, three doublings, skips added between the matching sizes."""
        super().__init__()
        self.highpass = highpass
        c = channels
        self.act = nn.LeakyReLU(SLOPE, inplace=True)

        # Encoder: h -> h/2 -> h/4 -> h/8
        self.enc0 = nn.Conv2d(bands, c, 3, padding=1)                  # b 64 h w
        self.enc1 = sn_conv(c, 2 * c, stride=2)                        # b 128 h/2
        self.enc2 = sn_conv(2 * c, 4 * c, stride=2)                    # b 256 h/4
        self.enc3 = sn_conv(4 * c, 8 * c, stride=2)                    # b 512 h/8

        # Decoder: h/8 -> h/4 -> h/2 -> h, each level adds the encoder features of the same size
        self.dec2 = sn_conv(8 * c, 4 * c, kernel_size=3)               # b 256 h/4, + enc2
        self.dec1 = sn_conv(4 * c, 2 * c, kernel_size=3)               # b 128 h/2, + enc1
        self.dec0 = sn_conv(2 * c, c, kernel_size=3)                   # b 64 h,    + enc0

        # Head: refine, then one score per pixel
        self.refine = nn.Sequential(sn_conv(c, c, kernel_size=3), nn.LeakyReLU(SLOPE, inplace=True),
                                    sn_conv(c, c, kernel_size=3), nn.LeakyReLU(SLOPE, inplace=True))
        self.score = nn.Conv2d(c, 1, 3, padding=1)
        logger.info("UNetDiscriminator: highpass %s, %s params", highpass, count_params(self))

    def up(self, x: torch.Tensor, conv: nn.Module, skip: torch.Tensor) -> torch.Tensor:
        """Double the size, conv, add the encoder features of that size."""
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(conv(x)) + skip

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """A list with one full-size score map, same interface as the multi-scale discriminator."""
        if self.highpass:
            x = high_pass(x)
        e0 = self.act(self.enc0(x))
        e1 = self.act(self.enc1(e0))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))

        d2 = self.up(e3, self.dec2, e2)
        d1 = self.up(d2, self.dec1, e1)
        d0 = self.up(d1, self.dec0, e0)
        return [self.score(self.refine(d0))]
