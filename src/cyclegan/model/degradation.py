#!/usr/bin/env python
"""Simulate the landsat view of a 10 m image (mtf blur), resize masks, and fit a per-band gain and offset."""

import math                                      # sigma formula

import torch                                     # tensors
from torch.nn import functional as F             # conv and pooling

LANDSAT_MTF = 0.3                                # landsat mtf at nyquist, same for the 6 bands
EPS = 1e-6                                       # avoids a division by zero in the fit


# =============================================================================
# MTF blur
# =============================================================================
def mtf_sigma(mtf: float, factor: int) -> float:
    """Width of the gaussian whose mtf equals mtf at the nyquist of the coarse grid, in fine pixels.

    mtf 0.3, factor 3 -> sigma ~ 1.48 sentinel pixels.
    """
    return factor * math.sqrt(-2.0 * math.log(mtf)) / math.pi


def gaussian_kernel(sigma: float) -> torch.Tensor:
    """1d gaussian of size 2 * ceil(3 sigma) + 1, summing to 1."""
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of every band, rows then columns, borders mirrored; b c h w -> b c h w."""
    kernel = gaussian_kernel(sigma).to(x)
    c, radius = x.shape[1], len(kernel) // 2
    rows = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)                 # vertical pass
    cols = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)                 # horizontal pass
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(F.conv2d(x, rows, groups=c), cols, groups=c)


def downsample_mtf(x: torch.Tensor, factor: int = 3, mtf: float = LANDSAT_MTF) -> torch.Tensor:
    """What the coarse sensor would record: mtf blur, then the centre pixel of each factor x factor block."""
    offset = factor // 2                                               # centre of the block
    return blur(x, mtf_sigma(mtf, factor))[..., offset::factor, offset::factor]


# =============================================================================
# Masks
# =============================================================================
def shrink_mask(mask: torch.Tensor, factor: int = 3) -> torch.Tensor:
    """Fine mask to coarse grid: a coarse pixel is good only if its whole block is good; b 1 h w."""
    bad = (~mask).float()
    return F.max_pool2d(bad, factor) == 0


def expand_mask(mask: torch.Tensor, factor: int = 3) -> torch.Tensor:
    """Coarse mask to fine grid: each pixel copied into a factor x factor block; b 1 h w."""
    return mask.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


# =============================================================================
# Per-band gain and offset
# =============================================================================
def fit_gain_offset(src: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """src mapped onto tgt with one gain and offset per image and band, fitted on good pixels.

    The fit is detached: it is a fixed reference, the model cannot lower the loss by changing it.
    src, tgt: b c h w. mask: b 1 h w, True = good pixel. Returns gain * src + offset, b c h w.
    """
    with torch.no_grad():
        weight = mask.float()
        count = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        src_mean = (src * weight).sum(dim=(2, 3), keepdim=True) / count
        tgt_mean = (tgt * weight).sum(dim=(2, 3), keepdim=True) / count
        src_centered, tgt_centered = src - src_mean, tgt - tgt_mean
        covariance = (src_centered * tgt_centered * weight).sum(dim=(2, 3), keepdim=True) / count
        variance = (src_centered ** 2 * weight).sum(dim=(2, 3), keepdim=True) / count
        gain = covariance / (variance + EPS)
        offset = tgt_mean - gain * src_mean
        return gain * src + offset
