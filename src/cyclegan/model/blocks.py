#!/usr/bin/env python
"""Super-resolution building blocks (esrgan style): conv block, dense block, rrdb, down and up sampling."""

import torch                                     # tensors
from torch import nn                             # layers

SLOPE = 0.2                                      # leaky relu negative slope
RESIDUAL_SCALE = 0.2                             # each residual branch is scaled down before the add


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    """3x3 conv that keeps the size, borders mirrored instead of zero."""
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect")


class ConvBlock(nn.Module):
    """Conv -> leaky relu, no normalization so the reflectance level is kept; the size is kept."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Square conv with reflect padding."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, padding_mode="reflect")
        self.act = nn.LeakyReLU(SLOPE, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b out h w."""
        return self.act(self.conv(x))


class DenseBlock(nn.Module):
    """Five convs, each fed with the input and every previous output; x + 0.2 * last."""

    def __init__(self, channels: int = 64, growth: int = 32):
        """Each conv adds growth channels, the last one goes back to channels."""
        super().__init__()
        self.layers = nn.ModuleList(
            ConvBlock(channels + k * growth, growth) for k in range(4)     # 64, 96, 128, 160 -> 32
        )
        self.last = conv3x3(channels + 4 * growth, channels)                # 192 -> 64, no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b c h w."""
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return x + RESIDUAL_SCALE * self.last(torch.cat(features, dim=1))


class RRDB(nn.Module):
    """Residual in residual dense block: three dense blocks, then x + 0.2 * result."""

    def __init__(self, channels: int = 64, growth: int = 32):
        """Three dense blocks of the same width."""
        super().__init__()
        self.blocks = nn.Sequential(*(DenseBlock(channels, growth) for _ in range(3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b c h w."""
        return x + RESIDUAL_SCALE * self.blocks(x)


class Down(nn.Module):
    """Divide the size by factor: each output pixel summarises one factor x factor block."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        """Conv with kernel and stride equal to factor, then leaky relu."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=factor, stride=factor)
        self.act = nn.LeakyReLU(SLOPE, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b out h/factor w/factor."""
        return self.act(self.conv(x))


class Up(nn.Module):
    """Multiply the size by factor: nearest copy, then a conv block refines it."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        """Nearest upsampling has no weights; the conv block does the learning."""
        super().__init__()
        self.up = nn.Upsample(scale_factor=factor, mode="nearest")
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b c h w -> b out h*factor w*factor."""
        return self.block(self.up(x))
