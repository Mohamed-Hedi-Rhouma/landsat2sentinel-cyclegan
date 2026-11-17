#!/usr/bin/env python
"""Super-resolution generator, used for both directions: landsat -> sentinel (up) and sentinel -> landsat (down)."""

import logging                                   # structured logging

import torch                                     # tensors
from torch import nn                             # layers
from torch.nn import functional as F             # interpolation and pooling

from cyclegan.model.blocks import RRDB, ConvBlock, Down, Up, conv3x3
from cyclegan.model.degradation import LANDSAT_MTF, downsample_mtf

logger = logging.getLogger(__name__)

DIRECTIONS = ("up", "down")                      # up = landsat -> sentinel, down = sentinel -> landsat


class Generator(nn.Module):
    """Interpolated input + learned correction; the rrdb body always works on the landsat grid.

    up:   x -> stem -> rrdb body -> Up(factor) -> refine -> head, + bicubic upsampling of x
    down: x -> Down(factor) -> rrdb body -> refine -> head,       + mtf downsampling of x (landsat view)
    """

    def __init__(
        self,
        direction: str,                          # 'up' or 'down'
        bands: int = 6,                          # bands in and out
        factor: int = 3,                         # resolution ratio between the sensors
        channels: int = 64,                      # feature width
        growth: int = 32,                        # channels added by each conv of a dense block
        nb_rrdb: int = 6,                        # rrdb blocks in the body
        mtf: float = LANDSAT_MTF,                # down only: blur of the base, as the landsat sensor sees
    ):
        """Build the stem, the body and the head for one direction."""
        super().__init__()
        assert direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}"
        self.direction = direction
        self.factor = factor
        self.mtf = mtf

        # Entry: to features on the landsat grid
        self.stem = conv3x3(bands, channels) if direction == "up" else Down(bands, channels, factor)

        # Body: rrdb blocks, with a long skip around them
        self.body = nn.Sequential(*(RRDB(channels, growth) for _ in range(nb_rrdb)))
        self.body_end = conv3x3(channels, channels)

        # Exit: to the output grid, then back to bands
        self.resize = Up(channels, channels, factor) if direction == "up" else nn.Identity()
        self.refine = ConvBlock(channels, channels)
        self.head = conv3x3(channels, bands)
        nn.init.zeros_(self.head.weight)                         # start as the interpolated input
        nn.init.zeros_(self.head.bias)

        n = sum(p.numel() for p in self.parameters())
        logger.info("Generator %s: x%d, %d rrdb, %d channels, %s params", direction, factor, nb_rrdb, channels, f"{n:,}")

    def base(self, x: torch.Tensor) -> torch.Tensor:
        """Input simply resized to the output grid: bicubic up, or mtf blur and decimation down."""
        if self.direction == "up":
            return F.interpolate(x, scale_factor=self.factor, mode="bicubic", align_corners=False)
        return downsample_mtf(x, self.factor, self.mtf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """up: b 6 h w -> b 6 3h 3w. down: b 6 3h 3w -> b 6 h w."""
        features = self.stem(x)                                  # b 64 h w, on the landsat grid
        features = features + self.body_end(self.body(features))  # long skip
        features = self.refine(self.resize(features))            # to the output grid
        return self.base(x) + self.head(features)                # interpolated input + correction
