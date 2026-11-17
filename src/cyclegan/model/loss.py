#!/usr/bin/env python
"""CycleGAN losses: gan, cycle, lpips, sam and landsat-view consistency, all masked on good pixels."""

import logging                                   # structured logging
from dataclasses import dataclass                # discriminator scores of one sensor

import torch                                     # tensors
from einops import rearrange, repeat             # bands as separate images for lpips
from piq import LPIPS  # type: ignore            # vgg perceptual distance
from torch import nn                             # module base class
from torch.nn import functional as F             # huber and l1

from cyclegan.data.datamodule import Batch
from cyclegan.data.dataset import load_stats
from cyclegan.data.sensors import LANDSAT, SENTINEL
from cyclegan.model.degradation import LANDSAT_MTF, downsample_mtf, fit_gain_offset, shrink_mask
from cyclegan.model.lpips import MaskedLPIPSLoss

logger = logging.getLogger(__name__)

GAN_TYPES = ("lsgan", "relativistic")
RECONSTRUCTIONS = ("huber", "l1")
EPS = 1e-6                                       # avoids divisions by zero


# =============================================================================
# Discriminator scores
# =============================================================================
@dataclass
class Scores:
    """Score grids of one discriminator on real and fake images, with the grids of usable scores."""

    real: list[torch.Tensor]                     # one b 1 h' w' grid per scale
    fake: list[torch.Tensor]
    real_masks: list[torch.Tensor]               # True = the score only covers good pixels
    fake_masks: list[torch.Tensor]


# =============================================================================
# Masked helpers
# =============================================================================
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of values over good pixels; mask b 1 h w broadcasts over the bands."""
    weight = mask.to(values.dtype).expand_as(values)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def mean_over_scales(losses: list[torch.Tensor]) -> torch.Tensor:
    """Average of one loss per discriminator scale."""
    return torch.stack(losses).mean()


def spectral_angle(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Angle in radians between the band vectors of each pixel; b c h w -> b 1 h w."""
    cosine = F.cosine_similarity(pred, ref, dim=1, eps=EPS).unsqueeze(1)
    return torch.acos(cosine.clamp(-1.0 + EPS, 1.0 - EPS))


# =============================================================================
# Loss
# =============================================================================
class CycleLoss(nn.Module):
    """Every loss of the cyclegan; generator() for the g step, discriminator() for the d step."""

    def __init__(
        self,
        stats_dir: str,                          # landsat.json and sentinel.json, sam works on reflectance
        lambda_gan: float = 1.0,                 # look like the other sensor
        lambda_cycle: float = 10.0,              # round trip gives the input back
        lambda_lpips: float = 1.0,               # round trip keeps the textures
        lambda_sam: float = 0.0,                 # round trip keeps the band ratios, 0 = off
        lambda_consistency_l2s: float = 1.0,     # fake sentinel seen by landsat = real landsat
        lambda_consistency_s2l: float = 0.0,     # fake landsat = real sentinel seen by landsat, 0 = off
        gan_type: str = "lsgan",                 # 'lsgan' or 'relativistic'
        reconstruction: str = "huber",           # 'huber' or 'l1', for cycle and consistency
        huber_delta: float = 0.1,                # huber switches from squared to linear here
        landsat_mtf: float = LANDSAT_MTF,        # blur of the landsat view
        factor: int = 3,                         # resolution ratio
    ):
        """Store the weights and build the frozen lpips network if it is used."""
        super().__init__()
        assert gan_type in GAN_TYPES, f"gan_type must be one of {GAN_TYPES}"
        assert reconstruction in RECONSTRUCTIONS, f"reconstruction must be one of {RECONSTRUCTIONS}"
        self.weights = {
            "gan": lambda_gan,
            "cycle": lambda_cycle,
            "lpips": lambda_lpips,
            "sam": lambda_sam,
            "consistency_l2s": lambda_consistency_l2s,
            "consistency_s2l": lambda_consistency_s2l,
        }
        self.gan_type = gan_type
        self.reconstruction = reconstruction
        self.huber_delta = huber_delta
        self.landsat_mtf = landsat_mtf
        self.factor = factor

        # Normalization stats, to go back to reflectance for sam
        for sensor in (LANDSAT, SENTINEL):
            loc, scale = load_stats(stats_dir, sensor)
            self.register_buffer(f"{sensor.name}_loc", loc.unsqueeze(0))          # 1 6 1 1
            self.register_buffer(f"{sensor.name}_scale", scale.unsqueeze(0))

        # Frozen vgg, no imagenet normalization (as in tamrfsits)
        self.lpips = None
        if lambda_lpips > 0:
            self.lpips = MaskedLPIPSLoss(LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])).requires_grad_(False).eval()

        active = ", ".join(f"{k}={v:g}" for k, v in self.weights.items() if v > 0)
        logger.info("CycleLoss: %s | gan %s | %s (delta %g) | landsat mtf %g",
                    active, gan_type, reconstruction, huber_delta, landsat_mtf)

    def train(self, mode: bool = True) -> "CycleLoss":
        """Keep the lpips network in eval mode."""
        super().train(mode)
        if self.lpips is not None:
            self.lpips.eval()
        return self

    # -------------------------------------------------------------------------
    # Single terms
    # -------------------------------------------------------------------------
    def pixel_loss(self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Huber or l1 per pixel and band, averaged over good pixels."""
        if self.reconstruction == "huber":
            values = F.huber_loss(pred, ref, delta=self.huber_delta, reduction="none")
        else:
            values = (pred - ref).abs()
        return masked_mean(values, mask)

    def lpips_loss(self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        """Masked lpips, every band as its own grayscale image copied to 3 channels; None if nothing is valid."""
        bands = pred.shape[1]
        pred = repeat(rearrange(pred, "b c h w -> (b c) h w"), "n h w -> n 3 h w")
        ref = repeat(rearrange(ref, "b c h w -> (b c) h w"), "n h w -> n 3 h w")
        mask = rearrange(repeat(mask, "b 1 h w -> b c h w", c=bands), "b c h w -> (b c) h w")
        return self.lpips(pred, ref, mask)

    def sam_loss(self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, sensor: str) -> torch.Tensor:
        """Mean spectral angle over good pixels, on reflectance (the normalized values have no physical angle)."""
        loc, scale = getattr(self, f"{sensor}_loc"), getattr(self, f"{sensor}_scale")
        return masked_mean(spectral_angle(pred * scale + loc, ref * scale + loc), mask)

    def consistency_l2s(self, fake_s: torch.Tensor, landsat: torch.Tensor, mask_l: torch.Tensor) -> torch.Tensor:
        """Fake sentinel seen by landsat vs real landsat, after a per-band gain and offset."""
        landsat_view = downsample_mtf(fake_s, self.factor, self.landsat_mtf)
        reference = fit_gain_offset(landsat, landsat_view, mask_l)         # landsat in fake-sentinel colours
        return self.pixel_loss(landsat_view, reference, mask_l)

    def consistency_s2l(self, fake_l: torch.Tensor, sentinel: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
        """Fake landsat vs real sentinel seen by landsat, after a per-band gain and offset."""
        landsat_view = downsample_mtf(sentinel, self.factor, self.landsat_mtf)
        mask = shrink_mask(mask_s, self.factor)
        reference = fit_gain_offset(landsat_view, fake_l, mask)            # sentinel view in fake-landsat colours
        return self.pixel_loss(fake_l, reference, mask)

    # -------------------------------------------------------------------------
    # GAN terms
    # -------------------------------------------------------------------------
    def gan_generator(self, scores: Scores) -> torch.Tensor:
        """Fakes should be scored as real."""
        losses = []
        for real, fake, real_mask, fake_mask in zip(scores.real, scores.fake, scores.real_masks, scores.fake_masks):
            if self.gan_type == "lsgan":
                losses.append(masked_mean((fake - 1) ** 2, fake_mask))
            else:                                                          # relativistic: swap the roles
                real_mean, fake_mean = masked_mean(real, real_mask), masked_mean(fake, fake_mask)
                losses.append(masked_mean((real - fake_mean + 1) ** 2, real_mask)
                              + masked_mean((fake - real_mean - 1) ** 2, fake_mask))
        return mean_over_scales(losses)

    def gan_discriminator(self, scores: Scores) -> torch.Tensor:
        """Reals should be scored as real, fakes as fake."""
        losses = []
        for real, fake, real_mask, fake_mask in zip(scores.real, scores.fake, scores.real_masks, scores.fake_masks):
            if self.gan_type == "lsgan":
                losses.append(0.5 * (masked_mean((real - 1) ** 2, real_mask) + masked_mean(fake ** 2, fake_mask)))
            else:                                                          # relativistic: compared with the other mean
                real_mean, fake_mean = masked_mean(real, real_mask), masked_mean(fake, fake_mask)
                losses.append(0.5 * (masked_mean((real - fake_mean - 1) ** 2, real_mask)
                                     + masked_mean((fake - real_mean + 1) ** 2, fake_mask)))
        return mean_over_scales(losses)

    # -------------------------------------------------------------------------
    # Totals
    # -------------------------------------------------------------------------
    def discriminator(self, scores_s: Scores, scores_l: Scores) -> tuple[torch.Tensor, dict[str, float]]:
        """D step: both discriminators."""
        loss_s, loss_l = self.gan_discriminator(scores_s), self.gan_discriminator(scores_l)
        total = loss_s + loss_l
        return total, {"d_sentinel": loss_s.item(), "d_landsat": loss_l.item(), "d_total": total.item()}

    def generator(
        self,
        batch: Batch,                            # real crops and masks
        fake_s: torch.Tensor,                    # g_l2s(landsat)
        fake_l: torch.Tensor,                    # g_s2l(sentinel)
        rec_l: torch.Tensor,                     # g_s2l(fake_s)
        rec_s: torch.Tensor,                     # g_l2s(fake_l)
        scores_s: Scores,                        # sentinel discriminator, fakes not detached
        scores_l: Scores,                        # landsat discriminator
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """G step: weighted sum of every active term, and each term for the logs."""
        mask_l, mask_s = batch.landsat_mask, batch.sentinel_mask
        terms = {"gan": self.gan_generator(scores_s) + self.gan_generator(scores_l),
                 "cycle": self.pixel_loss(rec_l, batch.landsat, mask_l) + self.pixel_loss(rec_s, batch.sentinel, mask_s)}

        if self.weights["lpips"] > 0:
            for name, pred, ref, mask in (("lpips_l", rec_l, batch.landsat, mask_l),
                                          ("lpips_s", rec_s, batch.sentinel, mask_s)):
                value = self.lpips_loss(pred, ref, mask)
                if value is not None:                                      # no valid feature: term skipped
                    terms["lpips"] = terms.get("lpips", 0.0) + value
        if self.weights["sam"] > 0:
            terms["sam"] = (self.sam_loss(rec_l, batch.landsat, mask_l, LANDSAT.name)
                            + self.sam_loss(rec_s, batch.sentinel, mask_s, SENTINEL.name))
        if self.weights["consistency_l2s"] > 0:
            terms["consistency_l2s"] = self.consistency_l2s(fake_s, batch.landsat, mask_l)
        if self.weights["consistency_s2l"] > 0:
            terms["consistency_s2l"] = self.consistency_s2l(fake_l, batch.sentinel, mask_s)

        total = sum(self.weights[name] * value for name, value in terms.items())
        logs = {f"g_{name}": float(value) for name, value in terms.items()}
        logs["g_total"] = total.item()
        return total, logs
