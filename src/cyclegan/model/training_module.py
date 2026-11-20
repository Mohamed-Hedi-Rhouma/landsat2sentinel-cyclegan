#!/usr/bin/env python
"""Lightning module of the cyclegan: one d step then one g step per batch, paired metrics in val and test."""

import logging                                   # structured logging
from dataclasses import dataclass                # optimization settings

import pytorch_lightning as pl                   # module base class
import torch                                     # tensors
from torch import nn                             # module type

from cyclegan.data.datamodule import Batch
from cyclegan.data.sensors import BANDS, LANDSAT, SENTINEL
from cyclegan.model.degradation import expand_mask, fit_gain_offset, shrink_mask
from cyclegan.model.discriminator import score_masks
from cyclegan.model.image_pool import ImagePool
from cyclegan.model.loss import CycleLoss, Scores, masked_mean

logger = logging.getLogger(__name__)

SUMMARY = (                                      # (logged key, short name) printed after each validation
    ("train/g", "g"),                            # last train step
    ("train/d", "d"),
    ("val/l2s_mae", "l2s mae"),                  # fake sentinel vs real sentinel, raw reflectance
    ("val/l2s_mae_fit", "l2s mae_fit"),          # same after gain/offset fit: structure only
    ("val/l2s_lpips", "l2s lpips"),              # checkpoint metric
    ("val/s2l_mae", "s2l mae"),                  # fake landsat vs real landsat
    ("val/cycle_l", "cycle l"),                  # round trips
    ("val/cycle_s", "cycle s"),
)


@dataclass(frozen=True)
class OptimizationParameters:
    """Adam settings and the learning-rate schedule: constant, then linear decay to 0."""

    lr_g: float = 1e-4                           # generators
    lr_d: float = 1e-4                           # discriminators
    beta1: float = 0.5                           # adam, cyclegan standard
    beta2: float = 0.999
    decay_start: float = 0.5                     # fraction of max_epochs after which the lr decays linearly
    grad_clip: float = 1.0                       # max gradient norm, 0 = off


class CycleGANModule(pl.LightningModule):
    """Two generators, two discriminators, one loss; manual optimization because d and g alternate."""

    def __init__(
        self,
        g_l2s: nn.Module,                        # landsat -> sentinel
        g_s2l: nn.Module,                        # sentinel -> landsat
        d_sentinel: nn.Module,                   # real or fake sentinel
        d_landsat: nn.Module,                    # real or fake landsat
        loss: CycleLoss,                         # every loss term
        optimization: OptimizationParameters,    # optimizers and schedule
        pool_size: int = 50,                     # past fakes shown to the discriminators, 0 = off
        factor: int = 3,                         # resolution ratio
    ):
        """Store the networks; the pools live outside the state dict."""
        super().__init__()
        self.automatic_optimization = False
        self.g_l2s, self.g_s2l = g_l2s, g_s2l
        self.d_sentinel, self.d_landsat = d_sentinel, d_landsat
        self.loss = loss
        self.optimization = optimization
        self.factor = factor
        self.pool_s, self.pool_l = ImagePool(pool_size), ImagePool(pool_size)
        self.nb_skipped = 0                                          # steps dropped for a non-finite loss

    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        """One adam for both generators, one for both discriminators, same linear-decay schedule."""
        op = self.optimization
        betas = (op.beta1, op.beta2)
        opt_g = torch.optim.Adam(list(self.g_l2s.parameters()) + list(self.g_s2l.parameters()), lr=op.lr_g, betas=betas)
        opt_d = torch.optim.Adam(list(self.d_sentinel.parameters()) + list(self.d_landsat.parameters()),
                                 lr=op.lr_d, betas=betas)
        max_epochs = self.trainer.max_epochs
        start = int(op.decay_start * max_epochs)

        def factor(epoch: int) -> float:
            """1 until start, then down to 0 at max_epochs."""
            return 1.0 if epoch < start else max(0.0, 1.0 - (epoch - start) / max(1, max_epochs - start))

        schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, factor) for opt in (opt_g, opt_d)]
        logger.info("adam g %.1e d %.1e betas %s | linear decay from epoch %d of %d | clip %g",
                    op.lr_g, op.lr_d, betas, start, max_epochs, op.grad_clip)
        return [opt_g, opt_d], schedulers

    def on_train_epoch_end(self):
        """Manual optimization: the schedulers are stepped here, once per epoch."""
        if self.trainer.should_stop:                                 # stopped mid-epoch (time limit): epoch not done
            return
        for scheduler in self.lr_schedulers():
            scheduler.step()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def fakes(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Both translations and the masks of their good pixels."""
        fake_s = self.g_l2s(batch.landsat)
        fake_l = self.g_s2l(batch.sentinel)
        mask_fake_s = expand_mask(batch.landsat_mask, self.factor)    # good if its landsat pixel was good
        mask_fake_l = shrink_mask(batch.sentinel_mask, self.factor)   # good if its 9 sentinel pixels were good
        return fake_s, fake_l, mask_fake_s, mask_fake_l

    @staticmethod
    def scores(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor,
               real_mask: torch.Tensor, fake_mask: torch.Tensor, real_grad: bool) -> Scores:
        """Discriminator scores on real and fake, and their usable grids; real_grad False in the g step."""
        with torch.set_grad_enabled(real_grad):
            real_scores = discriminator(real)
        fake_scores = discriminator(fake)
        return Scores(real_scores, fake_scores, score_masks(real_mask, real_scores), score_masks(fake_mask, fake_scores))

    def step(self, optimizer, loss: torch.Tensor) -> bool:
        """Backward, clip, update; False (nothing done) if the loss is not finite."""
        if not torch.isfinite(loss):
            self.nb_skipped += 1
            logger.warning("step %d: non-finite loss, update skipped (%d so far)", self.global_step, self.nb_skipped)
            return False
        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.optimization.grad_clip > 0:                       # lightning handles mixed precision here
            self.clip_gradients(optimizer, gradient_clip_val=self.optimization.grad_clip,
                                gradient_clip_algorithm="norm")
        optimizer.step()
        return True

    @staticmethod
    def freeze(networks: list[nn.Module], frozen: bool) -> None:
        """Stop or restart the gradients of some networks."""
        for net in networks:
            net.requires_grad_(not frozen)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def training_step(self, batch: Batch, _batch_idx: int):
        """D step on pooled fakes, then g step with the d's frozen."""
        opt_g, opt_d = self.optimizers()
        discriminators = [self.d_sentinel, self.d_landsat]
        fake_s, fake_l, mask_fake_s, mask_fake_l = self.fakes(batch)

        # D step: real vs past and present fakes, generators untouched
        pooled_s, pooled_mask_s = self.pool_s.query(fake_s.detach(), mask_fake_s)
        pooled_l, pooled_mask_l = self.pool_l.query(fake_l.detach(), mask_fake_l)
        scores_s = self.scores(self.d_sentinel, batch.sentinel, pooled_s, batch.sentinel_mask, pooled_mask_s, True)
        scores_l = self.scores(self.d_landsat, batch.landsat, pooled_l, batch.landsat_mask, pooled_mask_l, True)
        d_loss, d_logs = self.loss.discriminator(scores_s, scores_l)
        self.step(opt_d, d_loss)

        # G step: fool the frozen d's, keep the round trip and the landsat view
        self.freeze(discriminators, True)
        rec_l, rec_s = self.g_s2l(fake_s), self.g_l2s(fake_l)
        scores_s = self.scores(self.d_sentinel, batch.sentinel, fake_s, batch.sentinel_mask, mask_fake_s, False)
        scores_l = self.scores(self.d_landsat, batch.landsat, fake_l, batch.landsat_mask, mask_fake_l, False)
        g_loss, g_logs = self.loss.generator(batch, fake_s, fake_l, rec_l, rec_s, scores_s, scores_l)
        self.step(opt_g, g_loss)
        self.freeze(discriminators, False)

        logs = {f"train/{k}": v for k, v in {**d_logs, **g_logs}.items()}
        logs["train/skipped"] = float(self.nb_skipped)
        self.log_dict(logs, batch_size=batch.landsat.shape[0], prog_bar=False)
        self.log("train/g", g_logs["g_total"], prog_bar=True, batch_size=batch.landsat.shape[0])
        self.log("train/d", d_logs["d_total"], prog_bar=True, batch_size=batch.landsat.shape[0])

    # -------------------------------------------------------------------------
    # Validation and test: paired crops of the same place, at most a few days apart
    # -------------------------------------------------------------------------
    def to_reflectance(self, x: torch.Tensor, sensor_name: str) -> torch.Tensor:
        """Normalized values back to surface reflectance."""
        return x * getattr(self.loss, f"{sensor_name}_scale") + getattr(self.loss, f"{sensor_name}_loc")

    def paired_metrics(self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor,
                       sensor_name: str, prefix: str) -> dict[str, torch.Tensor]:
        """mae (radiometry), mae after gain and offset (structure), per-band mae, lpips and sam."""
        pred_r, ref_r = self.to_reflectance(pred, sensor_name), self.to_reflectance(ref, sensor_name)
        error = (pred_r - ref_r).abs()
        metrics = {
            f"{prefix}_mae": masked_mean(error, mask),
            f"{prefix}_mae_fit": masked_mean((pred_r - fit_gain_offset(ref_r, pred_r, mask)).abs(), mask),
            f"{prefix}_sam": self.loss.sam_loss(pred, ref, mask, sensor_name),
        }
        for k, band in enumerate(BANDS):
            metrics[f"{prefix}_mae_{band}"] = masked_mean(error[:, k:k + 1], mask)
        if self.loss.lpips is not None:
            value = self.loss.lpips_loss(pred, ref, mask)
            if value is not None:
                metrics[f"{prefix}_lpips"] = value
        return metrics

    def evaluate(self, batch: Batch, stage: str) -> None:
        """Translate both ways, compare with the real image of the other sensor and with the round trip."""
        fake_s, fake_l, mask_fake_s, mask_fake_l = self.fakes(batch)
        rec_l, rec_s = self.g_s2l(fake_s), self.g_l2s(fake_l)
        mask_s = mask_fake_s & batch.sentinel_mask                     # good in both images of the pair
        mask_l = mask_fake_l & batch.landsat_mask

        metrics = {
            **self.paired_metrics(fake_s, batch.sentinel, mask_s, SENTINEL.name, "l2s"),
            **self.paired_metrics(fake_l, batch.landsat, mask_l, LANDSAT.name, "s2l"),
            "cycle_l": self.loss.pixel_loss(rec_l, batch.landsat, batch.landsat_mask),
            "cycle_s": self.loss.pixel_loss(rec_s, batch.sentinel, batch.sentinel_mask),
        }
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()},
                      batch_size=batch.landsat.shape[0], on_epoch=True, sync_dist=True)

    def validation_step(self, batch: Batch, _batch_idx: int):
        """Paired metrics on the val tiles."""
        self.evaluate(batch, "val")

    def test_step(self, batch: Batch, _batch_idx: int):
        """Paired metrics on the test tiles."""
        self.evaluate(batch, "test")

    def on_validation_end(self):
        """One summary line per epoch in the log, after the val metrics are averaged."""
        if self.trainer.sanity_checking:
            return
        metrics = self.trainer.callback_metrics
        parts = [f"{name} {float(metrics[key]):.4f}" for key, name in SUMMARY if key in metrics]
        logger.info("epoch %3d | %s", self.current_epoch, " | ".join(parts))