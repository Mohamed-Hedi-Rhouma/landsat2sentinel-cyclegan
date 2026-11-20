#!/usr/bin/env python
"""Figures after each validation: the same clean val pairs every time, model beside bicubic and the real target."""

import logging                                   # structured logging
from datetime import date as Date                # day gaps and target months
from pathlib import Path                         # figure files

import matplotlib                                # figures
matplotlib.use("Agg")                            # no display needed
import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # image arrays
import pytorch_lightning as pl                   # callback base class
import torch                                     # tensors
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F             # bicubic baseline

from cyclegan.data.datamodule import collate_samples
from cyclegan.data.sensors import BANDS, LANDSAT, RATIO, SENTINEL
from cyclegan.model.degradation import expand_mask, shrink_mask

logger = logging.getLogger(__name__)

COMPOSITES = {                                   # name -> bands shown as red, green, blue
    "rgb": ("red", "green", "blue"),
    "nir": ("nir", "red", "green"),
    "swir": ("swir1", "nir", "red"),
}
PERCENTILES = (2, 98)                            # stretch limits, from the real target only
MASK_COLOUR = (1.0, 0.0, 1.0)                    # magenta = bad pixel (cloud, shadow, no data)
MONTHS = (1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12)  # target month of each shown aoi, so seasons are spread
YEAR = 2022                                      # year of the data, for the target dates

# what a "clean" comparison is
MAX_GAP_DAYS = 1                                 # almost no real change on the ground
MIN_CLEAR = 0.98                                 # share of good pixels in both crops
MAX_SNOW = 0.02                                  # share of snow pixels in the real sentinel
NDSI_SNOW = 0.4                                  # snow: ndsi above this ...
NIR_SNOW = 0.11                                  # ... and nir reflectance above this
CANDIDATES_PER_AOI = 12                          # pairs read per aoi to find a clean one


# =============================================================================
# Image helpers
# =============================================================================
def composite(x: np.ndarray, bands: tuple[str, ...]) -> np.ndarray:
    """6 h w reflectance -> h w 3 image of three bands."""
    return np.stack([x[BANDS.index(band)] for band in bands], axis=-1)


def stretch_limits(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel 2 and 98 percentiles of the good pixels of an h w 3 image."""
    good = image[mask]
    if len(good) == 0:                                              # fully masked, unlikely
        good = image.reshape(-1, 3)
    low, high = np.percentile(good, PERCENTILES, axis=0)
    return low, np.maximum(high, low + 1e-6)


def to_display(image: np.ndarray, mask: np.ndarray, limits: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Stretch with fixed limits, clip to 0-1, paint bad pixels magenta."""
    low, high = limits
    shown = np.clip((image - low) / (high - low), 0.0, 1.0)
    shown[~mask] = MASK_COLOUR
    return shown


def day_gap(date_a: str, date_b: str) -> int:
    return abs((Date.fromisoformat(date_a) - Date.fromisoformat(date_b)).days)


def snow_fraction(sentinel: np.ndarray, mask: np.ndarray) -> float:
    """Share of good pixels that look like snow: bright nir and high ndsi (green vs swir1)."""
    green, nir, swir1 = (sentinel[BANDS.index(band)] for band in ("green", "nir", "swir1"))
    ndsi = (green - swir1) / np.maximum(green + swir1, 1e-6)
    snow = (ndsi > NDSI_SNOW) & (nir > NIR_SNOW) & mask
    return float(snow.sum() / max(mask.sum(), 1))


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Error map (mean over bands, nan where masked) and its mean."""
    error = np.abs(a - b).mean(axis=0)
    return np.where(mask, error, np.nan), float(error[mask].mean()) if mask.any() else float("nan")


def centre(x: np.ndarray, fraction: int = 3) -> np.ndarray:
    """Middle 1/fraction of an image (last two axes), for the zoom row."""
    h, w = x.shape[-2:]
    top, left = (h - h // fraction) // 2, (w - w // fraction) // 2
    return x[..., top:top + h // fraction, left:left + w // fraction]


# =============================================================================
# Callback
# =============================================================================
class TranslationFigures(pl.Callback):
    """After each validation, super-resolution figures and a smaller set for the other direction.

    Super-resolution, two rows per pair (full crop, then its centre zoomed x3):
        landsat input | bicubic x3 | model | real sentinel | |model - real|
    Other direction, one row per pair:
        real sentinel | fake landsat | real landsat | |fake - real|
    """

    def __init__(
        self,
        output_dir: str,                         # figures/step_<step>_<composite>_<part>.png
        nb_pairs: int = 16,                      # val pairs shown, from different aois
        pairs_per_figure: int = 4,               # pairs in one png, so 16 pairs = 4 pngs per composite
        every_n_validations: int = 1,            # draw every n validations
        error_max: float = 0.05,                 # reflectance error shown as the brightest colour
        dpi: int = 120,                          # figure resolution
    ):
        """Store the settings; the pairs and their colour limits are chosen at the first validation."""
        super().__init__()
        self.output_dir = Path(output_dir)
        self.nb_pairs = nb_pairs
        self.pairs_per_figure = pairs_per_figure
        self.every_n_validations = every_n_validations
        self.error_max = error_max
        self.dpi = dpi
        self.indices: list[int] | None = None    # val pairs, fixed for the whole run
        self.limits: dict = {}                   # (pair, direction, composite) -> fixed stretch
        self.nb_validations = 0

    # -------------------------------------------------------------------------
    # Pairs
    # -------------------------------------------------------------------------
    def is_clean(self, dataset, pl_module, index: int) -> bool:
        """Both crops almost fully clear and no snow in the real sentinel."""
        sample = dataset[index]
        if sample.landsat_mask.float().mean() < MIN_CLEAR or sample.sentinel_mask.float().mean() < MIN_CLEAR:
            return False
        sentinel = pl_module.to_reflectance(sample.sentinel[None].to(pl_module.device), SENTINEL.name)[0].cpu().numpy()
        return snow_fraction(sentinel, sample.sentinel_mask[0].cpu().numpy().astype(bool)) <= MAX_SNOW

    def choose_pairs(self, dataset, pl_module) -> list[int]:
        """One clean pair per aoi, spread over the val aois, each aoi near its own target month."""
        by_aoi: dict[str, list[int]] = {}
        for index, (landsat_crop, sentinel_crop) in enumerate(dataset.pairs):
            if day_gap(landsat_crop.date, sentinel_crop.date) <= MAX_GAP_DAYS:
                by_aoi.setdefault(landsat_crop.aoi, []).append(index)
        aois = sorted(by_aoi)
        picks = np.linspace(0, len(aois) - 1, min(self.nb_pairs, len(aois))).round().astype(int) if aois else []

        indices = []
        for k, aoi_index in enumerate(picks):
            target = Date(YEAR, MONTHS[k % len(MONTHS)], 15)
            candidates = sorted(by_aoi[aois[aoi_index]],
                                key=lambda i: abs((Date.fromisoformat(dataset.pairs[i][0].date) - target).days))
            for index in candidates[:CANDIDATES_PER_AOI]:
                if self.is_clean(dataset, pl_module, index):
                    indices.append(index)
                    break
        logger.info("figures: %d clean val pairs %s", len(indices), [dataset.pairs[i][0].info() for i in indices])
        return indices

    # -------------------------------------------------------------------------
    # Hook
    # -------------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Translate the fixed pairs, draw the figures, log model vs bicubic on them."""
        if trainer.sanity_checking:
            return
        self.nb_validations += 1
        if self.nb_validations % self.every_n_validations != 0:
            return
        dataset = trainer.datamodule.val_set
        if self.indices is None:
            self.indices = self.choose_pairs(dataset, pl_module)
        if not self.indices:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        epoch, step = trainer.current_epoch, trainer.global_step       # several validations per epoch: name by step
        model_maes, bicubic_maes = [], []
        for start in range(0, len(self.indices), self.pairs_per_figure):     # one group of pairs per png
            part = start // self.pairs_per_figure + 1
            pairs = self.translate(dataset, pl_module, self.indices[start:start + self.pairs_per_figure])
            model_maes += [p["l2s"]["mae"] for p in pairs]
            bicubic_maes += [p["l2s"]["bicubic_mae"] for p in pairs]
            for name, bands in COMPOSITES.items():
                self.save(trainer, self.draw_l2s(pairs, name, bands, epoch, step, part), f"{name}_{part}", step)
            self.save(trainer, self.draw_s2l(pairs, epoch, step, part), f"s2l_rgb_{part}", step)
        logger.info("figures: %d pairs | model mae %.4f | bicubic mae %.4f", len(model_maes),
                    np.nanmean(model_maes), np.nanmean(bicubic_maes))

    def save(self, trainer: pl.Trainer, figure: plt.Figure, name: str, step: int) -> None:
        figure.savefig(self.output_dir / f"step_{step:07d}_{name}.png", dpi=self.dpi, bbox_inches="tight")
        for experiment_logger in trainer.loggers:
            if isinstance(experiment_logger, TensorBoardLogger):
                experiment_logger.experiment.add_figure(f"figures/{name}", figure, step)
        plt.close(figure)

    # -------------------------------------------------------------------------
    # Translate
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def translate(self, dataset, pl_module, indices: list[int]) -> list[dict]:
        """For each pair: both directions in reflectance, plus the bicubic baseline, as numpy."""
        batch = collate_samples([dataset[i] for i in indices]).to(pl_module.device)
        landsat = pl_module.to_reflectance(batch.landsat, LANDSAT.name)
        sentinel = pl_module.to_reflectance(batch.sentinel, SENTINEL.name)
        fake_s = pl_module.to_reflectance(pl_module.g_l2s(batch.landsat), SENTINEL.name)
        fake_l = pl_module.to_reflectance(pl_module.g_s2l(batch.sentinel), LANDSAT.name)
        bicubic = F.interpolate(landsat, scale_factor=RATIO, mode="bicubic", align_corners=False)

        def numpy(x):
            return x.cpu().numpy()

        def mask(m):
            return m[:, 0].cpu().numpy().astype(bool)

        landsat_mask, sentinel_mask = mask(batch.landsat_mask), mask(batch.sentinel_mask)
        up_mask, down_mask = mask(expand_mask(batch.landsat_mask)), mask(shrink_mask(batch.sentinel_mask))
        landsat, sentinel, fake_s, fake_l, bicubic = map(numpy, (landsat, sentinel, fake_s, fake_l, bicubic))

        pairs = []
        for k, index in enumerate(indices):
            both_up = up_mask[k] & sentinel_mask[k]
            both_down = down_mask[k] & landsat_mask[k]
            error, mae = masked_mae(fake_s[k], sentinel[k], both_up)
            _, bicubic_mae = masked_mae(bicubic[k], sentinel[k], both_up)
            s2l_error, s2l_mae = masked_mae(fake_l[k], landsat[k], both_down)
            pairs.append(dict(
                index=index, landsat_info=batch.landsat_info[k], sentinel_info=batch.sentinel_info[k],
                l2s=dict(source=landsat[k], source_mask=landsat_mask[k], bicubic=bicubic[k], fake=fake_s[k],
                         fake_mask=up_mask[k], real=sentinel[k], real_mask=sentinel_mask[k],
                         error=error, mae=mae, bicubic_mae=bicubic_mae),
                s2l=dict(source=sentinel[k], source_mask=sentinel_mask[k], fake=fake_l[k], fake_mask=down_mask[k],
                         real=landsat[k], real_mask=landsat_mask[k], error=s2l_error, mae=s2l_mae),
            ))
        return pairs

    # -------------------------------------------------------------------------
    # Draw
    # -------------------------------------------------------------------------
    def pair_limits(self, pair: dict, direction: str, name: str, bands: tuple[str, ...]):
        """Stretch of one pair, from its real target, computed once and kept for the whole run."""
        key = (pair["index"], direction, name)
        if key not in self.limits:
            d = pair[direction]
            self.limits[key] = stretch_limits(composite(d["real"], bands), d["real_mask"])
        return self.limits[key]

    def error_panel(self, figure, ax, error: np.ndarray, title: str) -> None:
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(MASK_COLOUR)                                           # nan = bad pixel
        shown = ax.imshow(error, cmap=cmap, vmin=0.0, vmax=self.error_max, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        figure.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)

    def draw_l2s(self, pairs: list[dict], name: str, bands: tuple[str, ...], epoch: int, step: int,
                 part: int) -> plt.Figure:
        """Super-resolution: per pair a full row and a zoom row (centre third, x3)."""
        figure, axes = plt.subplots(2 * len(pairs), 5, figsize=(20, 8 * len(pairs)), squeeze=False)
        for k, pair in enumerate(pairs):
            d = pair["l2s"]
            limits = self.pair_limits(pair, "l2s", name, bands)
            gap = day_gap(pair["landsat_info"].split()[1], pair["sentinel_info"].split()[1])
            images = (
                (d["source"], d["source_mask"], f"landsat 30 m (input)\n{pair['landsat_info']}"),
                (d["bicubic"], d["fake_mask"], f"bicubic x3 (baseline)\nmae {d['bicubic_mae']:.4f}"),
                (d["fake"], d["fake_mask"], f"model 10 m\nmae {d['mae']:.4f}"),
                (d["real"], d["real_mask"], f"real sentinel 10 m\n{pair['sentinel_info']}  ({gap} days)"),
            )
            for zoom, axes_row in enumerate(axes[2 * k:2 * k + 2]):
                for ax, (image, mask, title) in zip(axes_row[:4], images):
                    shown = to_display(composite(image, bands), mask, limits)
                    ax.imshow(centre(shown.transpose(2, 0, 1)).transpose(1, 2, 0) if zoom else shown,
                              interpolation="nearest")
                    ax.set_title(f"zoom x3 | {title.splitlines()[0]}" if zoom else title, fontsize=9)
                error = centre(d["error"]) if zoom else d["error"]
                self.error_panel(figure, axes_row[4], error, "zoom x3 | |model - real|" if zoom
                                 else "|model - real|, mean of bands")
                for ax in axes_row:
                    ax.axis("off")
        figure.suptitle(f"super-resolution | epoch {epoch} | step {step} | {name}: {' '.join(bands)} | part {part}"
                        f" | magenta = masked | same colour scale per pair", fontsize=13)
        figure.tight_layout()
        return figure

    def draw_s2l(self, pairs: list[dict], epoch: int, step: int, part: int) -> plt.Figure:
        """Other direction, rgb only: sentinel input | fake landsat | real landsat | error."""
        bands = COMPOSITES["rgb"]
        figure, axes = plt.subplots(len(pairs), 4, figsize=(16, 4 * len(pairs)), squeeze=False)
        for axes_row, pair in zip(axes, pairs):
            d = pair["s2l"]
            limits = self.pair_limits(pair, "s2l", "rgb", bands)
            images = (
                (d["source"], d["source_mask"], f"sentinel 10 m (input)\n{pair['sentinel_info']}"),
                (d["fake"], d["fake_mask"], f"fake landsat 30 m\nmae {d['mae']:.4f}"),
                (d["real"], d["real_mask"], f"real landsat 30 m\n{pair['landsat_info']}"),
            )
            for ax, (image, mask, title) in zip(axes_row[:3], images):
                ax.imshow(to_display(composite(image, bands), mask, limits), interpolation="nearest")
                ax.set_title(title, fontsize=9)
            self.error_panel(figure, axes_row[3], d["error"], "|fake - real|, mean of bands")
            for ax in axes_row:
                ax.axis("off")
        figure.suptitle(f"sentinel -> landsat | epoch {epoch} | step {step} | rgb | part {part}", fontsize=13)
        figure.tight_layout()
        return figure