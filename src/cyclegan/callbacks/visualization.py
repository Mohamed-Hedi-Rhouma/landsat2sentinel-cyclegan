#!/usr/bin/env python
"""Figures after each validation: the same val pairs every time, real target beside the fake, fixed colour scales."""

import logging                                   # structured logging
from datetime import date as Date                # day gap of a pair
from pathlib import Path                         # figure files

import matplotlib                                # figures
matplotlib.use("Agg")                            # no display needed
import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # image arrays
import pytorch_lightning as pl                   # callback base class
import torch                                     # tensors
from pytorch_lightning.loggers import TensorBoardLogger

from cyclegan.data.datamodule import collate_samples
from cyclegan.data.sensors import BANDS, LANDSAT, SENTINEL
from cyclegan.model.degradation import expand_mask, shrink_mask

logger = logging.getLogger(__name__)

COMPOSITES = {                                   # name -> bands shown as red, green, blue
    "rgb": ("red", "green", "blue"),
    "nir": ("nir", "red", "green"),
    "swir": ("swir1", "nir", "red"),
}
PERCENTILES = (2, 98)                            # stretch limits, from the real target only
MASK_COLOUR = (1.0, 0.0, 1.0)                    # magenta = bad pixel (cloud, shadow, no data)


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


def day_gap(info_a: str, info_b: str) -> int:
    """Days between the dates of two crop texts 'aoi yyyy-mm-dd (i,j)'."""
    return abs((Date.fromisoformat(info_a.split()[1]) - Date.fromisoformat(info_b.split()[1])).days)


# =============================================================================
# Callback
# =============================================================================
class TranslationFigures(pl.Callback):
    """After each validation, one figure per composite: rows = val pairs and directions.

    Row l2s: real landsat | fake sentinel | real sentinel | error map
    Row s2l: real sentinel | fake landsat | real landsat  | error map
    """

    def __init__(
        self,
        output_dir: str,                         # figures/epoch_<epoch>_<composite>_<part>.png
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
    def choose_pairs(self, dataset) -> list[int]:
        """First pair of nb_pairs aois spread over the sorted val aois, so several regions are shown."""
        first_of_aoi: dict[str, int] = {}
        for index, (landsat_crop, _) in enumerate(dataset.pairs):
            first_of_aoi.setdefault(landsat_crop.aoi, index)
        aois = sorted(first_of_aoi)
        if not aois:
            return []
        picks = np.linspace(0, len(aois) - 1, min(self.nb_pairs, len(aois))).round().astype(int)
        indices = [first_of_aoi[aois[k]] for k in picks]
        logger.info("figures: val pairs %s", [dataset.pairs[i][0].info() for i in indices])
        return indices

    # -------------------------------------------------------------------------
    # Hook
    # -------------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Translate the fixed pairs and draw one figure per composite."""
        if trainer.sanity_checking:
            return
        self.nb_validations += 1
        if self.nb_validations % self.every_n_validations != 0:
            return
        dataset = trainer.datamodule.val_set
        if self.indices is None:
            self.indices = self.choose_pairs(dataset)
        if not self.indices:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch
        for start in range(0, len(self.indices), self.pairs_per_figure):     # one group of pairs per png
            part = start // self.pairs_per_figure + 1
            rows = self.translate(dataset, pl_module, self.indices[start:start + self.pairs_per_figure])
            for name, bands in COMPOSITES.items():
                figure = self.draw(rows, name, bands, epoch, part)
                figure.savefig(self.output_dir / f"epoch_{epoch:03d}_{name}_{part}.png",
                               dpi=self.dpi, bbox_inches="tight")
                for experiment_logger in trainer.loggers:
                    if isinstance(experiment_logger, TensorBoardLogger):
                        experiment_logger.experiment.add_figure(f"figures/{name}_{part}", figure, trainer.global_step)
                plt.close(figure)

    # -------------------------------------------------------------------------
    # Translate
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def translate(self, dataset, pl_module, indices: list[int]) -> list[dict]:
        """For each pair and direction: input, fake, real target and masks, in reflectance, as numpy."""
        batch = collate_samples([dataset[i] for i in indices]).to(pl_module.device)
        fake_s = pl_module.g_l2s(batch.landsat)
        fake_l = pl_module.g_s2l(batch.sentinel)
        mask_fake_s = expand_mask(batch.landsat_mask)
        mask_fake_l = shrink_mask(batch.sentinel_mask)

        def reflectance(x, sensor):
            return pl_module.to_reflectance(x, sensor.name).cpu().numpy()

        def as_mask(m):
            return m[:, 0].cpu().numpy()

        directions = {
            "l2s": dict(source=reflectance(batch.landsat, LANDSAT), source_mask=as_mask(batch.landsat_mask),
                        fake=reflectance(fake_s, SENTINEL), fake_mask=as_mask(mask_fake_s),
                        real=reflectance(batch.sentinel, SENTINEL), real_mask=as_mask(batch.sentinel_mask),
                        labels=("real landsat", "fake sentinel", "real sentinel")),
            "s2l": dict(source=reflectance(batch.sentinel, SENTINEL), source_mask=as_mask(batch.sentinel_mask),
                        fake=reflectance(fake_l, LANDSAT), fake_mask=as_mask(mask_fake_l),
                        real=reflectance(batch.landsat, LANDSAT), real_mask=as_mask(batch.landsat_mask),
                        labels=("real sentinel", "fake landsat", "real landsat")),
        }
        rows = []
        for k, index in enumerate(indices):
            for direction, d in directions.items():
                both = d["fake_mask"][k] & d["real_mask"][k]
                error = np.abs(d["fake"][k] - d["real"][k]).mean(axis=0)              # h w, mean over bands
                rows.append(dict(
                    pair=index, direction=direction, labels=d["labels"],
                    source=d["source"][k], source_mask=d["source_mask"][k],
                    fake=d["fake"][k], fake_mask=d["fake_mask"][k],
                    real=d["real"][k], real_mask=d["real_mask"][k],
                    error=np.where(both, error, np.nan), mae=float(error[both].mean()) if both.any() else float("nan"),
                    landsat_info=batch.landsat_info[k], sentinel_info=batch.sentinel_info[k],
                ))
        return rows

    # -------------------------------------------------------------------------
    # Draw
    # -------------------------------------------------------------------------
    def row_limits(self, row: dict, name: str, bands: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Stretch of one row, from its real target, computed once and kept for the whole run."""
        key = (row["pair"], row["direction"], name)
        if key not in self.limits:
            self.limits[key] = stretch_limits(composite(row["real"], bands), row["real_mask"])
        return self.limits[key]

    def draw(self, rows: list[dict], name: str, bands: tuple[str, ...], epoch: int, part: int) -> plt.Figure:
        """One figure: a row per pair and direction, four panels per row."""
        figure, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)), squeeze=False)
        error_cmap = plt.get_cmap("magma").copy()
        error_cmap.set_bad(MASK_COLOUR)                                     # nan = bad pixel
        for axes_row, row in zip(axes, rows):
            limits = self.row_limits(row, name, bands)
            gap = day_gap(row["landsat_info"], row["sentinel_info"])
            source_info, target_info = ((row["landsat_info"], row["sentinel_info"]) if row["direction"] == "l2s"
                                        else (row["sentinel_info"], row["landsat_info"]))
            panels = (
                (row["source"], row["source_mask"], f"{row['labels'][0]}\n{source_info}"),
                (row["fake"], row["fake_mask"], f"{row['labels'][1]}\nmae {row['mae']:.4f}"),
                (row["real"], row["real_mask"], f"{row['labels'][2]}\n{target_info}  ({gap} days)"),
            )
            for ax, (image, mask, title) in zip(axes_row[:3], panels):
                ax.imshow(to_display(composite(image, bands), mask, limits), interpolation="nearest")
                ax.set_title(title, fontsize=9)
            shown = axes_row[3].imshow(row["error"], cmap=error_cmap, vmin=0.0, vmax=self.error_max,
                                       interpolation="nearest")
            axes_row[3].set_title(f"|fake - real|, mean of bands ({row['direction']})", fontsize=9)
            figure.colorbar(shown, ax=axes_row[3], fraction=0.046, pad=0.04)
            for ax in axes_row:
                ax.axis("off")
        figure.suptitle(f"epoch {epoch} | {name}: {' '.join(bands)} | part {part} | "
                        f"magenta = masked | same colour scale per row", fontsize=12)
        figure.tight_layout()
        return figure
