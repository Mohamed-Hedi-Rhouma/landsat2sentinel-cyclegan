#!/usr/bin/env python
"""Lightning data module: train, val and test crops of both sensors, stacked into batches."""

import logging                                   # structured logging
from dataclasses import dataclass, fields        # batch container

import pytorch_lightning as pl                   # data module base class
import torch                                     # stacking
from torch.utils.data import DataLoader          # loaders

from cyclegan.data.dataset import CycleDataset, Sample

logger = logging.getLogger(__name__)


# =============================================================================
# Batch
# =============================================================================
@dataclass
class Batch:
    """Samples stacked on b."""

    landsat: torch.Tensor                        # b 6 h w
    landsat_mask: torch.Tensor                   # b 1 h w, True = good pixel
    sentinel: torch.Tensor                       # b 6 3h 3w
    sentinel_mask: torch.Tensor                  # b 1 3h 3w
    landsat_info: list[str]                      # one text per sample
    sentinel_info: list[str]

    def _apply(self, fn) -> "Batch":
        """Apply fn to every tensor, keep the texts."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return Batch(**{k: fn(v) if isinstance(v, torch.Tensor) else v for k, v in values.items()})

    def to(self, device: str | torch.device) -> "Batch":
        """Move the tensors to the device (called by lightning)."""
        return self._apply(lambda t: t.to(device, non_blocking=True))

    def pin_memory(self) -> "Batch":
        """Pinned tensors for faster gpu copies (called by the dataloader)."""
        return self._apply(lambda t: t.pin_memory())


def collate_samples(samples: list[Sample]) -> Batch:
    """Stack the tensors of every sample, list the texts."""
    return Batch(
        landsat=torch.stack([s.landsat for s in samples]),
        landsat_mask=torch.stack([s.landsat_mask for s in samples]),
        sentinel=torch.stack([s.sentinel for s in samples]),
        sentinel_mask=torch.stack([s.sentinel_mask for s in samples]),
        landsat_info=[s.landsat_info for s in samples],
        sentinel_info=[s.sentinel_info for s in samples],
    )


# =============================================================================
# Data module
# =============================================================================
class CycleDataModule(pl.LightningDataModule):
    """Train on the train aois, validate on the val tiles of the test aois, test on the other test aois."""

    def __init__(
        self,
        landsat_root: str,                       # landsat download root
        sentinel_root: str,                      # sentinel download root
        stats_dir: str,                          # landsat.json and sentinel.json from stats.py
        cache_dir: str,                          # crop clean fractions, must be writable
        val_tiles: list[str],                    # test tiles used for val
        min_clean: float = 0.5,                  # smallest clean fraction of a date
        min_crop_clean: float = 0.9,             # smallest clean fraction of a crop
        crop_size: int = 128,                    # landsat crop, sentinel crop is 3 times larger
        crops_per_side: int = 3,                 # crop grid per image side
        max_pair_days: int = 5,                  # val and test: largest gap between the two dates
        batch_size: int = 4,                     # samples per batch
        num_workers: int = 4,                    # loader workers
    ):
        """Store the dataset settings and the loader settings."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dict(
            landsat_root=landsat_root,
            sentinel_root=sentinel_root,
            stats_dir=stats_dir,
            cache_dir=cache_dir,
            val_tiles=list(val_tiles),
            min_clean=min_clean,
            min_crop_clean=min_crop_clean,
            crop_size=crop_size,
            crops_per_side=crops_per_side,
            max_pair_days=max_pair_days,
        )
        self.train_set: CycleDataset | None = None
        self.val_set: CycleDataset | None = None
        self.test_set: CycleDataset | None = None
        logger.info("CycleDataModule: crop %d (sentinel %d), %dx%d cells, batch %d, %d workers",
                    crop_size, 3 * crop_size, crops_per_side, crops_per_side, batch_size, num_workers)
        logger.info("  val tiles %s", list(val_tiles))

    def setup(self, stage: str | None = None):
        """Build each split once."""
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = CycleDataset(split="train", **self.dataset_kwargs)
        if stage in (None, "fit", "validate") and self.val_set is None:
            self.val_set = CycleDataset(split="val", **self.dataset_kwargs)
        if stage in (None, "test") and self.test_set is None:
            self.test_set = CycleDataset(split="test", **self.dataset_kwargs)

    def _loader(self, dataset: CycleDataset, shuffle: bool) -> DataLoader:
        """Dataloader with the shared settings."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_samples,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,     # keep workers alive between epochs
        )

    def train_dataloader(self) -> DataLoader:
        """Train loader, shuffled."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Val loader, fixed order."""
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader, fixed order."""
        return self._loader(self.test_set, shuffle=False)