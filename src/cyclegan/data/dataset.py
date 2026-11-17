#!/usr/bin/env python
"""Landsat and Sentinel-2 crops for the cyclegan: unpaired in train, same place and close dates in val and test."""

import json                                      # metadata, stats and cache files
import logging                                   # structured logging
from collections import defaultdict              # crops grouped by place
from dataclasses import dataclass                # crop and sample descriptions
from datetime import date as Date                # day gap between two dates
from pathlib import Path                         # file paths

import numpy as np                               # qa arrays
import rasterio                                  # read the tifs
import torch                                     # tensors
from rasterio.windows import Window              # read one crop only

from cyclegan.data.sensors import BAND_FILES, LANDSAT, METADATA_FILE, RATIO, SENTINEL, Sensor

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")                # val and test both come from the test aois
SPLIT_FOLDER = {"train": "train", "val": "test", "test": "test"}


# =============================================================================
# Crops and samples
# =============================================================================
@dataclass(frozen=True)
class Crop:
    """One crop of one date: grid cell (i, j) and its top-left pixel."""

    aoi: str                                     # e.g. '22NBK_12'
    date: str                                    # 'yyyy-mm-dd'
    path: Path                                   # date folder
    i: int                                       # grid row of the crop
    j: int                                       # grid column of the crop
    row: int                                     # top-left pixel row
    col: int                                     # top-left pixel column

    def info(self) -> str:
        """Short text for logs and figures."""
        return f"{self.aoi} {self.date} ({self.i},{self.j})"


@dataclass
class Sample:
    """One landsat crop and one sentinel crop, normalized, with their good-pixel masks."""

    landsat: torch.Tensor                        # 6 h w
    landsat_mask: torch.Tensor                   # 1 h w, True = good pixel
    sentinel: torch.Tensor                       # 6 3h 3w
    sentinel_mask: torch.Tensor                  # 1 3h 3w
    landsat_info: str                            # aoi, date and cell of the landsat crop
    sentinel_info: str                           # aoi, date and cell of the sentinel crop


# =============================================================================
# Helpers
# =============================================================================
def crop_starts(image_size: int, crop_size: int, crops_per_side: int) -> list[int]:
    """Top-left offsets of crops_per_side crops spread from edge to edge, e.g. 330, 128, 3 -> 0 101 202."""
    if crops_per_side == 1:
        return [(image_size - crop_size) // 2]
    step = (image_size - crop_size) / (crops_per_side - 1)
    return [round(k * step) for k in range(crops_per_side)]


def tile(aoi: str) -> str:
    """MGRS tile of an aoi name, e.g. 22NBK_12 -> 22NBK."""
    return aoi.split("_")[0]


def in_split(aoi: str, split: str, val_tiles: list[str]) -> bool:
    """val = test aois of the val tiles, test = the other test aois."""
    if split == "val":
        return tile(aoi) in val_tiles
    if split == "test":
        return tile(aoi) not in val_tiles
    return True


def load_stats(stats_dir: str, sensor: Sensor) -> tuple[torch.Tensor, torch.Tensor]:
    """loc and scale of stats/<sensor>.json, shaped 6 1 1."""
    stats = json.loads(Path(stats_dir, f"{sensor.name}.json").read_text())
    loc = torch.tensor(stats["loc"], dtype=torch.float32).view(-1, 1, 1)
    scale = torch.tensor(stats["scale"], dtype=torch.float32).view(-1, 1, 1)
    return loc, scale


def days_between(a: str, b: str) -> int:
    """Absolute number of days between two 'yyyy-mm-dd' dates."""
    return abs((Date.fromisoformat(a) - Date.fromisoformat(b)).days)


# =============================================================================
# Dataset
# =============================================================================
class CycleDataset(torch.utils.data.Dataset):
    """Clean crops of both sensors.

    train: every clean landsat crop once per epoch, with a random clean sentinel crop.
    val, test: landsat and sentinel crops of the same aoi and cell, at most max_pair_days apart.
    No augmentation: crops are always the grid cells, north-up.
    """

    def __init__(
        self,
        landsat_root: str,                       # landsat download root
        sentinel_root: str,                      # sentinel download root
        stats_dir: str,                          # landsat.json and sentinel.json from stats.py
        cache_dir: str,                          # crop clean fractions, computed once
        split: str,                              # 'train', 'val' or 'test'
        val_tiles: list[str],                    # test tiles used for val
        min_clean: float = 0.5,                  # smallest clean fraction of a date
        min_crop_clean: float = 0.9,             # smallest clean fraction of a crop
        crop_size: int = 128,                    # landsat crop, sentinel crop is RATIO times larger
        crops_per_side: int = 3,                 # crop grid per image side
        max_pair_days: int = 5,                  # val and test: largest gap between the two dates
    ):
        """Index the clean crops of both sensors; no image is read here except the qa bands, once."""
        super().__init__()
        assert split in SPLITS, f"split must be one of {SPLITS}"
        self.split = split
        self.val_tiles = list(val_tiles)
        self.min_clean = min_clean
        self.min_crop_clean = min_crop_clean
        self.crop_size = crop_size
        self.crops_per_side = crops_per_side
        self.cache_dir = Path(cache_dir)
        self.training = split == "train"

        self.stats = {sensor: load_stats(stats_dir, sensor) for sensor in (LANDSAT, SENTINEL)}
        self.landsat_crops = self._index(LANDSAT, landsat_root)
        self.sentinel_crops = self._index(SENTINEL, sentinel_root)
        self.pairs = [] if self.training else self._pair(max_pair_days)
        self._log_summary()

    # -------------------------------------------------------------------------
    # Index
    # -------------------------------------------------------------------------
    def _usable_dates(self, root: str) -> list[tuple[str, str, Path, int]]:
        """(aoi, date, date folder, image size) of every complete date above min_clean in this split."""
        dates = []
        for metadata_path in sorted(Path(root, SPLIT_FOLDER[self.split]).glob(f"*/{METADATA_FILE}")):
            aoi = metadata_path.parent.name
            if not in_split(aoi, self.split, self.val_tiles):
                continue
            metadata = json.loads(metadata_path.read_text())
            for date, info in metadata["dates"].items():
                if info["complete"] and (info.get("clean_fraction") or 0.0) >= self.min_clean:
                    dates.append((aoi, date, metadata_path.parent / date, metadata["width"]))
        return dates

    def _crop_fractions(self, sensor: Sensor, path: Path, starts: list[int], size: int) -> list[list[float]]:
        """Clean fraction of every grid cell of one date, from its qa band."""
        with rasterio.open(path / sensor.qa_file) as src:
            bad = sensor.bad_pixels(src.read(1))
        return [[round(float(1.0 - bad[r:r + size, c:c + size].mean()), 4) for c in starts] for r in starts]

    def _index(self, sensor: Sensor, root: str) -> list[Crop]:
        """Every crop above min_crop_clean; cell fractions are cached by date folder."""
        scale = RATIO if sensor is SENTINEL else 1               # sentinel grid is RATIO times finer
        size = self.crop_size * scale
        cache_path = self.cache_dir / f"{sensor.name}_crop{self.crop_size}_n{self.crops_per_side}.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        nb_new = 0

        crops = []
        for aoi, date, path, image_size in self._usable_dates(root):
            # same cells for both sensors: sentinel offsets are the landsat offsets times RATIO
            starts = [s * scale for s in crop_starts(image_size // scale, self.crop_size, self.crops_per_side)]
            key = str(path)
            if key not in cache:                                   # first time this date is seen
                cache[key] = self._crop_fractions(sensor, path, starts, size)
                nb_new += 1
            for i, row in enumerate(starts):
                for j, col in enumerate(starts):
                    if cache[key][i][j] >= self.min_crop_clean:
                        crops.append(Crop(aoi, date, path, i, j, row, col))

        if nb_new:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
            logger.info("%s: %d new dates added to %s", sensor.name, nb_new, cache_path)
        return crops

    def _pair(self, max_days: int) -> list[tuple[Crop, Crop]]:
        """For every landsat crop, the sentinel crop of the same aoi and cell with the closest date, if close enough."""
        by_place = defaultdict(list)
        for crop in self.sentinel_crops:
            by_place[(crop.aoi, crop.i, crop.j)].append(crop)

        pairs = []
        for landsat in self.landsat_crops:
            candidates = by_place.get((landsat.aoi, landsat.i, landsat.j), [])
            if not candidates:
                continue
            closest = min(candidates, key=lambda s: days_between(s.date, landsat.date))
            if days_between(closest.date, landsat.date) <= max_days:
                pairs.append((landsat, closest))
        return pairs

    def _log_summary(self) -> None:
        """Crops of each sensor and, outside train, the number of pairs."""
        logger.info("%s | landsat %d crops | sentinel %d crops%s", self.split,
                    len(self.landsat_crops), len(self.sentinel_crops),
                    "" if self.training else f" | {len(self.pairs)} pairs")

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------
    def _read(self, sensor: Sensor, crop: Crop) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized 6-band crop and its good-pixel mask; bad pixels are 0."""
        scale = RATIO if sensor is SENTINEL else 1
        size = self.crop_size * scale
        window = Window(crop.col, crop.row, size, size)

        bands = []
        for file in BAND_FILES:
            with rasterio.open(crop.path / file) as src:
                bands.append(src.read(1, window=window))
        with rasterio.open(crop.path / sensor.qa_file) as src:
            good = ~sensor.bad_pixels(src.read(1, window=window))

        loc, spread = self.stats[sensor]
        image = (torch.from_numpy(sensor.reflectance(np.stack(bands))) - loc) / spread
        mask = torch.from_numpy(good).unsqueeze(0)
        return image * mask, mask

    # -------------------------------------------------------------------------
    # Items
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Landsat crops in train, pairs in val and test."""
        return len(self.landsat_crops) if self.training else len(self.pairs)

    def __getitem__(self, index: int) -> Sample:
        """Train: landsat crop index with a random sentinel crop. Val and test: pair index."""
        if self.training:
            landsat_crop = self.landsat_crops[index]
            sentinel_crop = self.sentinel_crops[int(torch.randint(0, len(self.sentinel_crops), (1,)))]
        else:
            landsat_crop, sentinel_crop = self.pairs[index]

        landsat, landsat_mask = self._read(LANDSAT, landsat_crop)
        sentinel, sentinel_mask = self._read(SENTINEL, sentinel_crop)
        return Sample(landsat, landsat_mask, sentinel, sentinel_mask, landsat_crop.info(), sentinel_crop.info())