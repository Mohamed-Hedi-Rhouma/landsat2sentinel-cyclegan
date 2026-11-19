#!/usr/bin/env python
"""Quick look: same scene seen by landsat (top) and sentinel (bottom), for random aois, in rgb.
"""

import random                                    # random aois
from datetime import date                        # date gaps
from functools import cache                      # clean fraction read once per date
from pathlib import Path                         # file paths

import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # images
import rasterio                                  # read the tifs

from cyclegan.data.sensors import BAND_FILES, LANDSAT, SENTINEL, Sensor

# =============================================================================
# Config
# =============================================================================
SENSOR_ROOTS = {
    LANDSAT: "/media/mrhouma/2C1E-61AC/landsat_gan",
    SENTINEL: "/media/mrhouma/2C1E-61AC/sentinel2_gan",
}
NB_AOIS = 4                                      # aois drawn at random, one per column
MIN_CLEAN = 0.5                                  # clean fraction of a usable date
MAX_DAYS = 5                                     # max gap between the landsat and sentinel dates
SEED = None                                      # None = new aois every run
OUT_FILE = "preview_rgb.png"                     # also saved here


def complete_dates(sensor: Sensor, root: Path) -> list[Path]:
    """Date folders <split>/<aoi>/<date> with all 6 bands and the qa file."""
    files = (*BAND_FILES, sensor.qa_file)
    return [d for d in root.glob("*/*/*") if d.is_dir() and all((d / f).exists() for f in files)]


def dates_by_aoi(sensor: Sensor) -> dict[str, list[Path]]:
    """Aoi name -> its complete date folders."""
    by_aoi = {}
    for date_dir in complete_dates(sensor, Path(SENSOR_ROOTS[sensor])):
        by_aoi.setdefault(date_dir.parent.name, []).append(date_dir)
    return by_aoi


@cache
def clean_fraction(sensor: Sensor, date_dir: Path) -> float:
    """Fraction of good pixels, from the qa file."""
    with rasterio.open(date_dir / sensor.qa_file) as src:
        return float(1.0 - sensor.bad_pixels(src.read(1)).mean())


def days_between(a: Path, b: Path) -> int:
    return abs((date.fromisoformat(a.name) - date.fromisoformat(b.name)).days)


def find_pair(landsat_dates: list[Path], sentinel_dates: list[Path]) -> tuple[Path, Path] | None:
    """Closest clean (landsat, sentinel) dates of one aoi, at most MAX_DAYS apart."""
    pairs = [(days_between(l, s), l, s) for l in landsat_dates for s in sentinel_dates
             if days_between(l, s) <= MAX_DAYS]
    for _, landsat_dir, sentinel_dir in sorted(pairs):
        if clean_fraction(LANDSAT, landsat_dir) >= MIN_CLEAN and clean_fraction(SENTINEL, sentinel_dir) >= MIN_CLEAN:
            return landsat_dir, sentinel_dir
    return None


def pick_pairs() -> list[tuple[Path, Path]]:
    """Up to NB_AOIS random aois with a clean pair: (landsat date folder, sentinel date folder)."""
    landsat, sentinel = dates_by_aoi(LANDSAT), dates_by_aoi(SENTINEL)
    aois = sorted(landsat.keys() & sentinel.keys())
    random.shuffle(aois)
    print(f"{len(aois)} aois on disk for both sensors")
    pairs = []
    for aoi in aois:
        pair = find_pair(landsat[aoi], sentinel[aoi])
        if pair:
            pairs.append(pair)
        if len(pairs) == NB_AOIS:
            break
    return pairs


def read_rgb(sensor: Sensor, date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """h w 3 reflectance (red green blue) and h w good-pixel mask."""
    bands = []
    for name in ("red", "green", "blue"):
        with rasterio.open(date_dir / f"{name}.tif") as src:
            bands.append(sensor.reflectance(src.read(1)))
    with rasterio.open(date_dir / sensor.qa_file) as src:
        good = ~sensor.bad_pixels(src.read(1))
    return np.stack(bands, axis=-1), good


def stretch(rgb: np.ndarray, good: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Scale to 0-1 with the given limits, bad pixels magenta."""
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    rgb[~good] = (1, 0, 1)
    return rgb


def main():
    random.seed(SEED)
    pairs = pick_pairs()
    print(f"  shown: {[f'{l.parent.name} {l.name} / {s.name}' for l, s in pairs] or 'no clean pair yet'}")

    figure, axes = plt.subplots(2, NB_AOIS, figsize=(4.5 * NB_AOIS, 10), squeeze=False,
                                layout="constrained")                  # leaves room for every title
    for ax in axes.flat:
        ax.axis("off")

    for k, (landsat_dir, sentinel_dir) in enumerate(pairs):
        landsat_rgb, landsat_good = read_rgb(LANDSAT, landsat_dir)
        sentinel_rgb, sentinel_good = read_rgb(SENTINEL, sentinel_dir)
        low, high = np.percentile(sentinel_rgb[sentinel_good], (2, 98), axis=0)   # same stretch for both
        gap = days_between(landsat_dir, sentinel_dir)
        split = landsat_dir.parent.parent.name

        rows = ((LANDSAT, landsat_dir, landsat_rgb, landsat_good), (SENTINEL, sentinel_dir, sentinel_rgb, sentinel_good))
        for ax, (sensor, date_dir, rgb, good) in zip(axes[:, k], rows):
            ax.imshow(stretch(rgb, good, low, high), interpolation="nearest")
            ax.set_title(f"{sensor.name} | {split}/{date_dir.parent.name}\n"
                         f"{date_dir.name} | {100 * clean_fraction(sensor, date_dir):.0f}% clean | {gap} days apart",
                         fontsize=10)

    figure.suptitle("same aoi per column, same 2-98 % stretch per column, magenta = masked", fontsize=13)
    figure.savefig(OUT_FILE, dpi=120)
    print(f"saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()