#!/usr/bin/env python
"""Quick look: for each sensor, random dates already on disk with enough clean pixels, in rgb.

Works during the download: reads the date folders directly, no metadata.json or clean.py needed.
"""

import random                                    # random dates
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
NB_AOIS = 4                                      # aois drawn at random, per sensor
MIN_CLEAN = 0.5                                  # clean fraction of a usable date
SEED = None                                      # None = new aois every run
OUT_FILE = "preview_rgb.png"                     # also saved here


def complete_dates(sensor: Sensor, root: Path) -> list[Path]:
    """Date folders <split>/<aoi>/<date> with all 6 bands and the qa file."""
    files = (*BAND_FILES, sensor.qa_file)
    return [d for d in root.glob("*/*/*") if d.is_dir() and all((d / f).exists() for f in files)]


def clean_fraction(sensor: Sensor, date_dir: Path) -> float:
    """Fraction of good pixels, from the qa file."""
    with rasterio.open(date_dir / sensor.qa_file) as src:
        return float(1.0 - sensor.bad_pixels(src.read(1)).mean())


def read_rgb(sensor: Sensor, date_dir: Path) -> np.ndarray:
    """h w 3 image, red green blue, 2-98 % stretch on good pixels, bad pixels magenta."""
    bands = []
    for name in ("red", "green", "blue"):
        with rasterio.open(date_dir / f"{name}.tif") as src:
            bands.append(sensor.reflectance(src.read(1)))
    with rasterio.open(date_dir / sensor.qa_file) as src:
        good = ~sensor.bad_pixels(src.read(1))
    rgb = np.stack(bands, axis=-1)
    low, high = np.percentile(rgb[good], (2, 98), axis=0)
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    rgb[~good] = (1, 0, 1)
    return rgb


def pick(sensor: Sensor, root: Path) -> list[tuple[Path, float]]:
    """Up to NB_AOIS random clean dates, each from a different aoi: (date folder, clean fraction)."""
    date_dirs = complete_dates(sensor, root)
    random.shuffle(date_dirs)
    picks, seen_aois = [], set()
    for date_dir in date_dirs:
        if date_dir.parent in seen_aois:
            continue
        clean = clean_fraction(sensor, date_dir)
        if clean >= MIN_CLEAN:
            picks.append((date_dir, clean))
            seen_aois.add(date_dir.parent)
        if len(picks) == NB_AOIS:
            break
    print(f"{sensor.name}: {len(date_dirs)} complete dates on disk")
    return picks


def main():
    random.seed(SEED)
    figure, axes = plt.subplots(2, NB_AOIS, figsize=(4.5 * NB_AOIS, 10), squeeze=False,
                                layout="constrained")                  # leaves room for every title
    for ax_row, (sensor, root) in zip(axes, SENSOR_ROOTS.items()):
        picks = pick(sensor, Path(root)) if Path(root).is_dir() else []
        print(f"  shown: {[f'{d.parent.name} {d.name}' for d, _ in picks] or 'nothing clean yet'}")
        for k, ax in enumerate(ax_row):
            ax.axis("off")
            if k >= len(picks):
                continue
            date_dir, clean = picks[k]
            ax.imshow(read_rgb(sensor, date_dir), interpolation="nearest")
            split = date_dir.parent.parent.name
            ax.set_title(f"{sensor.name} | {split}/{date_dir.parent.name}\n{date_dir.name} | {100 * clean:.0f}% clean",
                         fontsize=10)

    figure.suptitle("rgb, 2-98 % stretch per image, magenta = masked", fontsize=13)
    figure.savefig(OUT_FILE, dpi=120)
    print(f"saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()