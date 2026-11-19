#!/usr/bin/env python
"""Per-band normalization stats of each sensor, from the clean pixels of the train dates."""

import json                                      # metadata and stats files
import logging                                   # structured logging
from collections import defaultdict              # pixels grouped by satellite
from concurrent.futures import ThreadPoolExecutor  # dates read in parallel
from itertools import repeat                     # same sensor for every date
from pathlib import Path                         # file paths

import numpy as np                               # percentiles
import rasterio                                  # read the tifs

from cyclegan.data.sensors import BAND_FILES, BANDS, LANDSAT, METADATA_FILE, SENTINEL, Sensor

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/Cyclegan/cyclegan"
SENSOR_ROOTS = {                                  # download root of each sensor
    LANDSAT: "/media/mrhouma/2C1E-61AC/landsat_gan",
    SENTINEL: "/media/mrhouma/2C1E-61AC/sentinel2_gan",
}
STATS_DIR = f"{ROOT}/stats"                       # <sensor>.json, read by the dataset
SPLIT = "train"                                   # stats never see the test aois
MIN_CLEAN = 0.5                                   # dates below this clean fraction are skipped
PIXELS_PER_DATE = 20_000                          # random clean pixels kept per date
PERCENTILES = (5, 50, 95)                         # low, center, high
SEED = 0                                          # pixel sampling, one seed per date
NB_WORKERS = 8                                    # dates read in parallel
LOG_EVERY = 100                                   # progress line every n dates


# =============================================================================
# Read
# =============================================================================
def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def read_date(sensor: Sensor, date_dir: Path, seed: tuple[int, int]) -> np.ndarray:
    """Reflectance of at most PIXELS_PER_DATE random good pixels of one date, n_pixels x 6."""
    good = np.flatnonzero(~sensor.bad_pixels(read_band(date_dir / sensor.qa_file)))
    if len(good) > PIXELS_PER_DATE:                                  # sample first, convert only these
        good = np.random.default_rng(seed).choice(good, PIXELS_PER_DATE, replace=False)
    dn = np.stack([read_band(date_dir / file).ravel()[good] for file in BAND_FILES], axis=1)
    return sensor.reflectance(dn)


def date_satellite(info: dict) -> str:
    """Satellite of one date, e.g. LC08, or LC08+LC09 if the mosaic mixes both."""
    return "+".join(sorted({scene["satellite"] for scene in info["scenes"]}))


def usable_dates(sensor: Sensor, root: str) -> list[tuple[Path, str]]:
    """(date folder, satellite) of every train date with at least MIN_CLEAN clean pixels."""
    usable = []
    paths = sorted(Path(root, SPLIT).glob(f"*/{METADATA_FILE}"))
    for path in paths:
        dates = json.loads(path.read_text())["dates"]
        used = [(path.parent / date, date_satellite(info)) for date, info in dates.items()
                if (info.get("clean_fraction") or 0.0) >= MIN_CLEAN]   # cloudy, incomplete or not measured
        usable += used
        logger.info("%s %-10s | %3d/%3d dates used", sensor.name, path.parent.name, len(used), len(dates))
    logger.info("%s: %d aois, %d dates to read", sensor.name, len(paths), len(usable))
    return usable


def collect(sensor: Sensor, root: str) -> tuple[dict[str, np.ndarray], int]:
    """Sampled clean pixels of every usable train date, grouped by satellite, and the number of dates used."""
    dates = usable_dates(sensor, root)
    date_dirs = [date_dir for date_dir, _ in dates]
    seeds = [(SEED, k) for k in range(len(dates))]                   # same pixels whatever the thread order
    chunks = defaultdict(list)
    with ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        for k, pixels in enumerate(pool.map(read_date, repeat(sensor), date_dirs, seeds), 1):
            chunks[dates[k - 1][1]].append(pixels)
            if k % LOG_EVERY == 0 or k == len(dates):
                logger.info("%s %d/%d dates read", sensor.name, k, len(dates))
    by_satellite = {satellite: np.concatenate(parts) for satellite, parts in sorted(chunks.items())}
    return by_satellite, len(dates)


# =============================================================================
# Stats
# =============================================================================
def band_stats(pixels: np.ndarray) -> dict:
    """Median as loc, 5-95 percentile range as scale, per band."""
    low, center, high = np.percentile(pixels, PERCENTILES, axis=0)
    return {
        "loc": center.round(6).tolist(),
        "scale": (high - low).round(6).tolist(),
        "p05": low.round(6).tolist(),
        "p95": high.round(6).tolist(),
    }


def log_satellites(sensor: Sensor, by_satellite: dict[str, np.ndarray]) -> None:
    """Median per band of each satellite, to check they can share one domain."""
    logger.info("%s median per band | %s", sensor.name, " ".join(f"{band:>6s}" for band in BANDS))
    for satellite, pixels in by_satellite.items():
        medians = np.median(pixels, axis=0)
        logger.info("  %-10s %9d px | %s", satellite, len(pixels), " ".join(f"{m:6.4f}" for m in medians))


def save(sensor: Sensor, stats: dict, nb_pixels: int, nb_dates: int) -> None:
    """Write stats/<sensor>.json."""
    path = Path(STATS_DIR, f"{sensor.name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"sensor": sensor.name, "bands": list(BANDS), "split": SPLIT, "min_clean": MIN_CLEAN,
               "nb_dates": nb_dates, "nb_pixels": nb_pixels, **stats}
    path.write_text(json.dumps(content, indent=2))
    logger.info("%s loc   %s", sensor.name, stats["loc"])
    logger.info("%s scale %s", sensor.name, stats["scale"])
    logger.info("%s saved -> %s", sensor.name, path)


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for sensor, root in SENSOR_ROOTS.items():
        by_satellite, nb_dates = collect(sensor, root)
        if not by_satellite:
            logger.error("%s: no clean date found, run clean.py first", sensor.name)
            continue
        pixels = np.concatenate(list(by_satellite.values()))
        log_satellites(sensor, by_satellite)
        save(sensor, band_stats(pixels), len(pixels), nb_dates)


if __name__ == "__main__":
    main()