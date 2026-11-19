#!/usr/bin/env python
"""Copy the usable dates of each sensor, compressed, into a smaller folder for kaggle."""

import json                                      # metadata files
import logging                                   # structured logging
import shutil                                    # disk usage
from concurrent.futures import ThreadPoolExecutor  # dates written in parallel
from pathlib import Path                         # file paths

import rasterio                                  # read and write the tifs

from cyclegan.data.sensors import BAND_FILES, LANDSAT, METADATA_FILE, SENTINEL, Sensor

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
SENSOR_ROOTS = {                                  # download root of each sensor
    LANDSAT: "/media/mrhouma/2C1E-61AC/landsat_gan",
    SENTINEL: "/media/mrhouma/2C1E-61AC/sentinel2_gan",
}
OUT_ROOTS = {                                     # export root of each sensor, one kaggle dataset each
    LANDSAT: "/media/mrhouma/2C1E-61AC/kaggle/landsat-gan",
    SENTINEL: "/media/mrhouma/2C1E-61AC/kaggle/sentinel2-gan",
}
SPLITS = ("train", "test")
MIN_CLEAN = 0.5                                   # same threshold as the dataset
NB_WORKERS = 8                                    # dates written in parallel

# lossless: same values, smaller files; tiles make crop reads fast
COMPRESSION = {"compress": "deflate", "predictor": 2, "zlevel": 6,
               "tiled": True, "blockxsize": 256, "blockysize": 256}


# =============================================================================
# Copy
# =============================================================================
def copy_tif(src_path: Path, dst_path: Path) -> None:
    """Same pixels and grid, compressed and tiled; existing files are skipped."""
    if dst_path.exists():
        return
    with rasterio.open(src_path) as src:
        profile = {**src.profile, **COMPRESSION}
        data = src.read()
    part = dst_path.with_suffix(".part")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(data)
    part.rename(dst_path)


def copy_date(sensor: Sensor, src_dir: Path, dst_dir: Path) -> None:
    """The 6 bands and the quality band of one date."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file in (*BAND_FILES, sensor.qa_file):
        copy_tif(src_dir / file, dst_dir / file)


def usable(info: dict) -> bool:
    return info["complete"] and (info.get("clean_fraction") or 0.0) >= MIN_CLEAN


def export_aoi(sensor: Sensor, metadata_path: Path, out_dir: Path, pool: ThreadPoolExecutor) -> tuple[int, int]:
    """Usable dates of one aoi, then its metadata.json with only those dates."""
    metadata = json.loads(metadata_path.read_text())
    dates = {date: info for date, info in metadata["dates"].items() if usable(info)}
    jobs = [pool.submit(copy_date, sensor, metadata_path.parent / date, out_dir / date) for date in dates]
    for job in jobs:
        job.result()                                                 # raise here if a copy failed
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / METADATA_FILE).write_text(json.dumps({**metadata, "dates": dates, "export_min_clean": MIN_CLEAN}, indent=2))
    return len(dates), len(metadata["dates"])


def export_sensor(sensor: Sensor) -> None:
    src_root, out_root = Path(SENSOR_ROOTS[sensor]), Path(OUT_ROOTS[sensor])
    kept = total = 0
    with ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        for split in SPLITS:
            paths = sorted((src_root / split).glob(f"*/{METADATA_FILE}"))
            for i, path in enumerate(paths, 1):
                aoi_kept, aoi_total = export_aoi(sensor, path, out_root / split / path.parent.name, pool)
                kept, total = kept + aoi_kept, total + aoi_total
                logger.info("%s %s %3d/%d %-10s | %3d/%3d dates kept", sensor.name, split, i, len(paths),
                            path.parent.name, aoi_kept, aoi_total)
    logger.info("%s: %d/%d dates kept -> %s (%.1f GB)", sensor.name, kept, total, out_root, folder_gb(out_root))


def folder_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drive = Path(SENSOR_ROOTS[LANDSAT]).parent                       # the export goes on the same drive
    logger.info("free space on %s: %.0f GB", drive, shutil.disk_usage(drive).free / 1e9)
    for sensor in SENSOR_ROOTS:
        export_sensor(sensor)


if __name__ == "__main__":
    main()