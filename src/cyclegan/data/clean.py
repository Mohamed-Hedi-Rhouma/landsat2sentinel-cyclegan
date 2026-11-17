#!/usr/bin/env python
"""Clean-pixel fraction of every date, written into metadata.json."""

import json                                      # metadata file
import logging                                   # structured logging
from pathlib import Path                         # file paths

import numpy as np                               # pixel counts
import rasterio                                  # read the qa tif

from cyclegan.data.sensors import LANDSAT, METADATA_FILE, SENTINEL, Sensor

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
SENSOR_ROOTS = {                                  # download root of each sensor
    LANDSAT: "/media/mrhouma/2C1E-61AC/landsat_gan",
    SENTINEL: "/media/mrhouma/2C1E-61AC/sentinel2_gan",
}
MIN_CLEAN = 0.5                                   # used in the logs only


# =============================================================================
# Measure
# =============================================================================
def clean_fraction(sensor: Sensor, qa_path: Path) -> float:
    """Fraction of good pixels in one qa tif."""
    with rasterio.open(qa_path) as src:
        qa = src.read(1)
    return float(1.0 - sensor.bad_pixels(qa).mean())


def clean_aoi(sensor: Sensor, metadata_path: Path) -> list[float]:
    """Write clean_fraction for every date of one aoi; None if the date is incomplete."""
    metadata = json.loads(metadata_path.read_text())
    fractions = []
    for date, info in metadata["dates"].items():
        qa_path = metadata_path.parent / date / sensor.qa_file
        if not info["complete"] or not qa_path.exists():     # missing band
            info["clean_fraction"] = None
            continue
        info["clean_fraction"] = round(clean_fraction(sensor, qa_path), 4)
        fractions.append(info["clean_fraction"])
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return fractions


def clean_split(sensor: Sensor, split_dir: Path) -> None:
    """Every aoi of one split, one log line per aoi and one summary."""
    all_fractions = []
    for metadata_path in sorted(split_dir.glob(f"*/{METADATA_FILE}")):
        fractions = clean_aoi(sensor, metadata_path)
        all_fractions.extend(fractions)
        logger.info("%s %s %-10s | %s", sensor.name, split_dir.name, metadata_path.parent.name, count(fractions))
    logger.info("%s %s TOTAL      | %s", sensor.name, split_dir.name, count(all_fractions))


def count(fractions: list[float]) -> str:
    """'n dates, k clean (p%)' with clean meaning >= MIN_CLEAN."""
    if not fractions:
        return "0 dates"
    kept = int((np.array(fractions) >= MIN_CLEAN).sum())
    return f"{len(fractions):4d} dates, {kept:4d} clean ({100 * kept / len(fractions):3.0f}%)"


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("clean = at least %.0f%% good pixels", 100 * MIN_CLEAN)
    for sensor, root in SENSOR_ROOTS.items():
        if not Path(root).is_dir():
            logger.warning("%s: %s not found, skipped", sensor.name, root)
            continue
        for split_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            clean_split(sensor, split_dir)


if __name__ == "__main__":
    main()