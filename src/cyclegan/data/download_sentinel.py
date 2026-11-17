#!/usr/bin/env python
"""Download Sentinel-2 of every aoi on its 10 m grid."""

import json                                      # metadata file
import logging                                   # structured logging
from collections import defaultdict              # scenes grouped by date
from concurrent.futures import ThreadPoolExecutor  # aois in parallel
from pathlib import Path                         # file paths

import ee                                        # earth engine api

from cyclegan.data.earth_engine import Aoi, Grid, aoi_grid, day_mosaic, download_band, init_earth_engine, read_aois
from cyclegan.data.sensors import BAND_FILES, METADATA_FILE, SENTINEL

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
ROOT = "/home/mrhouma/Documents/Cyclegan/cyclegan"
AOI_DIR = f"{ROOT}/src/cyclegan/data/aois"        # train/ and test/ aoi files
OUT_DIR = "/media/mrhouma/2C1E-61AC/sentinel2_gan"   # download root
EE_PROJECT = "get-eo-data-500210"                # earth engine cloud project
MAX_CLOUD = 100.0                                  # max granule cloud cover in %, loose first filter
NB_WORKERS = 8                                    # aois downloaded in parallel

# Earth Engine names
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"                           # sentinel-2 surface reflectance
CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"             # cloud score+, linked by system:index
EE_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")                    # same order as sensors.BANDS, b11 b12 at 20 m
SCORE_BAND = "cs_cdf"                                                # 0 = not clear ... 1 = clear, 10 m
QA_BAND = "QA"                                                       # cs_cdf x 100, 0 where scl is no data or saturated
SCL_UNUSABLE = (0, 1)                                                # scl no data, saturated
PROPERTIES = ("system:index", "SPACECRAFT_NAME", "CLOUDY_PIXEL_PERCENTAGE")   # read for every scene


# =============================================================================
# Scenes
# =============================================================================
def add_quality(image: ee.Image) -> ee.Image:
    """Add QA = cloud score+ cs_cdf x 100, set to 0 where scl says no data or saturated."""
    scl = image.select("SCL")
    unusable = scl.eq(SCL_UNUSABLE[0]).Or(scl.eq(SCL_UNUSABLE[1]))
    score = image.select(SCORE_BAND).multiply(100).round()
    return image.addBands(score.where(unusable, 0).rename(QA_BAND))


def aoi_scenes(aoi: Aoi, max_cloud: float) -> ee.ImageCollection:
    """Sentinel-2 granules over the aoi, in its dates, below max_cloud % of granule cloud cover, with QA."""
    return (ee.ImageCollection(COLLECTION)
            .filterBounds(aoi.region())
            .filterDate(aoi.start_date, aoi.end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .linkCollection(ee.ImageCollection(CLOUD_SCORE), [SCORE_BAND])
            .filter(ee.Filter.listContains("system:band_names", SCORE_BAND))   # granules without a score dropped
            .map(add_quality))


def scenes_by_date(scenes: ee.ImageCollection) -> dict[str, list[dict]]:
    """Granules grouped by date; id = <sensing>_<generation>_T<tile>."""
    rows = scenes.reduceColumns(ee.Reducer.toList(len(PROPERTIES)), list(PROPERTIES)).get("list").getInfo()
    dates = defaultdict(list)
    for scene_id, satellite, cloud in rows:
        date = f"{scene_id[0:4]}-{scene_id[4:6]}-{scene_id[6:8]}"          # yyyymmdd -> yyyy-mm-dd
        dates[date].append({"id": scene_id, "tile": scene_id[-5:], "satellite": satellite, "cloud_cover": cloud})
    return dict(sorted(dates.items()))


# =============================================================================
# Download
# =============================================================================
def download_date(scenes: ee.ImageCollection, date: str, grid: Grid, date_dir: Path) -> bool:
    """Every band of one date, same-day granules mosaicked; True if all files are there."""
    date_dir.mkdir(parents=True, exist_ok=True)
    start = ee.Date(date)
    day = scenes.filterDate(start, start.advance(1, "day"))
    reflectance = day_mosaic(day, "bilinear")                # smooth values, also 20 m swir onto 10 m
    qa = day_mosaic(day, "nearest")                          # scores, never interpolated

    done = [download_band(reflectance, band, grid, date_dir / file) for band, file in zip(EE_BANDS, BAND_FILES)]
    done.append(download_band(qa, QA_BAND, grid, date_dir / SENTINEL.qa_file, fill=SENTINEL.qa_fill))
    return all(done)


def write_metadata(aoi: Aoi, grid: Grid, dates: dict[str, list[dict]], complete: dict[str, bool], path: Path) -> None:
    """Aoi, grid and granules of every date; complete is False when a band failed to download."""
    metadata = {
        "name": aoi.name,
        "split": aoi.split,
        "sensor": SENTINEL.name,
        "collections": [COLLECTION, CLOUD_SCORE],
        "crs": grid.crs,
        "transform": list(grid.transform),
        "width": grid.width,
        "height": grid.height,
        "start_date": aoi.start_date,
        "end_date": aoi.end_date,
        "dates": {date: {"complete": complete[date], "scenes": scenes} for date, scenes in dates.items()},
    }
    path.write_text(json.dumps(metadata, indent=2))


def download_aoi(aoi: Aoi, out_dir: Path, max_cloud: float) -> None:
    """Every date of one aoi, then its metadata.json."""
    aoi_dir = out_dir / aoi.split / aoi.name
    aoi_dir.mkdir(parents=True, exist_ok=True)
    grid = aoi_grid(aoi, SENTINEL)
    scenes = aoi_scenes(aoi, max_cloud)
    dates = scenes_by_date(scenes)
    logger.info("%s: %d dates to download", aoi.name, len(dates))

    complete = {date: download_date(scenes, date, grid, aoi_dir / date) for date in dates}
    write_metadata(aoi, grid, dates, complete, aoi_dir / METADATA_FILE)
    logger.info("%s: done, %d/%d dates complete", aoi.name, sum(complete.values()), len(dates))


def safe_download_aoi(aoi: Aoi, out_dir: Path, max_cloud: float) -> None:
    """download_aoi that logs an error instead of stopping the other aois."""
    try:
        download_aoi(aoi, out_dir, max_cloud)
    except Exception as e:
        logger.error("%s failed: %s", aoi.name, e)


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_earth_engine(EE_PROJECT)
    aois = read_aois(AOI_DIR)
    out_dir = Path(OUT_DIR)
    logger.info("%d aois -> %s | max granule cloud %.0f%% | %d workers", len(aois), out_dir, MAX_CLOUD, NB_WORKERS)

    with ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        for aoi in aois:
            pool.submit(safe_download_aoi, aoi, out_dir, MAX_CLOUD)
    logger.info("all aois done")


if __name__ == "__main__":
    main()