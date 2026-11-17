#!/usr/bin/env python
"""What the project knows about each sensor: band files, reflectance scaling, grid and bad-pixel rules."""

from dataclasses import dataclass                # frozen sensor description

import numpy as np                               # qa and reflectance arrays

# =============================================================================
# Files on disk, same names for both sensors
# =============================================================================
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")   # model band order
BAND_FILES = tuple(f"{band}.tif" for band in BANDS)          # one single-band tif per band
METADATA_FILE = "metadata.json"                              # aoi info and the scene of every date, one per aoi


# =============================================================================
# Sensor description
# =============================================================================
@dataclass(frozen=True)
class Sensor:
    """One sensor: grid, reflectance scaling and which quality values mark a bad pixel."""

    name: str                                    # short name, used in folders, logs and keys
    resolution_m: int                            # pixel size of the grid, in meters
    scale: float                                 # reflectance = dn * scale + offset
    offset: float                                # reflectance offset
    qa_file: str                                 # quality band file in each date folder
    qa_mode: str                                 # 'bits' (bit flags), 'classes' (class values) or 'threshold' (score)
    qa_invalid: tuple[int, ...]                  # bits or classes that make a pixel bad (bits and classes modes)
    qa_fill: int                                 # qa value written where there is no image, must count as bad
    qa_threshold: int = 0                        # threshold mode: a score below this is a bad pixel

    def pixels(self, side_m: float) -> int:
        """Number of pixels along a side of side_m meters; the side must fit the grid exactly."""
        n = side_m / self.resolution_m
        if n != int(n):
            raise ValueError(f"{side_m} m is not a multiple of the {self.name} {self.resolution_m} m grid")
        return int(n)

    def reflectance(self, dn: np.ndarray) -> np.ndarray:
        """Digital numbers to surface reflectance, float32."""
        return dn.astype(np.float32) * self.scale + self.offset

    def bad_pixels(self, qa: np.ndarray) -> np.ndarray:
        """True where the quality band marks a pixel as bad (no data, cloud, shadow, cirrus)."""
        if self.qa_mode == "bits":
            flags = sum(1 << bit for bit in self.qa_invalid)        # every bad bit in one mask
            return (qa.astype(np.int64) & flags) > 0
        if self.qa_mode == "threshold":
            return qa < self.qa_threshold                           # low clear score
        return np.isin(qa, self.qa_invalid)                         # bad class values


LANDSAT = Sensor(
    name="landsat",                              # landsat 8 and 9, collection 2 level 2
    resolution_m=30,
    scale=2.75e-5,                               # collection 2 surface reflectance scaling
    offset=-0.2,
    qa_file="qa_pixel.tif",                      # QA_PIXEL band
    qa_mode="bits",
    qa_invalid=(0, 1, 2, 3, 4),                  # fill, dilated cloud, cirrus, cloud, cloud shadow
    qa_fill=1,                                   # bit 0 set = fill
)

SENTINEL = Sensor(
    name="sentinel",                             # sentinel-2 surface reflectance, harmonized
    resolution_m=10,
    scale=1e-4,                                  # dn / 10000
    offset=0.0,
    qa_file="cs.tif",                            # cloud score+ cs_cdf x 100, 0 where scl says no data or saturated
    qa_mode="threshold",
    qa_invalid=(),                               # not used in threshold mode
    qa_fill=0,                                   # no image = score 0 = bad
    qa_threshold=60,                             # cs_cdf < 0.60 = not clear (earth engine example value)
)

SENSORS = {sensor.name: sensor for sensor in (LANDSAT, SENTINEL)}   # lookup by name
RATIO = LANDSAT.resolution_m // SENTINEL.resolution_m                # sentinel pixels per landsat pixel, per side (3)