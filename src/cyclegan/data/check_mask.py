#!/usr/bin/env python
"""Check the cloud mask of one date: rgb, the quality value of every pixel, and the resulting mask."""

from pathlib import Path                         # file paths

import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # arrays
import rasterio                                  # read the tifs

from cyclegan.data.sensors import LANDSAT, SENTINEL

# =============================================================================
# Config: the date to check
# =============================================================================
SENSOR = SENTINEL                                # SENTINEL or LANDSAT
ROOTS = {SENTINEL: "/media/mrhouma/2C1E-61AC/sentinel2_gan", LANDSAT: "/media/mrhouma/2C1E-61AC/landsat_gan"}
AOI = "30SUF_12"                                 # full aoi name, or just its tile: every match is listed
DATE = "2022-01-28"                              # yyyy-mm-dd

LANDSAT_BITS = {0: "fill", 1: "dilated cloud", 2: "cirrus", 3: "cloud", 4: "cloud shadow", 5: "snow",
                6: "clear", 7: "water"}


def read(path: Path) -> np.ndarray:
    """First band of a tif."""
    with rasterio.open(path) as src:
        return src.read(1)


def rgb(date_dir: Path) -> np.ndarray:
    """Red green blue reflectance, 2-98 % stretch."""
    image = np.stack([SENSOR.reflectance(read(date_dir / f"{b}.tif")) for b in ("red", "green", "blue")], -1)
    low, high = np.percentile(image, (2, 98), axis=(0, 1))
    return np.clip((image - low) / np.maximum(high - low, 1e-6), 0, 1)


def find_date_dir() -> Path:
    """The <split>/<aoi>/<date> folder matching AOI and DATE; the first one if several aois match."""
    matches = sorted(p for p in Path(ROOTS[SENSOR]).glob(f"*/{AOI}*/{DATE}") if (p / SENSOR.qa_file).exists())
    if not matches:
        raise SystemExit(f"no {SENSOR.name} folder for {AOI}* on {DATE} with {SENSOR.qa_file}")
    print("matches:", [f"{p.parent.parent.name}/{p.parent.name}" for p in matches])
    return matches[0]


def print_quality(qa: np.ndarray) -> None:
    """Sentinel: share of pixels per clear-score range. Landsat: share of pixels per qa bit."""
    if SENSOR is SENTINEL:
        edges = (0, 1, 20, 40, SENSOR.qa_threshold, 80, 101)
        for low, high in zip(edges[:-1], edges[1:]):
            share = 100 * ((qa >= low) & (qa < high)).mean()
            flag = "BAD" if high <= SENSOR.qa_threshold else ""
            label = "0 (no data, saturated or surely cloud)" if low == 0 else f"{low}-{high - 1}"
            print(f"  clear score {label:40s} {share:5.1f}%  {flag}")
    else:
        for bit, name in LANDSAT_BITS.items():
            share = 100 * ((qa.astype(np.int64) >> bit) & 1).mean()
            flag = "BAD" if bit in SENSOR.qa_invalid else ""
            print(f"  bit {bit} {name:13s} {share:5.1f}%  {flag}")


def main():
    date_dir = find_date_dir()
    qa = read(date_dir / SENSOR.qa_file)
    bad = SENSOR.bad_pixels(qa)
    print(f"{date_dir} | {SENSOR.qa_file} values {qa.min()}..{qa.max()} | bad {100 * bad.mean():.1f}%")
    print_quality(qa)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained")
    axes[0].imshow(rgb(date_dir), interpolation="nearest")
    axes[0].set_title(f"rgb | {date_dir.parent.name} {date_dir.name}")
    if SENSOR is SENTINEL:
        shown = axes[1].imshow(qa, cmap="RdYlGn", vmin=0, vmax=100, interpolation="nearest")
        figure.colorbar(shown, ax=axes[1], fraction=0.046, label="clear score (cs_cdf x 100)")
        axes[1].set_title(f"cloud score+: red = cloud, green = clear, bad below {SENSOR.qa_threshold}")
    else:
        axes[1].imshow(qa, interpolation="nearest")
        axes[1].set_title("qa_pixel raw values")
    axes[2].imshow(bad, cmap="gray_r", interpolation="nearest")
    axes[2].set_title(f"our mask: black = bad ({100 * bad.mean():.1f}%)")
    for ax in axes:
        ax.axis("off")
    figure.savefig("check_mask.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()