# CycleGAN Landsat ↔ Sentinel-2

Translate **Landsat 8/9 (30 m)** images into **Sentinel-2-like (10 m)** images, and back, without paired training data.

The task is treated as **super-resolution plus a radiometric correction**: fields, roads and rivers stay in place. What changes is the resolution (30 m → 10 m) and the reflectance values themselves, because the two sensors don't measure exactly the same values for the same ground (slightly different band widths, processing and acquisition conditions). The model is a CycleGAN whose generators are built like super-resolution networks (ESRGAN-style RRDB blocks).

---

## How it works

```
Landsat 128×128 (30 m) ──G_L2S──► fake Sentinel 384×384 (10 m) ──G_S2L──► reconstructed Landsat
Sentinel 384×384 (10 m) ──G_S2L──► fake Landsat 128×128 (30 m) ──G_L2S──► reconstructed Sentinel

D_sentinel: real or fake Sentinel?      D_landsat: real or fake Landsat?
```

- **Training is unpaired:** each Landsat crop is paired with a random clean Sentinel crop, from any place and date.
- **Validation and test are paired:** a Landsat and a Sentinel crop of the same place, at most 5 days apart, so the fake can be compared with the real image of the other sensor.
- **Clouds are masked everywhere:** in crop selection, in every loss and in every metric.

---

## Data

### Areas of interest

105 squares of **9.9 km**, each in its own UTM projection, with corners on the 30 m grid (`src/cyclegan/data/aois/`). Dates: **2022**.

| Split | Source | AOIs | MGRS tiles |
|---|---|---|---|
| train | `aois/train` | 64 | 28 |
| val | `aois/test`, tiles listed in `datamodule.val_tiles` | 18 | 9 |
| test | `aois/test`, the other tiles | 23 | 11 |

Splits are by MGRS tile, so val and test areas are never neighbours of each other. Regions: French Guiana, West Africa (test only), Spain, France, Germany / Switzerland / Italy, Sweden, Balkans.

### Sensors

| | Landsat | Sentinel-2 |
|---|---|---|
| Earth Engine collections | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | `COPERNICUS/S2_SR_HARMONIZED` |
| Bands | blue, green, red, nir, swir1, swir2 | same 6 bands (B2, B3, B4, B8, B11, B12) |
| Grid | 30 m, 330 × 330 per AOI | 10 m, 990 × 990 per AOI |
| Reflectance | `dn × 2.75e-5 − 0.2` | `dn / 10000` |
| Cloud mask | `QA_PIXEL` bits: fill, dilated cloud, cirrus, cloud, shadow | Cloud Score+ `cs_cdf` < 0.6, or SCL no data / saturated |

Both sensors are downloaded on the **same grid** (same CRS and corner), so one Landsat pixel covers exactly 3 × 3 Sentinel pixels. Same-day scenes are mosaicked so the whole AOI is covered.

Sentinel-2's own SCL cloud mask was tested first and rejected: it missed cloud edges and haze, and labelled mountain slopes as cloud shadow. Cloud Score+ replaced it.

### On disk

```
<root>/<split>/<aoi>/metadata.json                  grid, scenes of every date, clean fraction
<root>/<split>/<aoi>/<date>/blue.tif ... swir2.tif   uint16 digital numbers
<root>/<split>/<aoi>/<date>/qa_pixel.tif | cs.tif    quality band
```

---

## Dataset

- **Dates:** complete, and at least **50 %** clean pixels.
- **Crops:** a fixed 3 × 3 grid per image. Landsat 128 px at offsets 0, 101, 202, and Sentinel 384 px at the same offsets × 3, so cell (i, j) covers the same ground for both sensors. Only cells with at least **90 %** clean pixels are kept. Cell clean fractions are cached.
- **Normalization:** `(reflectance − median) / (p95 − p5)` per band, computed on the clean pixels of the train dates (`stats/<sensor>.json`).
- **No augmentation:** crops are always the grid cells, north-up.
- **One sample:** `landsat [6,128,128]`, `landsat_mask [1,128,128]`, `sentinel [6,384,384]`, `sentinel_mask [1,384,384]`, and the AOI, date and cell of each crop.

---

## Model

### Generators (`model/generator.py`)

One class, used for both directions. The body works on the Landsat grid (128 × 128).

```
G_L2S:  Landsat → conv → N × RRDB (+ long skip) → Up ×3 → conv → head   + bicubic ×3 of the input
G_S2L:  Sentinel → Down ×3 → N × RRDB (+ long skip) → conv → head       + MTF-blurred ×3 downsampling of the input
```

- **Residual output:** the network only learns the correction on top of the interpolated input. The head starts at zero, so training starts from the interpolated image.
- **RRDB blocks** (ESRGAN): dense connections, residual scaling 0.2, no normalization layers (the reflectance level is kept), LeakyReLU 0.2.
- **Up** = nearest upsampling + conv. **Down** = conv with kernel = stride = 3, one output pixel per Landsat pixel footprint.

### Discriminators (`model/discriminator.py`)

| Option | Output |
|---|---|
| `MultiScaleDiscriminator` (default) | PatchGAN with spectral norm, at full and half scale: one score per ~70 × 70 px patch |
| `UNetDiscriminator` | U-Net with spectral norm (Real-ESRGAN style): one score per pixel |

Either can take a **high-pass** input (image minus its blur), so it judges detail only.

### Losses (`model/loss.py`)

| Loss | What it checks | Default weight |
|---|---|---|
| GAN (`lsgan` or `relativistic`) | fakes look like real images of the other sensor | 1 |
| Cycle (Huber, δ = 0.1, or L1) | the round trip gives the input back | 10 |
| LPIPS, per band, masked (VGG16) | the round trip keeps textures | 1 |
| SAM, on reflectance | the round trip keeps band ratios | 0 (off) |
| Consistency L2S | the fake Sentinel, blurred like Landsat (MTF 0.3) and downsampled, matches the real Landsat after a per-band gain/offset fit | 1 |
| Consistency S2L | the fake Landsat matches the real Sentinel seen by Landsat | 0 (off) |

The per-band gain/offset fit is detached: it absorbs the colour difference between the sensors, so the consistency loss only checks where things are, not their colour.

### Training (`model/training_module.py`)

- PyTorch Lightning, manual optimization. Each batch: **D step**, then **G step** with the discriminators frozen.
- **Image pool** of 50 past fakes for the discriminators (0 = off).
- **Adam** at 1e-4 with betas (0.5, 0.999). The learning rate is constant, then decays linearly to 0 over the second half of training. Gradient norm is clipped at 1.
- Steps with a non-finite loss are skipped and counted.

### Validation and test

On the paired crops, for each direction (`l2s_*`, `s2l_*`):

| Metric | Meaning |
|---|---|
| `mae` | reflectance error, raw (radiometry) |
| `mae_fit` | reflectance error after a per-band gain/offset fit (structure only) |
| `mae_<band>` | error of each band |
| `lpips`, `sam` | perceptual and spectral similarity |
| `cycle_l`, `cycle_s` | round-trip error |

- **Checkpoints:** `best.ckpt` (lowest `val/l2s_lpips`) and `last.ckpt`, both overwritten. A run restarts from `last.ckpt` automatically.
- **Figures:** after each validation, 16 fixed val pairs from different AOIs, in 4 PNGs per composite (RGB, NIR, SWIR). Each row shows the input, the fake, the real target with its date gap, and an error map. Colour scales are fixed per row for the whole run, and masked pixels are magenta. The figures are also in TensorBoard.

---

## Project structure

```
hydra/                         configs, one folder per component
src/cyclegan/
  train.py                     entry point
  data/
    aois/                      area files, train/ and test/
    sensors.py                 bands, scaling, cloud-mask rules, grid
    earth_engine.py            read AOIs, exact grid, download one band
    download_landsat.py        script
    download_sentinel.py       script
    clean.py                   script: clean fraction of every date
    stats.py                   script: normalization stats
    dataset.py                 CycleDataset
    datamodule.py              Batch, CycleDataModule
    preview.py, check_mask.py  quick visual checks of the data and masks
  model/
    blocks.py                  ConvBlock, DenseBlock, RRDB, Down, Up
    generator.py               Generator
    discriminator.py           MultiScaleDiscriminator, UNetDiscriminator
    degradation.py             MTF blur, mask resizing, gain/offset fit
    lpips.py                   masked LPIPS (CESBIO)
    loss.py                    CycleLoss
    image_pool.py              ImagePool
    training_module.py         CycleGANModule
  callbacks/
    visualization.py           TranslationFigures
```

---

## Usage

### Environment

```bash
pixi install
pixi run earthengine authenticate          # once, for the downloads
```

Main packages: `pytorch`, `pytorch-lightning`, `hydra-core`, `rasterio`, `numpy`, `matplotlib`, `tensorboard`, `einops`, `piq`, `earthengine-api`, `requests`.

### Data pipeline

Paths and settings are in the `Config` block at the top of each script.

```bash
python -m cyclegan.data.download_landsat     # resumable: existing files are skipped
python -m cyclegan.data.download_sentinel
python -m cyclegan.data.clean                # writes clean_fraction into metadata.json
python -m cyclegan.data.preview              # random clean dates of each sensor, RGB
python -m cyclegan.data.check_mask           # one date: RGB, quality band, mask
python -m cyclegan.data.stats                # writes stats/landsat.json and stats/sentinel.json
```

If the masks change, run `clean.py` again and delete the crop cache (`results/cache/`).

### Training

```bash
python -m cyclegan.train                                   # paths in hydra/location/
python -m cyclegan.train run_test=true                     # also test once with best.ckpt at the end
```

Any setting can be changed from the command line. Use a new `name` for each experiment, because the same name resumes the previous run.

```bash
python -m cyclegan.train name=unet_d discriminator@d_sentinel=unet
python -m cyclegan.train name=highpass d_sentinel.highpass=true
python -m cyclegan.train name=relativistic loss.model.gan_type=relativistic
python -m cyclegan.train name=sam loss.model.lambda_sam=1
python -m cyclegan.train name=no_pool training_module.module.pool_size=0
```

Outputs go to `results/<name>/`: `checkpoints/`, `figures/`, `tensorboard/`.