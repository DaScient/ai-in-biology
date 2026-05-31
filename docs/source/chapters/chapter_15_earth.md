# Chapter 15 — Earth Systems & Planetary Biology

> *"The biosphere is the largest single experiment we are running, and we are an inadvertent perturbation in it."*

## Learning objectives

- Connect remote-sensing data products (Sentinel-2, MODIS, ICESat-2, GEDI) to biological questions.
- Use foundation models trained on Earth observation (Prithvi, SatMAE, Clay) for downstream segmentation and regression.
- Couple ecological models with climate model output (CMIP6) for plausible projections.
- Reason about uncertainty cascades in planetary-scale analyses.

## 15.1  Earth-observation data, mapped to biological questions

| Sensor | Native resolution | Biological signal |
|--------|--------------------|--------------------|
| Sentinel-2 (MSI) | 10–60 m, 5 day | Vegetation, water, snow |
| Sentinel-1 (SAR) | 10 m, 6 day | Biomass through cloud, flooding |
| MODIS | 250–1000 m, daily | Phenology, sea-surface temp |
| ICESat-2 (lidar) | 17 m footprint | Canopy height, ice elevation |
| GEDI | 25 m footprint | Forest carbon |
| PACE / OCI | 1 km | Phytoplankton functional types |

### 15.1a  Accessing and processing Sentinel-2 — a practical guide

The table above lists what each sensor measures. Here is a hands-on recipe for actually downloading and preprocessing Sentinel-2 surface reflectance.

**Access options:**

- **Microsoft Planetary Computer** (free, requires login): STAC (SpatioTemporal Asset Catalog) API access to Sentinel-2 L2A (surface reflectance, cloud-masked).
- **Google Earth Engine** (free for research, requires registration): best for large-scale analysis.
- **Copernicus Open Access Hub** (free, no registration for downloads): direct download of original SAFE files.

**Accessing Sentinel-2 via the Planetary Computer STAC API:**

```python
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
import matplotlib.pyplot as plt

# Open the STAC catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

# Define area of interest (e.g., Amazon rainforest, bounding box)
bbox = [-60.0, -10.0, -59.5, -9.5]  # (min_lon, min_lat, max_lon, max_lat)

# Search for Sentinel-2 L2A scenes in 2023 with low cloud cover
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2023-06-01/2023-09-01",
    query={"eo:cloud_cover": {"lt": 10}},
)
items = search.get_all_items()
print(f"Found {len(items)} scenes")

# Load the first scene's red, green, blue, and NIR bands
item = items[0]
band_urls = {
    "R": item.assets["B04"].href,    # Red
    "G": item.assets["B03"].href,    # Green
    "B": item.assets["B02"].href,    # Blue
    "NIR": item.assets["B08"].href,  # Near-infrared
}
# Sign URLs (Planetary Computer requires signing)
for band, url in band_urls.items():
    band_urls[band] = pc.sign(url)

# Read a subset (1000x1000 pixels) for speed
window = rasterio.windows.Window(1000, 1000, 1000, 1000)
with rasterio.open(band_urls["R"]) as src:
    red_subset = src.read(1, window=window)
with rasterio.open(band_urls["G"]) as src:
    green_subset = src.read(1, window=window)
with rasterio.open(band_urls["B"]) as src:
    blue_subset = src.read(1, window=window)

# RGB composite (scaling to 0-255 for display)
rgb = np.stack([red_subset, green_subset, blue_subset], axis=-1)
rgb = (rgb / rgb.max() * 255).astype(np.uint8)
plt.imshow(rgb)
plt.axis("off")
plt.title("Sentinel-2 RGB (subset)")
```

**Computing NDVI (Normalized Difference Vegetation Index):**

```python
# NDVI = (NIR - Red) / (NIR + Red)
with rasterio.open(band_urls["NIR"]) as src:
    nir = src.read(1, window=window).astype(np.float32)
red_arr = red_subset.astype(np.float32)
ndvi = (nir - red_arr) / (nir + red_arr + 1e-6)
plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
plt.colorbar(label="NDVI")
plt.title("NDVI (green = high vegetation)")
```

**Pitfall.** Sentinel-2 L2A includes a cloud mask, but thin clouds and shadows remain. Use the `SCL` (Scene Classification) band to mask out clouds, shadows, and cirrus before computing indices.

## 15.2  Earth-observation foundation models

`Prithvi` (NASA / IBM), `Clay`, and `SatMAE` are masked-autoencoder transformers pre-trained on multi-spectral, multi-temporal patches. Their embeddings transfer to:

- Crop-type classification with < 1 % of normal labels.
- Burned-area mapping immediately after fire.
- Mangrove cover regression matched to LIDAR ground truth.

A reproducible recipe: freeze backbone, attach a UPerNet decoder, fine-tune on 500 labeled tiles.

### 15.2a  Fine-tuning Prithvi for a downstream task

Here is a complete fine-tuning example for **burned-area mapping** (a binary segmentation task).

**Assumptions:** you have a small set of labeled tiles (500 tiles, 256×256 pixels each) where each pixel is labeled as burned (1) or unburned (0) from a previous fire event.

**Steps:**

1. Load a pre-trained Prithvi model (ViT backbone + decoder).
2. Replace the decoder head with a segmentation head (2 classes: burned / unburned).
3. Fine-tune on your labeled tiles.
4. Compare to a baseline (e.g., thresholding dNBR as in §15.4).

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel

# Load pre-trained Prithvi model (example; actual model name may vary).
# In practice, use the official checkpoint.
model_name = "ibm-nasa-geospatial/prithvi-100m"  # placeholder
config = AutoConfig.from_pretrained(model_name)
backbone = AutoModel.from_pretrained(model_name)

class PrithviSegmentationHead(nn.Module):
    """Add a segmentation decoder on top of Prithvi."""
    def __init__(self, backbone, num_classes=2, hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        # Simplified decoder: conv stack + upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(backbone.config.hidden_size, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=16, mode="bilinear"),  # adjust to backbone stride
        )

    def forward(self, x):
        # x: (B, C, H, W) multispectral image
        features = self.backbone(x).last_hidden_state  # (B, seq_len, D)
        # Reshape tokens back to spatial (assumes square patch grid)
        B, seq_len, D = features.shape
        H = int(seq_len ** 0.5)
        W = H
        features = features.permute(0, 2, 1).reshape(B, D, H, W)
        return self.decoder(features)

class BurnedAreaDataset(Dataset):
    def __init__(self, tile_paths, label_paths, transform=None):
        self.tile_paths = tile_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile = np.load(self.tile_paths[idx])    # (C, H, W) numpy
        label = np.load(self.label_paths[idx])  # (H, W) binary
        if self.transform:
            tile = self.transform(tile)
        tile = torch.as_tensor(tile, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)
        return tile, label

# Fine-tune
model = PrithviSegmentationHead(backbone, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Build the dataset from your labeled tiles (paths to .npy arrays)
train_dataset = BurnedAreaDataset(train_tile_paths, train_label_paths)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for epoch in range(20):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)  # (B, 2, H, W)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss {total_loss / len(dataloader):.4f}")
```

**Expected improvement.** Fine-tuned Prithvi should achieve IoU (Intersection over Union) > 0.7 on held-out burned areas, compared to dNBR thresholding (IoU ≈ 0.5). The gain comes from handling cloud shadows, mixed pixels, and varied vegetation.

**Pitfall.** Prithvi was pre-trained on a specific set of bands (Sentinel-2 bands 2–12). If your input has a different band set, you need to align it or fine-tune a new projection layer.

## 15.3  Coupling biology with climate

Steps to do this responsibly:

1. **Choose scenarios.** SSP1-2.6 (mitigation) and SSP3-7.0 (high emissions) bracket plausible futures.
2. **Bias-correct climate output.** `xclim` or `ISIMIP` recipes; do not feed raw GCM output to an SDM.
3. **Downscale.** Statistical (BCSD) or dynamical (CMIP regional) — be explicit.
4. **Propagate uncertainty.** Sample multiple GCMs and scenarios; report ensemble spread.
5. **Validate against held-out years.** A model that cannot reproduce 2010–2020 will not reliably predict 2050.

### 15.3a  Bias correction using quantile mapping

Step 2 above says "do not feed raw GCM output to an SDM." Here is a complete implementation of **quantile mapping**, the workhorse method for bias-correcting climate model outputs against historical observations.

**Why it is needed.** Global Climate Models (GCMs) have systematic biases (e.g., too warm or too dry). Quantile mapping aligns the distribution of GCM outputs to observed historical data.

**Method.** For each month and location, compute the empirical cumulative distribution function (CDF) of the GCM historical runs and of the observed data. Map future GCM values to the observed value at the same quantile.

```python
import numpy as np
from scipy.interpolate import interp1d

def quantile_mapping(gcm_historical, obs_historical, gcm_future, n_quantiles=100):
    """Bias-correct gcm_future using quantile mapping from the historical period.

    gcm_historical: historical GCM values (e.g., daily temperature)
    obs_historical: observed historical values
    gcm_future:     future GCM values to correct
    Returns the bias-corrected future array.
    """
    quantiles = np.linspace(0, 1, n_quantiles)
    gcm_quantiles = np.percentile(gcm_historical, quantiles * 100)
    obs_quantiles = np.percentile(obs_historical, quantiles * 100)

    # Map a GCM value to its quantile in the historical GCM distribution
    interp_to_quantile = interp1d(
        gcm_quantiles, quantiles, kind="linear",
        bounds_error=False, fill_value=(0, 1),
    )
    # Map a quantile to the corresponding observed value
    interp_to_obs = interp1d(
        quantiles, obs_quantiles, kind="linear",
        bounds_error=False, fill_value=(obs_quantiles[0], obs_quantiles[-1]),
    )

    quantile_future = interp_to_quantile(gcm_future)
    return interp_to_obs(quantile_future)

# Example: correct temperature for a single grid cell.
# Toy data (np.random) stands in for real series; in practice use actual GCM
# output (e.g., CMIP6) and an observational product (e.g., ERA5, CHIRPS).
gcm_hist = np.random.normal(15, 2.0, 1000)    # GCM historical (mean 15 C)
obs_hist = np.random.normal(14, 1.5, 1000)    # Observed (mean 14 C)
gcm_future = np.random.normal(17, 2.5, 1000)  # Future GCM (mean 17 C)
corrected_future = quantile_mapping(gcm_hist, obs_hist, gcm_future)

print(f"GCM future mean: {gcm_future.mean():.2f} C")
print(f"Corrected future mean: {corrected_future.mean():.2f} C")
# Roughly 16 C: the future GCM mean (17 C) minus the GCM warm bias (~1 C),
# since the GCM runs ~1 C warmer than observations over the historical period.
```

**Visualization.** Plot Q-Q plots before and after correction to confirm the distributions align.

**Pitfall.** Quantile mapping assumes the bias is **stationary** — that the relationship between GCM and reality does not change over time. For some variables (e.g., precipitation extremes), this may be false.

## 15.4  Worked example — burned-area mapping

```python
import rasterio
import numpy as np
import torch

# load pre/post Sentinel-2 stacks (B, T, H, W) for an area of interest
pre = torch.tensor(load_s2("aoi", "2024-08-01", "2024-08-10"))
post = torch.tensor(load_s2("aoi", "2024-08-15", "2024-08-25"))

nbr_pre = (pre[7] - pre[11]) / (pre[7] + pre[11] + 1e-6)   # NIR vs SWIR
nbr_post = (post[7] - post[11]) / (post[7] + post[11] + 1e-6)
dnbr = nbr_pre - nbr_post

burned = (dnbr > 0.27).cpu().numpy()
print(f"Burned-area mask: {burned.sum() * 100 / burned.size:.2f} % of AOI")
```

Replace the threshold step with a fine-tuned Prithvi head to halve false positives.

### 15.4a  Beyond dNBR — a U-Net trained on Sentinel-2

The threshold above is a strong baseline, but a learned model captures spatial context. Here we extend to a **U-Net** trained on Sentinel-2 imagery.

**Data.** Pre-fire and post-fire Sentinel-2 mosaics (6 bands: B2, B3, B4, B8, B11, B12). Labels: burned/unburned from high-resolution reference data (e.g., VIIRS active fire or manual delineation).

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.out(d1))
```

**Training.** Use binary cross-entropy loss and data augmentation (random flips, rotations, brightness shifts). Evaluate on held-out fires, never on tiles from the same fire used in training.

**Expected result.** The U-Net should reach IoU ≈ 0.75–0.85, compared to dNBR thresholding IoU ≈ 0.5–0.6.

**Pitfall.** A U-Net needs many labeled pixels (millions) to generalize. For small fires (rare events), use transfer learning from a pre-trained segmentation model (e.g., the Prithvi fine-tuning in §15.2a).

## 15.5  Pitfalls

- **Cloud contamination.** Use cloud-masked composites or SAR.
- **Sensor calibration drift.** Long time series across sensor generations need harmonization (HLS, Landsat Collection 2).
- **Equifinality.** Many parameter combinations fit historical data; only some predict the future. Always show ensembles.
- **Attribution overreach.** A correlation between a biological collapse and warming is not yet causation.

### 15.5a  Equifinality and attribution in depth

**Equifinality.** Many different combinations of parameters (e.g., tree growth rate, mortality rate, soil carbon decay) can produce the same historical carbon flux. A model that fits history well may still fail in the future because the *mechanisms* are wrong.

*Diagnostic.* Run the model with multiple plausible parameter sets (e.g., draws from a Bayesian posterior) and project future outcomes. If the range is large (e.g., a 500% spread in carbon uptake), the model is not robust — report that ensemble spread as uncertainty rather than a single point estimate.

**Attribution caution.** A correlation between warming and species decline is not causation. To strengthen causal claims:

- Use **convergent cross-mapping** (CCM) from nonlinear dynamics.
- Use **instrumental variables** (e.g., volcanic eruptions as natural experiments for cooling).
- Use **process-based models** that explicitly include mechanisms (e.g., temperature-dependent metabolism).

Many published Earth-system "attributions" are correlational and may be overturned by new data.

## 15.6  Exercises

1. **Phenology.** Build NDVI time series from Sentinel-2 for a deciduous forest plot. Fit a double-logistic curve and report start-of-season for each year 2017–2024.
2. **Foundation transfer.** Fine-tune Prithvi for crop classification on 1 % of the EuroCrops labels. Compare to a UNet trained from scratch on 100 %.
3. **Climate ensemble.** Project an SDM for a temperate tree using three GCMs × two SSPs. Report the ensemble agreement raster.
4. **Lidar biomass.** Match GEDI footprints to Landsat tiles. Regress above-ground biomass with XGBoost; report R² spatial-blocked.
5. **dNBR vs U-Net.** Download a Sentinel-2 tile pair before and after a known fire (e.g., the 2019–2020 Australian bushfires). Compute dNBR and threshold it. Then train a small U-Net (§15.4a) on 10 tiles and test on a held-out tile. Which method yields higher IoU? How much does adding a second pre-fire tile (time series) help the U-Net?
6. **Quantile mapping for precipitation.** Download daily precipitation from a GCM (e.g., CMIP6) and a gridded observation product (e.g., CHIRPS) for the same region. Apply quantile mapping (§15.3a) and plot the Q-Q plot of monthly totals before and after correction. Does the correction reduce the dry bias?
7. **Prithvi few-shot crops.** Use the EuroCrops dataset (or a subset). Fine-tune Prithvi for pixel-wise crop-type classification. Compare to a U-Net trained from scratch on 100% of the labels. How few labeled tiles (1%, 10%, 50%) does Prithvi need to match the U-Net?
8. **Phenology shift detection.** Compute an NDVI time series for a forest site from 2015–2024. Fit a double-logistic curve for each year and extract start-, peak-, and end-of-season. Test for a linear trend: is the growing season getting longer? Compare to a site with a known drought impact.

## 15.7  Further reading

- Camps-Valls, G. *Deep learning for the Earth sciences.* Wiley (2021).
- Jakubik, J. *Prithvi.* (NASA / IBM, 2023).
- Rolf, E. *A generalizable approach for satellite imagery analysis (MOSAIKS).* Nat Commun (2021).
- Reichstein, M. *Deep learning and process understanding for data-driven Earth system science.* Nature (2019).
- Rolf, E. et al. *Generalizable machine learning for Earth system science: a review and benchmark.* Nature Reviews Earth & Environment (2024).
- Jakubik, J. et al. *Prithvi: a foundation model for Earth observation.* NASA Technical Report (2023).
- Cannon, A. J. *Multivariate quantile mapping bias correction: an N-dimensional pdf transform for climate simulations of multiple variables.* Climate Dynamics (2018).
- Stammer, D. *Attribution of extreme weather events: methods and challenges.* Annual Review of Marine Science (2020).

## See also

- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)
- [Ecology API](../api/ecology.md)
- [Chapter 22 — Co-evolution of AI & Life](chapter_22_coevolution.md)
