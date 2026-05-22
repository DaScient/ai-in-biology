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

## 15.2  Earth-observation foundation models

`Prithvi` (NASA / IBM), `Clay`, and `SatMAE` are masked-autoencoder transformers pre-trained on multi-spectral, multi-temporal patches. Their embeddings transfer to:

- Crop-type classification with < 1 % of normal labels.
- Burned-area mapping immediately after fire.
- Mangrove cover regression matched to LIDAR ground truth.

A reproducible recipe: freeze backbone, attach a UPerNet decoder, fine-tune on 500 labeled tiles.

## 15.3  Coupling biology with climate

Steps to do this responsibly:

1. **Choose scenarios.** SSP1-2.6 (mitigation) and SSP3-7.0 (high emissions) bracket plausible futures.
2. **Bias-correct climate output.** `xclim` or `ISIMIP` recipes; do not feed raw GCM output to an SDM.
3. **Downscale.** Statistical (BCSD) or dynamical (CMIP regional) — be explicit.
4. **Propagate uncertainty.** Sample multiple GCMs and scenarios; report ensemble spread.
5. **Validate against held-out years.** A model that cannot reproduce 2010–2020 will not reliably predict 2050.

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

## 15.5  Pitfalls

- **Cloud contamination.** Use cloud-masked composites or SAR.
- **Sensor calibration drift.** Long time series across sensor generations need harmonization (HLS, Landsat Collection 2).
- **Equifinality.** Many parameter combinations fit historical data; only some predict the future. Always show ensembles.
- **Attribution overreach.** A correlation between a biological collapse and warming is not yet causation.

## 15.6  Exercises

1. **Phenology.** Build NDVI time series from Sentinel-2 for a deciduous forest plot. Fit a double-logistic curve and report start-of-season for each year 2017–2024.
2. **Foundation transfer.** Fine-tune Prithvi for crop classification on 1 % of the EuroCrops labels. Compare to a UNet trained from scratch on 100 %.
3. **Climate ensemble.** Project an SDM for a temperate tree using three GCMs × two SSPs. Report the ensemble agreement raster.
4. **Lidar biomass.** Match GEDI footprints to Landsat tiles. Regress above-ground biomass with XGBoost; report R² spatial-blocked.

## 15.7  Further reading

- Camps-Valls, G. *Deep learning for the Earth sciences.* Wiley (2021).
- Jakubik, J. *Prithvi.* (NASA / IBM, 2023).
- Rolf, E. *A generalizable approach for satellite imagery analysis (MOSAIKS).* Nat Commun (2021).
- Reichstein, M. *Deep learning and process understanding for data-driven Earth system science.* Nature (2019).

## See also

- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)
- [Ecology API](../api/ecology.md)
- [Chapter 22 — Co-evolution of AI & Life](chapter_22_coevolution.md)
