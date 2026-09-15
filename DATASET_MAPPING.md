# Dataset mapping for this project

This project is best organized by hazard and source:

## 1) Flood
- Dataset: Global S1S2-Water Benchmark Dataset
- Source: Zenodo
- Link: https://zenodo.org/records/11278238
- Purpose: flood / water segmentation using Sentinel-1 VV/VH and Sentinel-2 bands
- Loader: S1S2WaterDataset

## 2) Wildfire
- Dataset: HLS Burn Scar Scenes
- Source: Hugging Face
- Link: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars
- Purpose: burn scar segmentation / wildfire mapping
- Loader: HLSBurnScarDataset

## 3) Ground truth / auxiliary geospatial context
- Platform: Microsoft Planetary Computer
- Source: Microsoft
- Link: https://planetarycomputer.microsoft.com/
- Purpose: terrain and context layers such as DEM, slope, NDVI, lithology, and EO retrieval
- Not a direct label dataset; use it to fetch external geospatial features

## 4) Landslide
- Dataset: CAS Landslide Dataset
- Source: Chinese Academy of Sciences
- Purpose: optional third hazard branch for landslide detection
- This is not part of the 3 links you provided, so it should be added separately if needed

## Recommended use
Use the flood and wildfire datasets for the main experiment, and use Planetary Computer for auxiliary features. Add landlside only if you want a full 3-hazard model.
