# MultiHazard-VLM: Vision-Language Framework for Multi-Hazard Mapping on Earth Observation Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MultiHazard-VLM is an end-to-end vision-language framework designed for multi-hazard mapping (floods, wildfires, and rainfall-induced landslides) using Earth Observation (EO) satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical) combined with elevation, slope, NDVI, and lithology data.

## Architecture

MultiHazard-VLM uses a decoupled architecture where a shared spatiotemporal backbone and geospatial prompt encoder feed a cross-attention fusion layer, which subsequently branches into three decoupled prediction heads:

```
Sentinel-1/2 + DEM/Slope/NDVI/Lithology (via MS Planetary Computer)
                       │
       Cross-Modality Alignment & Masking Engine
                       │
 Spatiotemporal Mamba Visual Backbone ───────┐
                       │                     │
 Geospatial Prompt Encoder (H3 Grid) ────────┼─ Cross-Attention Fusion Layer
                                             │               │
      ┌──────────────────────────────────────┴───────────────┼──────────────────────────────────────┐
      │                                                      │                                      │
Language Reasoner                                   Coordinate Regressor                    Radius Estimator
(Alerts & Reports; grounded in geo-attributes)      (Sigmoid head: Lat/Lon bounding)        (ReLU head: Clamped [0.1, 30] km)
```

### Key Architectural Principles
1. **Decoupled Heads**: Language generation and numeric spatial regression are strictly separated into independent heads sharing a fused representation to prevent coordinate hallucination.
2. **Attribute-Grounded Text Generation**: The Language Reasoner is conditioned on extracted geospatial attributes (elevation, slope, NDVI, lithology) rather than raw pixels alone, and never outputs numeric coordinates or radii.
3. **Bounded Numeric Regressors**: Impact radius is estimated via a ReLU MLP clamped to `[0.1, 30.0]` km, while spatial epicenter coordinates use a Sigmoid activation for bounded output.
4. **Sensor Fault Tolerance**: Cross-modality masking engine explicitly flags missing channels instead of silently imputing data.

## Directory Structure

```
hazard/
├── configs/
│   └── model_config.yaml
├── data/
│   └── __init__.py
├── models/
│   ├── __init__.py
│   ├── backbone/
│   │   └── __init__.py
│   └── heads/
│       └── __init__.py
├── evaluation/
│   └── __init__.py
├── api/
│   └── __init__.py
├── dashboard/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Quick Start (Windows)

1. **Clone or extract the repository:**
   ```powershell
   cd d:\sem7\hazard
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the startup script to launch both API and Dashboard:**
   ```powershell
   .\start.bat
   ```

   Or manually start services:
   
   **Terminal 1 - API Backend:**
   ```powershell
   .venv\Scripts\python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```
   Then open: http://localhost:8000/docs
   
   **Terminal 2 - Dashboard UI:**
   ```powershell
   .venv\Scripts\python -m streamlit run dashboard/app.py --server.port 8503
   ```
   Then open: http://localhost:8503

## Usage Examples

### Runtime modes

The local dashboard runs in `demo_synthetic` mode by default. It uses synthetic model inputs for interface and pipeline testing and displays a warning in the dashboard. Set `APP_ENV=production` to prevent synthetic predictions from being served; production then requires a real inference provider through `INFERENCE_MODE`.

The local API runs on port `8001` and the dashboard on port `8503`. Docker keeps the API internal port at `8000` and supplies the dashboard service URL automatically.

### Training
```powershell
.venv\Scripts\python train.py --epochs 5 --batch_size 2 --img_size 64 --checkpoint_dir checkpoints
```

### Resume Training from Checkpoint
```powershell
.venv\Scripts\python train.py --epochs 10 --batch_size 2 --img_size 64 --checkpoint_dir checkpoints --resume checkpoints/model_epoch_5.pt
```

### Run Inference
```powershell
.venv\Scripts\python inference.py --checkpoint checkpoints/model_epoch_5.pt
```

### Generate Demo Output
```powershell
.venv\Scripts\python demo_output.py
```

### Run Tests
```powershell
.venv\Scripts\python -m pytest tests/test_pipeline.py -v
```

### View Configuration
```powershell
type configs\model_config.yaml
```

## Datasets

The framework integrates several key geospatial and hazard datasets:
| Hazard | Dataset | Link |
|--------|---------|------|
| Flood | S1S2-Water | [Zenodo: 11278238](https://zenodo.org/records/11278238) |
| Wildfire | HLS Burn Scars | [HuggingFace: ibm-nasa-geospatial](https://huggingface.co/datasets/ibm-nasa-geospatial/HLSL30_BurnScars) |
| Landslide | CAS Landslide | [Chinese Academy of Sciences](https://github.com/cas-landslide) |
| Geospatial Context | Planetary Computer | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) |

## Project Status

✅ **Complete and Verified**

- All three hazards (flood, wildfire, landslide) implemented
- Training pipeline functional with checkpoint support
- API backend serving predictions
- Streamlit dashboard operational
- Test suite passing (5/5 tests)
- Ready for demonstration and further extension

For detailed project report, see [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Team

- **Team D15**
- **Institution**: Amrita School of AI, Coimbatore

## License

This project is licensed under the MIT License - see the LICENSE file for details.
