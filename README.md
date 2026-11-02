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
   
   **Terminal 2 - React Dashboard UI:**
   ```powershell
   cd web
   npm install
   npm run dev
   ```
   Then open: http://localhost:8503

## Usage Examples

### Runtime modes

The Windows startup script and Docker deployment use `planetary_computer` mode. This fetches recent public Sentinel-1/Sentinel-2 imagery around the selected map point. If a sensor or scene is unavailable, the response reports that limitation. `demo_synthetic` remains available for offline interface testing only. Set `APP_ENV=production` to prevent synthetic predictions from being served.

The local API runs on port `8000` and the React dashboard on port `8503`. In Docker, the API is available on port `8000` and the React dashboard on port `8504`.

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

### Run Evaluation
```powershell
.venv\Scripts\python scripts\run_evaluation.py
```

This evaluates only reviewed independent test cases from `dataset/` and actual
saved predictions. It writes `evaluation/metrics_report.json`, consumed by the API
and React Evaluation page. All 17 metrics include eligibility counts; missing
verified truth or inference stays unavailable. The current downloaded examples
are acquisition candidates, not a completed research benchmark.

See [the dataset card and annotation protocol](dataset/README.md) for provenance,
source limitations, split review and remaining acquisition requirements. The local
`sample_*.tif` files correspond to a random generator and are excluded. Explicit
`--local-smoke` writes separate software diagnostics.

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

The current prototype uses **classification + coordinate regression + fixed-radius estimation**. The project is being upgraded toward **hazard-specific spatial footprint prediction + severity + derived area/radius + validation**. The spatial reconstruction system is not implemented yet; the model, training, inference, API, and dashboard still use the existing point/radius workflow.

The project is currently in the **event-data foundation phase**. The event manifest and per-event documentation track the data and independent ground truth that must be verified before historical reconstruction or quantitative validation. An event with unverified ground truth must not be reported as a quantitative benchmark result.

For detailed project report, see [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Team

- **Team D15**
- **Institution**: Amrita School of AI, Coimbatore

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Output assessment and reports

The **Assessment & report** page checks model output integrity, calculates
population/infrastructure exposure from supplied GeoJSON, and exports maps and
reports. It supports optional local Ollama wording. See
[evaluation/OUTPUT_ASSESSMENT.md](evaluation/OUTPUT_ASSESSMENT.md) for inputs,
commands and limits. This workflow does not substitute for independent model
validation or claim a vulnerability score without a reviewed fragility model.

## Saved analysis PDFs

Every successful point analysis automatically saves a PDF in `reports/analyses/` under the project root (including results served from cache). The result panel provides **View PDF**, which opens the saved report in a new tab, and **Download PDF**. The one-page PDF includes an executive summary, coordinates, all hazard scores, and estimated radius and area. Files remain available after restarting the app. Set `ANALYSIS_REPORTS_DIR` to override the storage directory; relative paths resolve from the project root. If storage fails, the analysis remains visible with a PDF saving error.

The **View reports** navbar page reads all saved PDFs directly from the local folder, including earlier reports without metadata. It supports searching, an embedded preview, opening in a new tab, and downloading. Reports are never automatically deleted. New reports contain one page with an executive summary, comparison chart, and impact estimates. A JSON snapshot is saved beside each new PDF for catalogue details and reproducibility. A failed automatic save can be retried with **Save PDF locally** without rerunning inference.

The PDF follows the selected **Hazard lens**. Flood, Wildfire, and Landslide reports identify that focus while preserving the model primary signal and all model scores. Switching lenses saves a matching report without rerunning inference. Impact radius and circular area are labelled with the primary signal they describe. The report library distinguishes the selected report focus from the model primary signal.
