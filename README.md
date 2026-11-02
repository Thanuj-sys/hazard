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

## Running on a new system

**The repository contains the application source, configuration, map boundaries, and saved evaluation results. A clone alone is not enough for real satellite predictions: you must also transfer a trained model checkpoint.** Dependencies must be installed on each new system. Downloaded training/evaluation imagery, private configuration, database contents, and locally generated PDFs are separate from Git.

### What to transfer separately

| Item | Destination in the new clone | When needed |
| --- | --- | --- |
| Trained checkpoint | `checkpoints/model_epoch_9.pt` | Required for real satellite predictions; this is the checkpoint referenced by the saved transfer evaluation. |
| Matching model configuration | `configs/model_config.yaml` | Keep the configuration paired with the checkpoint; the current configuration is tracked. |
| Downloaded imagery and masks | Original paths under `dataset/` and `datasets/` | Required to rerun training or data-based evaluations, not to view saved results. Restore a backup or reacquire from the documented sources. |
| Saved PDF reports and JSON snapshots | `reports/analyses/` | Required to restore the old report library; a new system starts without those PDFs. |
| Database contents and private settings | PostgreSQL backup; environment variables | Needed to preserve database history and custom connections. Never commit credentials. |

Copy checkpoints from the original system or obtain them from the project maintainer. There is currently no checkpoint download bundled with this repository. The existing `model_epoch_9.pt` is approximately 145 MB and is excluded from Git. Its SHA-256 at the time these instructions were written is:

```text
9ab2119cda7e85e056f37dd5c8d494737acf2822dd78074af76b4768be6a11b1
```

The API selects the **lexicographically last `*.pt` filename** in `checkpoints/`, not the newest file by modification time or epoch number. On a new installation, keep only the checkpoint you intend to serve in that folder. Real inference rejects missing or incompatible weights. The API adapts the older checkpoint positional embedding to the configured image grid; loading `model_epoch_9.pt` through that adapter was verified with the tracked configuration. Loading the same weights directly without the adapter can raise a positional-embedding size mismatch. Training a new model is a separate workflow and will not reproduce the existing model automatically.

### Windows setup (PowerShell)

Use 64-bit Python and Node.js with npm, plus Git. The existing development environment uses Python **3.13.7** and Node.js **24.7.0**; these are reference versions, not a guarantee that every newer dependency combination is compatible. A GPU is not required by the current API, which loads the model on CPU. Allow disk space for Python/PyTorch dependencies and any downloaded satellite data. Internet access is needed for dependency installation, satellite imagery, weather, and map tiles.

Run these commands in a directory where you want the project:

```powershell
git clone --branch master https://github.com/Thanuj-sys/hazard.git
cd hazard
python --version
node --version
npm --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Push-Location web
npm ci
Pop-Location
```

Commands use the virtual environment's Python directly; activation and PowerShell execution-policy changes are unnecessary. Python requirements currently specify version ranges rather than a complete lockfile, so a fresh install can resolve different package versions. `npm ci` uses the committed frontend lockfile.

Restore the trained checkpoint before starting real inference. Replace the source path below with your backup location:

```powershell
New-Item -ItemType Directory -Path checkpoints -Force | Out-Null
Copy-Item -LiteralPath 'E:\hazard-backup\checkpoints\model_epoch_9.pt' -Destination checkpoints\model_epoch_9.pt
Get-FileHash -LiteralPath checkpoints\model_epoch_9.pt -Algorithm SHA256
```

Run all backend commands from the repository root. The API reads configuration and checkpoints relative to its working directory.

### Start the API and dashboard

In **terminal 1**, from the repository root:

```powershell
$env:APP_ENV = 'development'
$env:INFERENCE_MODE = 'planetary_computer'
$env:ALLOW_SYNTHETIC_INFERENCE = 'false'
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

In **terminal 2**, open the same repository and run:

```powershell
cd web
npm run dev
```

Open the dashboard at **http://localhost:8503** and API documentation at **http://localhost:8000/docs**. The frontend development server proxies `/api` to port `8000`. After setup, `start.bat` is also available as a Windows convenience launcher for real satellite mode; it sets its own inference variables, so use the manual commands for demo mode. Stop manually started servers with **Ctrl+C** in their terminals. For the batch launcher, stop the backend and dashboard processes in the windows it opens.

In another terminal, check API reachability:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

**The health response does not verify model readiness or database/cache connectivity.** It currently reports fixed component labels, and startup continues after model-loading errors. Check the backend log for `Loaded checkpoint:` and `Model pre-loaded successfully!`, then run an analysis on a point inside India from the dashboard. A successful satellite prediction also requires available imagery and working access to the external data services. Missing weights can leave the dashboard accessible while predictions return HTTP 503.

### Optional database and cache

PostgreSQL and Redis improve persistence and caching. They are optional for basic local startup: database failures fall back to local prediction history in `evaluation/prediction_history.json`, and Redis failures disable caching. Earlier example history and saved evaluation results are already tracked; viewing them does not prove that a new prediction has run.

If Docker is installed, you can start just these services using the repository configuration:

```powershell
docker compose up -d postgres redis
```

The API defaults match these local services. Configure `DATABASE_URL`, `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD` in the backend terminal if using different services. `.env.example` documents settings, but the Python API and `start.bat` **do not automatically load `.env` or `.env.local`**. Set variables before starting the API. For example:

```powershell
$env:DATABASE_URL = 'postgresql://postgres:hazard_password@localhost:5432/hazard_db'
$env:REDIS_HOST = 'localhost'
$env:REDIS_PORT = '6379'
$env:REDIS_PASSWORD = 'hazard_redis_password'
```

The supplied service passwords are local-development examples. Ollama is optional for assessment report wording; it is not required to start the dashboard or generate standard analysis PDFs. See [output assessment instructions](evaluation/OUTPUT_ASSESSMENT.md).

### Interface demo without a checkpoint

For an interface test using synthetic inputs, start terminal 1 with the following settings instead. This mode can initialize random model weights when no checkpoint exists; results are demonstration output and cannot reproduce trained-model predictions. Satellite downloads are not needed for this prediction path, but map tiles and weather still need internet.

```powershell
$env:APP_ENV = 'development'
$env:INFERENCE_MODE = 'demo_synthetic'
$env:ALLOW_SYNTHETIC_INFERENCE = 'true'
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Start the frontend separately as above. Synthetic mode is rejected with `APP_ENV=production`. Stop and restart the API with the real-mode variables to return to satellite inference.

### Docker alternative

With Docker and Compose available, restore the checkpoint into the clone first, then run from the repository root:

```powershell
docker compose up -d --build api dashboard
```

Compose also starts PostgreSQL and Redis. Open **http://localhost:8504** for the containerized dashboard and **http://localhost:8000/docs** for the API. The current backend Dockerfile uses a CUDA-based PyTorch image, so its download is large. Compose bind-mounts the repository at `/app`, making the restored `checkpoints/` directory available even though `.dockerignore` excludes it from image builds. A standalone image without that mount needs checkpoints supplied separately. These Docker instructions follow the checked-in configuration; a clean Docker build has not been verified as part of this README update.

```powershell
docker compose logs --tail 100 api dashboard
docker compose down
```

`docker compose down` stops the services without deleting their named data volumes.

### Tests, evaluation, and common problems

For test dependencies and the existing test suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

Tests requiring downloaded assets may skip when those assets are absent. For transfer evaluation and chart exports, also install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

Restore/reacquire data before rerunning data-based evaluations. Follow the [dataset acquisition and review instructions](dataset/README.md) and [expanded transfer reproduction protocol](evaluation/real_transfer_expanded/PROTOCOL.md). Saved metrics can be viewed without restoring raw imagery; reproducing them requires the recorded checkpoint, source assets, and evaluation configuration. Running the ordinary training command with missing datasets does not restore those evaluated weights.

| Symptom | Action |
| --- | --- |
| `No checkpoint found` or prediction HTTP 503 | Restore the intended trained `.pt` file; confirm its configuration matches and restart the API. Inspect backend logs for the underlying error. |
| Checkpoint mismatch | Use the model configuration paired with the weights and remove unintended checkpoint files from the serving folder. |
| Dashboard opens but API calls fail | Start the backend from the repository root and inspect its log. The default frontend proxy expects port `8000`. |
| Port `8503` already in use | Stop the previous frontend process; the Vite development server requires that exact port by default. |
| Database or Redis connection warnings | Start the optional services or correct their environment variables; basic local operation can continue without them. |
| Satellite prediction times out or imagery is unavailable | Check network access and backend logs; try another point/time availability. Real mode does not silently substitute synthetic predictions. |
| Python package installation fails | Confirm 64-bit Python, upgrade pip, and retain the actual package error. Package ranges are not a reproducible Python lockfile. |
| Old PDFs are missing | Copy `reports/analyses/` from the old machine; that folder is excluded from Git. |

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
