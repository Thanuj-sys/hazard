# Evaluation workflow

Run the independent benchmark with:

```powershell
.venv\Scripts\python.exe scripts/build_evaluation_dataset.py validate
.venv\Scripts\python.exe scripts/run_evaluation.py
```

The dataset contract and annotation protocol are in [dataset/README.md](../dataset/README.md).
The evaluator implements all 17 metrics with per-metric eligibility counts. Missing
verified references or real predictions yield null scores. An empty or candidate-only
dataset is explicitly incomplete. No source imagery, labels or model outputs are
fabricated. Only reviewed test cases enter evaluation.

The saved report remains compatible with `GET /api/v1/evaluation/metrics` through
legacy metric aliases. It records hashes of the case and prediction registries and
contains validation/readiness details. The dashboard displays only finite scores.

The old local `sample_*.tif` files correspond to the random-data generator in
`data/generate_sample_tiffs.py`; they are software fixtures, not verified satellite
observations. The old default evaluation could produce misleading perfect scores
by treating their directory names as class labels. That path is now explicit:

```powershell
.venv\Scripts\python.exe scripts/run_evaluation.py --local-smoke
```

It writes `evaluation/smoke_metrics_report.json`, labels results as generated-sample
diagnostics, and cannot overwrite the canonical benchmark report. It does not
establish classification, event geospatial, reasoning or segmentation validity.
The existing checkpoints do not establish research readiness.

`EvaluationRunner` and `MultiHazardEvaluator` remain available for legacy callers;
use `evaluation.benchmark` for the strict provenance-aware benchmark. Its priority
precision and numeric tolerance definitions are documented in the dataset card.

## Detailed reports, screenshots, and image exports

The Evaluation page now includes confusion matrices (counts or row-normalized),
per-class support and TP/FP/FN/TN, precision/recall/F1/specificity/NPV, balanced
accuracy, weighted/micro F1, MCC, Cohen's kappa, a 95% Wilson accuracy interval,
misclassified cases, metric coverage and per-case error distribution summaries.
Full probability vectors additionally enable one-versus-rest ROC/AUC and PR/AP,
log loss, multiclass Brier score, calibration bins and expected calibration error.
Missing branches remain unavailable; figures never manufacture scores.

Restart the backend and refresh the frontend after updating the code. The API
can derive a confusion matrix from old smoke reports' saved per-sample classes,
but cannot reconstruct full probabilities from a single confidence value. New
smoke runs save full probabilities. Smoke diagnostics remain visibly labeled
as generated data, including in exports. The Wilson interval assumes independent
cases; it is not an event-clustered confidence interval.

The page's Print / Save PDF button prints its detailed evaluation section. Export
class CSV and Export report JSON save the underlying tables and data.

To create PNG/SVG charts and CSV tables from an ALREADY SAVED report, without
running inference or evaluation:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.venv\Scripts\python.exe scripts/export_evaluation_figures.py --report evaluation/smoke_metrics_report.json
```

Open `evaluation/figures/overall/index.html` to view the image gallery. Its PNGs
and SVGs are suitable for slides; CSV tables and `diagnostics.json` retain the
underlying values. For reviewed benchmark results, select
`--report evaluation/metrics_report.json`. Choose `--scope flood`, `wildfire`, or
`landslide` for a hazard subset. Empty-data charts explicitly say NOT EVALUATED.

To run the added tests yourself:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation_diagnostics.py tests/test_benchmark_dataset.py -q
```

## Image explainability

See [EXPLAINABILITY.md](EXPLAINABILITY.md) for gradient-times-input and integrated
gradients, screenshot-ready comparison figures, numerical attributions and
checkpoint provenance. This explains the classifier and does not generate new
benchmark truth or LLM reports. It can use exact prepared inference inputs or an
explicitly labeled generated-sample demonstration.

## Output validation, vulnerability and report generation

The **Assessment & report** frontend page implements the three requested stages.
See [OUTPUT_ASSESSMENT.md](OUTPUT_ASSESSMENT.md) for prediction/GeoJSON inputs,
calculation limits, PNG/SVG/report exports, and optional local Ollama setup.
Output validation is not independently measured prediction accuracy; vulnerability
scores stay unavailable without a reviewed fragility/susceptibility model.
