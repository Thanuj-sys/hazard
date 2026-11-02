# Expanded transfer-test protocol

Acquisition: 290 published image/annotation pairs (90 Sen1Floods11, 100 HLS burn-scar, 100 Landslide4Sense). Scored: 248 positive-labelled patches (82 / 100 / 66). All 42 exclusions were mask-empty patches; they were not relabelled as hazards or certified non-events. The new-only cohort contains 173 scored patches after removing previously acquired IDs and identical image SHA256 combinations.

The same `checkpoints/model_epoch_9.pt`, input adapters, 64px grid, four repeated frames and zero-context ablation were used. No retraining, threshold fitting or checkpoint selection took place. All 75 repeated baseline cases retained their original predicted class. Six adapter/comparison software tests passed; saved aggregate metrics were independently recomputed with scikit-learn. The original baseline report SHA256 remains `d2013498c70ea1c7f8211fa476e2f5aec0733713251d4ee6bdaf6d450c9764b2`.

## Reproduce

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts/evaluate_real_holdout.py --phase acquire --per-hazard 100 --seed 20261027 --data-root dataset/real_holdout_expanded --output evaluation/real_transfer_expanded/metrics_report.json
.venv\Scripts\python.exe scripts/evaluate_real_holdout.py --phase evaluate --per-hazard 100 --seed 20261027 --data-root dataset/real_holdout_expanded --output evaluation/real_transfer_expanded/metrics_report.json
.venv\Scripts\python.exe scripts/compare_transfer_evaluations.py
.venv\Scripts\python.exe scripts/export_evaluation_figures.py --report evaluation/real_transfer_expanded/metrics_report.json --output-dir evaluation/real_transfer_expanded/figures --scope overall
```

The manifest records pinned revisions, exact URLs and asset SHA256s. Seed 20261027 is fixed before inference. Acquisition is capped at 90 flood test CSV rows and the first 100 HLS converted validation rows. Existing raw downloads were copied to the new folder to avoid unnecessary downloads; original dataset files and manifests were not overwritten.

## Source attribution and limits

- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11 . Uses the official hand-labelled test CSV. Water-source patches do not establish confirmed new inundation.
- HLS Burn Scars: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars . The converted test split aliases official validation. Burn scars are not active fire. CC-BY-4.0; credit Phillips, Roy, Ankur and Ramachandran, DOI 10.57967/hf/0956.
- Landslide4Sense: https://github.com/iarai/Landslide4Sense-2022 and https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense . Uses official test images/annotations and the documented RGB transfer adapter.

This remains a positive-only source-domain transfer diagnostic. Source/sensor confounding, unresolved historical training exposure, absent event-held-out proof, repeated frames and missing geographic context prevent an operational-generalization claim. Source-prefix slices are descriptive rather than proof of independent events. Classification scores do not validate segmentation, location/radius, exposure or language quality.

See `../MODEL_IMPROVEMENT_PLAN.md` before retraining. Any dataset inspected during model development must not subsequently be described as an untouched final test.
