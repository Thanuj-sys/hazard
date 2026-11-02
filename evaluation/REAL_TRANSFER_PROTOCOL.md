# Real public-data transfer test

This directory contains results produced with the existing model checkpoint, not new training.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_real_holdout.py --phase all --per-hazard 30 --seed 20261027
.\.venv\Scripts\python.exe scripts/export_evaluation_figures.py --report evaluation/real_transfer/metrics_report.json --output-dir evaluation/real_transfer/figures --scope overall
```

Data are downloaded under `dataset/real_holdout/` on the repository drive. The manifest records upstream splits, exact URLs, pinned Hugging Face revisions and SHA256 hashes. The acquisition seed is fixed before inference. Published masks establish positive labelled pixels; predictions never supply reference labels. Empty-mask patches and exact duplicates of known local training image files are excluded and listed. The checkpoint stays unchanged; no retraining, threshold optimization or checkpoint search is performed.

## Interpretation

This is a small, positive-only, source-domain transfer diagnostic of the classification branch. It measures whether the checkpoint selects the expected hazard family on real public source images under the documented adapters. It does not validate operational hazard detection, negative-case rejection, mask segmentation, location/radius estimation, vulnerability, damage or reports. Each hazard has a different dataset and input-channel pattern, creating a source/sensor confound. Event and geographic independence are not established.

The selected checkpoint is a bare state dictionary without an archived training manifest. Exact known local training-file duplicates can be checked; all historical training exposure cannot be proven. Therefore these records must not be advertised as a certified unseen benchmark. Input-size positional embeddings are interpolated as in the current deployment path; geographic context is omitted or zero-filled explicitly, and a single date is repeated four times. No genuine temporal-change performance is measured.

## Primary data sources

- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11 — fixed-seed subset of official hand-labelled test CSV. Labels distinguish water, not permanent water from new flooding. This test labels the source family as flood; it does not claim verified inundation.
- HLS Burn Scars: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars — fixed-seed sample of 30 rows from the first 100-row official converted validation shard. The converted `test` split aliases upstream validation; there is no separate independent test split. Complete shard is verified against its published LFS SHA256. Already-reflectance six-band TIFF bytes are preserved without dividing by 10000 again. Burn scars are not active-fire labels. CC-BY-4.0; credit Phillips, Roy, Ankur and Ramachandran, doi:10.57967/hf/0956.
- Landslide4Sense: https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense — fixed-seed subset of official test images and pixel masks. Dataset authors: https://github.com/iarai/Landslide4Sense-2022. For RGB, the author's published per-band mean division is reversed before converting recovered DN to reflectance: https://github.com/iarai/Landslide4Sense-2022/issues/4#issuecomment-1195457420. This physical-unit adapter is not a demonstrated match to the earlier CAS 8-bit RGB training pipeline. Confirm upstream redistribution terms before sharing raw data.

The frozen scientific benchmark `evaluation/metrics_report.json` and earlier smoke results are not overwritten. Read actual `metrics_report.json`, `predictions.jsonl`, exclusions and plots in this directory after completion. Figure warnings identify the limits of this real-data test.
