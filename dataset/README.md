# Multi-hazard evaluation dataset

Status: acquisition candidates only. This is NOT a completed research benchmark.
No candidate is an independent held-out test case until the evidence and review
requirements below pass. Empty split files are intentional. No predictions or
expert judgments have been fabricated.

## Files and commands

Run from the repository root using the project environment:

```powershell
.venv\Scripts\python.exe scripts/build_evaluation_dataset.py init
.venv\Scripts\python.exe scripts/build_evaluation_dataset.py acquire --per-hazard 2
.venv\Scripts\python.exe scripts/build_evaluation_dataset.py validate
.venv\Scripts\python.exe scripts/run_evaluation.py
```

`images/{flood,wildfire,landslide}` contains real downloaded observations.
`raw/` retains source labels that are not eligible hazard masks.
`masks/` contains normalized binary GeoTIFF references where supported.
`validity/` contains aligned 0/1 masks: only 1 is evaluated.
`annotations/cases.jsonl` is the case registry; the adjacent template describes
all requested fields. `metadata/sources.json` catalogs provenance and limitations;
`metadata/evidence/` stores downloaded source documentation and asset listings.
`metadata/validation_report.json` reports counts, integrity errors and readiness
blockers. `predictions/predictions.jsonl` has one separate prediction record per
case. All outputs remain null with reasons until actual inference is run.
`predictions/prediction_template.json` defines the output record.

Raster and raw assets are excluded from Git; metadata, schemas, splits and audit
reports are tracked. Retain downloaded assets separately when sharing this data.
All recorded asset paths are relative to this directory. Per-file SHA-256 hashes
allow integrity checking. Downloads use upstream checksums when available;
partial streaming of the HLS archive cannot verify the entire archive hash.
Re-running acquisition preserves existing case annotations and model outputs.

## Source constraints

- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11): real Sentinel-1
  observations and published human water labels. India is preferred when choosing
  examples. Water labels include permanent water. They are stored as raw labels,
  never silently promoted to flood inundation or event-presence truth. Otsu and
  other weak model-generated labels are not used.
- [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars):
  real HLS observations with MTBS-derived reference scars. Normalization preserves
  0/1 labels; missing labels become 255 and are excluded by the validity mask.
  A burn scar is evidence of past burning; active fire, urgency and severity are
  unavailable. The upstream random split is not accepted as event independence.
- [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022): published imagery
  and annotation HDF5 patches from the IBM-NASA mirror. Patch location, transform,
  acquisition time and event identity are not established by the mirror card.
  No invented CRS or coordinates are assigned. These raw patches stay candidates.
- NRSC Wayanad source links are cataloged for further Indian event acquisition.
  Official map PDFs alone are not aligned evaluation masks.

Check original licenses before redistribution. Missing license information is
recorded as unresolved, rather than assigning a license from the code repository.
No generic response checklist is presented as event-specific expert ground truth.

## Annotation and independent review

Copy `annotations/case_template.json`, retain null values for unavailable fields,
and supply a nonempty `missing_reasons[field]` for each. A raster center is an
observation location, not an event epicenter. The tool never derives an event
radius from an image bounding box. Event area and radius require explicit source
measurements or a reviewed derivation from independently verified footprints;
document whether radius means equivalent-area radius or another definition.

For EVERY populated observation or truth field, add `evidence[field]`:

```json
{
  "source_id": "registered_source_id",
  "reference": "Exact document page, product identifier, or annotation record",
  "method": "source_metadata",
  "status": "verified"
}
```

Methods: `source_metadata`, `published_label`, `documented_derivation`, or
`expert_annotation`. A derivation must describe its inputs, formula and scope in
`reference`; it must not be model-derived truth. Add new authoritative references
to `metadata/sources.json`. Review each fact against the cited evidence before
setting `status` to verified. This validator checks records and integrity; it
cannot authenticate the truth of a curator's assertion by itself.

Expert annotations additionally require distinct `annotator_id` and `reviewer_id`,
`protocol`, and timezone-bearing `reviewed_at`. Priority order always requires this
expert annotation process. Record expert qualifications and approved event-specific
response guidance in the referenced protocol. Record unsafe, unnecessary or
prohibited actions explicitly; do not infer an empty list without review.

`geo_attributes` maps attribute names to records containing numeric `value`,
`unit`, numeric nonnegative `absolute_tolerance`, `source_id`, and `reference`.
Use source-specific tolerances, including for values equal to zero. Verified claims
and action lists use unique canonical strings; maintain the human-readable claim
and exact supporting evidence in the referenced annotation record. Predictions
must use the same canonical identifiers after a documented, independently reviewed
claim/action extraction step. Raw free-text equivalence is not guessed by an LLM.

For each reviewed case:

1. Recover authoritative event dates, acquisition time with timezone, quality,
   true event coordinates, uncertainty, administrative region and ground-truth
   reliability. Record what is unavailable instead of estimating it.
2. Verify hazard-specific target semantics and the image/target/validity CRS,
   transform, dimensions and resolution. `mask_metadata` must match the raster
   header; background=0, target hazard=1, target nodata=255 (or documented source
   nodata), validity=1 for evaluated pixels and 0 otherwise. Do not feed targets
   or annotation evidence into the model.
3. Assign stable `event_id` and `region_id`. All cases sharing either group must
   remain in one split; adjoining imagery should share a region group. The
   validator also rejects cross-split overlapping bboxes and duplicate imagery.
   Review nearby non-overlapping tiles for geographic leakage beyond this check.
4. Add `review` with distinct annotator/reviewer IDs, protocol and review date;
   set `annotation_status` to `reviewed`. Explicitly assign `split` to `train`,
   `validation`, or `test`. For test, document the model's training inventory,
   pretraining exposure limitations, and attest `training_exposure: not_used`.
   Use validation for thresholds and tuning; never inspect test results to tune.
5. Stage assets within this directory, provide asset SHA-256 hashes, and import:
   `python scripts/build_evaluation_dataset.py import --cases reviewed.jsonl`.
   Imports validate before replacing case/split registries.

The recommended benchmark needs at least 100 reviewed TEST cases PER hazard,
multiple events/regions/dates/severities, verified negatives and all metric-specific
annotations. Do not inflate counts by splitting overlapping pixels from one event.
Dataset integrity and benchmark readiness are separate statuses. A valid candidate
registry may correctly have zero eligible test samples.

## Model predictions

Run a frozen model on the held-out input observations. Save real outputs with
`prediction_status: completed`, model version/checkpoint identity and an ISO UTC
inference timestamp. Unknown output branches remain null with explicit reasons.
Hazard probabilities include flood/wildfire/landslide; a calibrated four-class
model may additionally include `none`. Probabilities must sum to one. An existing
three-class model cannot establish non-event performance without a separately
validated abstention/presence mechanism. No fake negative probabilities are added.

`predicted_mask_path` is a separate aligned binary raster. Evaluate the fixed
threshold selected on validation only. The current checkpoint was trained using
repository sample files with generated content; this work does not claim it is a
research-trained model. Prediction records stay `not_run` until actual inference.

## The 17 metrics

All scores report eligible sample counts and total test counts. Missing references
or predictions produce null, not zero. An explicitly empty predicted claim/action
list is a scored failure when a verified nonempty reference exists. Zero-reference
recall, zero-radius relative error, and fewer than two priority actions are
undefined and excluded. Partial failures remain visible in coverage.

| Metric | Definition / reference needed |
|---|---|
| Accuracy | Correct hazard / scored cases; verified `true_hazard` |
| Macro precision | Mean per-class TP/(TP+FP), fixed flood/wildfire/landslide/none |
| Macro recall | Mean per-class TP/(TP+FN), same fixed classes |
| Macro F1 | Mean per-class harmonic precision/recall; undefined class denominator=0 |
| Haversine distance | Great-circle error in km; Earth radius 6371.0088 km |
| Radius absolute error | Absolute predicted minus verified radius, km |
| Radius relative error | Absolute error / strictly positive verified radius |
| Direct Assessment Precision | Exact canonical claim intersection / predicted claims |
| Direct Discovery Recall | Exact canonical claim intersection / verified claims |
| Factual correctness | Numeric claims within per-attribute absolute tolerances / all predicted numeric claims; unsupported claims fail |
| Recovery necessity | Verified necessary, safe predicted actions / all predicted actions |
| Strategic completeness | Verified safe checklist actions recovered / checklist actions |
| Action priority precision | Correctly ordered expert action pairs / all expert pairs plus unknown predicted pairs; missing actions penalized |
| Average reasoning score | Equal mean of preceding six reasoning scores, only when all six are defined for that case |
| Pixel accuracy | Correct valid binary pixels / valid pixels |
| Intersection over Union | Valid foreground intersection / union |
| Dice | Twice valid foreground intersection / sum of foreground sizes |

Geospatial/reasoning/segmentation scores are means across eligible cases, not
pooled pixels. Both-empty foreground IoU/Dice=1; all-invalid masks are errors.
Four-class macro scores include absent classes as zero; use the accompanying case
counts to interpret small subsets. Priority precision is an explicitly defined
pairwise metric, not the legacy evaluator's overlap-only Kendall correlation.
API-compatible legacy metric aliases are included in saved reports; canonical
metric names and coverage keys identify the 17 definitions above.

## Remaining work

The current acquisition is a small source-validation subset, not 300 test cases.
Required work includes independently verified flood footprints, georeferenced
landslide samples, event/region metadata, review of severity/claims, qualified
response annotations, negative cases, a frozen split inventory and actual model
inference. `metadata/validation_report.json` is the machine-readable current
readiness record. Never publish the candidate registry as a completed benchmark.
