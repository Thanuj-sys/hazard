# Validation, disaster vulnerability assessment, and report generation

The frontend navigation now contains **Assessment & report**. This is a separate
three-stage workflow using a model output. It does not rerun prediction, train a
model, download event data, or turn the smoke metrics report into ground truth.
No extra Python dependencies are required beyond the existing requirements.

## Use the page

1. Start your backend and frontend using the existing commands.
2. Run a point prediction in Situation map. Select Assessment & report, then
   Use latest prediction. Alternatively upload a saved `/api/v1/predict-point`
   response JSON. `smoke_metrics_report.json` is an evaluation report, not a
   prediction response, and is not accepted as an assessment input.
3. Optionally upload a WGS84 GeoJSON exposure layer and enter its source name
   and version/URL/annotation reference. Use a regional subset: at most 5000
   features and approximately 3 MB total input.
4. Select Validate, assess & generate report.
5. Download map PNG/SVG, report TXT, complete JSON, or printable HTML. Open the
   downloaded HTML and print/save as PDF. The HTML contains the same map and
   evidence table and requires no remote assets.

## 1. Output validation

Checks hazard label, probability range and sum, coordinates, radius, reported
area, source, synthetic-mode flag and acquisition timestamp. Class/probability
mismatch and circle-area disagreement produce warnings. Malformed required
outputs stop exposure calculations. This is **output-integrity validation**,
not independently measured classification accuracy or event verification.
The report explicitly marks prediction correctness as not evaluated.

If a polygon `footprint` is present, it is treated as a user-supplied model
footprint, not a verified event boundary. Otherwise the system constructs a
geodesic circle from the predicted coordinates and radius. Radius-only outputs
are search/assessment regions and must not be described as observed damage.
A positive radius or a polygon is required for spatial assessment.

## 2. Vulnerability / exposure

Intersection counts use the supplied data only. No missing population, facility,
severity, or vulnerability values are filled. With no exposure layer, counts
remain null and the footprint sketch may still be downloaded.

GeoJSON must use longitude/latitude in WGS84, have no legacy CRS declaration,
contain valid two-dimensional geometry, and avoid dateline crossing and regions
beyond 85 degrees latitude. IDs must be unique. Properties:

| Property | Meaning |
|---|---|
| `kind` | population, building, hospital, clinic, healthcare, school, college, road, shelter, infrastructure |
| `population` | Numeric population COUNT for a population polygon; not density |
| `name` | Optional human-readable feature name |
| `vulnerability_class` | Optional source-provided low, medium, high, or critical |

Population features must be polygons; road features must be line geometries.
Overlapping population polygons are rejected to prevent double counting. Population
is summed ONLY for fully contained source units. Partly intersecting population
units are counted separately; their population is not apportioned. The sum is a
subset count and is never reported as total affected population. Source coverage,
reference year, validity and granularity require independent review.

Geometry is intersected in a local Lambert azimuthal equal-area projection.
Search area is computed from that geometry. Clipped road length uses the same
local projection and is approximate. It does not prove road closure/disruption.
Points on the assessment boundary are included. Polygon/road features that only
touch at zero area/length are excluded. The map is a WGS84 longitude/latitude
sketch, north up, not a satellite basemap or a verified damage map.

**A vulnerability score is not fabricated.** Exposure is one input to vulnerability,
not a substitute for fragility/susceptibility. The workflow reports source-provided
vulnerability classes when present, explicitly attributed to the supplied dataset.
A quantitative score remains unavailable until a reviewed scoring model and its
required inputs are supplied. No arbitrary low/high scores are derived from
classifier confidence or radius.

## 3. Reports and optional local LLM

A structured report is always generated, including validation findings, numbered
facts, calculation methods, source references, unavailable information, input
SHA-256, time, and scope. Failed output validation is itself reportable. Reports
are marked as unverified model assessments or generated-input demonstrations.
No disaster occurrence, severity, damage or emergency instruction is invented.

Optional local Ollama rewrites the numbered facts into short cited paragraphs.
Install/start Ollama separately and select a model already installed on your
machine. In the backend PowerShell terminal:

```powershell
ollama list
$env:OLLAMA_MODEL = "REPLACE_WITH_AN_INSTALLED_MODEL_NAME"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_TIMEOUT_SECONDS = "120"
.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

If Ollama is not already running, start it separately with `ollama serve`.
Select Add local Ollama wording on the page. No hosted LLM API or API key is
required. `.env.example` documents settings, but this app does not automatically
load .env/.env.local; set environment variables before starting the backend.

The LLM receives only the numbered facts, not a request to invent disaster
knowledge. The response must contain structured paragraphs with valid fact IDs.
Unknown citations, incomplete or malformed responses are rejected. Only one LLM
request runs at a time per backend process, with a bounded timeout. Loopback
endpoints only; redirects and environment proxy routing are disabled. Timeout or
missing model/service leaves the deterministic report intact.

Citation-ID checking DOES NOT prove semantic correctness. Generated wording
remains explicitly labeled for human review. This feature does not claim measured
hallucination rate or replace expert-approved disaster response guidance.

## API

- `GET /api/v1/assessment/status`: configuration status; does not invoke an LLM.
- `POST /api/v1/assessment/generate`: body with `prediction`, optional `exposure`
  FeatureCollection, optional `source` object (`name`, `reference`), and
  `use_llm` boolean. Returns validation, vulnerability/exposure, report and map SVG.

Assessment results are returned to the browser, not silently written to a dataset
or submitted as independently verified benchmark annotations.

## Verification

Regression tests cover malformed outputs, absent exposure, population intersection
and non-apportionment, overlapping/duplicate units, clipped roads, source records,
HTML escaping, synthetic labeling and local-LLM citation checks. All test geometries
are software fixtures, not real disaster labels.

Run yourself:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_output_assessment.py -q
```

The implementation was edited and reviewed without running inference, the app,
frontend builds, or tests, following the user's instruction. Runtime behavior and
visual layout have not yet been verified by execution.

## Your imported HOT dataset

Use `area_geojson_uid/exposure_ready.geojson` as the exposure upload. It is a
compact, UTF-8 normalized copy of the original `area.geojson`, with all 3341
source geometries retained and stable OSM object IDs. Contributor `uid` is a
user account ID and is not used as a feature ID. Source name/reference prefill
from the prepared file's `assessment_source`. The original export is unchanged.

Source inventory: 2312 building features, 1001 road features, 22 hospital features,
2 clinics, 1 healthcare centre, 1 college, and 2 other infrastructure features.
These are inventory counts, not affected counts or a verified census of sites.
The app also interprets the original HOT/OSM tags directly; the prepared file is
smaller and includes provenance, so it is preferred for uploads.

After uploading the existing prediction JSON and this exposure file, the page
calculates per-type intersecting counts and shares of the supplied inventory,
clipped mapped road length, and unioned mapped-building footprint overlap area.
The HTML/text report records supplied inventory and computed intersections
separately. The map shows only features intersecting the model footprint.
Building area is not floor area, occupancy, damage, or vulnerability severity.
OSM points and polygons can represent the same physical facility; counts are
source-feature counts. Completeness and current condition require review.

There are no population counts, occupied-bed counts, or reviewed fragility classes
in the prepared inventory. Those remain unavailable. This conversion does not
invent population, severity, damage, or a vulnerability score.
`area_geojson_uid/vulnerability_import_summary.json` records source/output hashes
and import counts. No assessment, inference, or tests were executed during this
preparation, following the user's instruction.
