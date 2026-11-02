import ReportsPage from "./ReportsPage";
import AssessmentPage from "./AssessmentPage";
import EvaluationDetails from "./EvaluationDetails";
import { useEffect, useRef, useState } from "react";
import booleanPointInPolygon from "@turf/boolean-point-in-polygon";
import { point } from "@turf/helpers";
import pointOnFeature from "@turf/point-on-feature";
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  ChevronRight,
  CircleAlert,
  Crosshair,
  ExternalLink,
  Layers3,
  LoaderCircle,
  MapPin,
  Menu,
  Mountain,
  Radio,
  Satellite,
  ShieldCheck,
  Waves,
  Wind,
  X,
  Flame,
} from "lucide-react";
import {
  Circle,
  CircleMarker,
  GeoJSON,
  MapContainer,
  Popup,
  ZoomControl,
  useMap,
  useMapEvents,
} from "react-leaflet";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const INDIA_CENTER = [22.7, 79.2];
const HAZARDS = [
  { id: "flood", label: "Flood", icon: Waves, color: "#1687a7" },
  { id: "wildfire", label: "Wildfire", icon: Flame, color: "#de704c" },
  { id: "landslide", label: "Landslide", icon: Mountain, color: "#8170b5" },
];

function MapClickHandler({ onSelect }) {
  useMapEvents({
    click(event) {
      onSelect([event.latlng.lat, event.latlng.lng]);
    },
  });
  return null;
}

function indiaBoundaryBounds(stateCollection) {
  const bounds = { minLatitude: Infinity, minLongitude: Infinity, maxLatitude: -Infinity, maxLongitude: -Infinity };
  function visitCoordinates(value) {
    if (!Array.isArray(value)) return;
    if (typeof value[0] === "number" && typeof value[1] === "number") {
      bounds.minLongitude = Math.min(bounds.minLongitude, value[0]);
      bounds.maxLongitude = Math.max(bounds.maxLongitude, value[0]);
      bounds.minLatitude = Math.min(bounds.minLatitude, value[1]);
      bounds.maxLatitude = Math.max(bounds.maxLatitude, value[1]);
      return;
    }
    value.forEach(visitCoordinates);
  }

  (stateCollection?.features || []).forEach((feature) => visitCoordinates(feature.geometry?.coordinates));
  if (!Number.isFinite(bounds.minLatitude)) return null;
  return [[bounds.minLatitude, bounds.minLongitude], [bounds.maxLatitude, bounds.maxLongitude]];
}

function FitIndiaMap({ stateCollection }) {
  const map = useMap();

  useEffect(() => {
    const bounds = indiaBoundaryBounds(stateCollection);
    if (!bounds) return undefined;

    function fitToIndia() {
      map.setMinZoom(2);
      map.fitBounds(bounds, { padding: [20, 20], animate: false });
      const indiaZoom = map.getZoom();
      map.setMinZoom(Math.max(2, indiaZoom - 1));
      map.setMaxZoom(Math.min(indiaZoom + 5, 12));
    }

    fitToIndia();
    return undefined;
  }, [map, stateCollection]);

  return null;
}

function RegionViewport({ region, onSelectPoint }) {
  const map = useMap();

  useEffect(() => {
    if (!region?.feature) return;
    const bounds = indiaBoundaryBounds({ features: [region.feature] });
    if (!bounds) return;

    const [longitude, latitude] = pointOnFeature(region.feature).geometry.coordinates;
    map.flyToBounds(bounds, {
      padding: [42, 42],
      maxZoom: region.kind === "district" ? 8 : 6,
      duration: 0.55,
    });
    onSelectPoint([latitude, longitude]);
  }, [map, region]);

  return null;
}

function districtInteractions(feature, layer) {
  const properties = feature.properties || {};
  const district = properties.DISTRICT || properties.district || "District";
  const state = properties.ST_NM || properties.state || "India";
  const [longitude, latitude] = pointOnFeature(feature).geometry.coordinates;
  let requestId = 0;
  let weatherSummary = null;
  layer.bindTooltip(`${district}, ${state}`, {
    className: "district-tooltip",
    sticky: true,
    direction: "top",
  });
  layer.on({
    mouseover(event) {
      event.target.setStyle({ color: "#267b65", weight: 1.35, fillOpacity: 0.2 });
      const activeRequest = ++requestId;
      event.target.bindTooltip(`${district}, ${state} · ${weatherSummary || "loading weather…"}`, {
        className: "district-tooltip",
        sticky: true,
        direction: "top",
      }).openTooltip();
      if (weatherSummary) return;

      const query = new URLSearchParams({ latitude: String(latitude), longitude: String(longitude) });
      fetch(`${API_BASE}/weather?${query}`)
        .then((response) => {
          if (!response.ok) throw new Error("Weather unavailable");
          return response.json();
        })
        .then((payload) => {
          weatherSummary = `${formatNumber(payload.temperature_c)}°C ${payload.condition || ""}`.trim();
          if (activeRequest === requestId) {
            event.target.setTooltipContent(`${district}, ${state} · ${weatherSummary}`);
          }
        })
        .catch(() => {
          weatherSummary = "weather unavailable";
          if (activeRequest === requestId) {
            event.target.setTooltipContent(`${district}, ${state} · ${weatherSummary}`);
          }
        });
    },
    mouseout(event) {
      requestId += 1;
      event.target.setStyle({ color: "#a9b8b2", weight: 0.45, fillOpacity: 0.035 });
      event.target.closeTooltip();
    },
  });
}

function stateInteractions(feature, layer) {
  const name = feature.properties?.ST_NM || feature.properties?.name;
  if (name) {
    layer.bindTooltip(name, {
      className: "state-tooltip state-label",
      permanent: true,
      direction: "center",
      interactive: false,
    });
  }
}

function findDistrictAtPoint(districts, latitude, longitude) {
  if (!districts?.features) return null;
  const location = point([longitude, latitude]);
  const match = districts.features.find((feature) => (
    feature.geometry && booleanPointInPolygon(location, feature)
  ));
  if (!match) return null;
  return {
    district: match.properties?.DISTRICT || match.properties?.district || "Unknown district",
    state: match.properties?.ST_NM || match.properties?.state || "India",
  };
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(Number(value))
    ? Number(value).toLocaleString("en-IN", { maximumFractionDigits: digits })
    : "—";
}

function formatAcquisition(value) {
  if (!value) return "Latest available scene";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

async function readJsonResponse(response, fallbackMessage) {
  const body = await response.text();
  let payload = null;

  if (body.trim()) {
    try {
      payload = JSON.parse(body);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const upstreamMessage = payload?.detail || payload?.message || body.trim();
    throw new Error(
      upstreamMessage
        ? `${fallbackMessage} (${response.status}): ${upstreamMessage}`
        : `${fallbackMessage} (${response.status}). Start the FastAPI backend and try again.`,
    );
  }

  if (!payload || typeof payload !== "object") {
    throw new Error(`${fallbackMessage}: the API returned an empty or invalid response.`);
  }

  return payload;
}

const EVALUATION_SECTIONS = [
  {
    id: "classification",
    index: "01",
    title: "Hazard classification",
    description: "Measures event-class predictions against reference hazard labels.",
    metrics: [
      { code: "ACC", name: "Accuracy", unit: "%", direction: "Higher is better", detail: "Share of predictions matching the reference class." },
      { code: "P", name: "Macro precision", unit: "%", direction: "Higher is better", detail: "Per-class precision averaged across classes present in the evaluation batch." },
      { code: "R", name: "Macro recall", unit: "%", direction: "Higher is better", detail: "Per-class recall averaged across classes present in the evaluation batch." },
      { code: "F1", name: "Macro F1", unit: "%", direction: "Higher is better", detail: "Per-class F1 averaged across classes present in the evaluation batch." },
    ],
  },
  {
    id: "geospatial",
    index: "02",
    title: "Geospatial regression",
    description: "Measures localization and scalar impact-radius estimates against independent spatial references.",
    metrics: [
      { code: "KM", name: "Haversine distance", unit: "km", direction: "Lower is better", detail: "Great-circle distance between predicted and reference coordinates." },
      { code: "MAE", name: "Radius absolute error", unit: "km", direction: "Lower is better", detail: "Absolute difference between predicted and reference radius; aggregated as a mean by the runner." },
      { code: "REL", name: "Radius relative error", unit: "%", direction: "Lower is better", detail: "Absolute radius error divided by the positive reference radius." },
    ],
  },
  {
    id: "grounding",
    index: "03",
    title: "Report grounding",
    description: "Checks whether generated claims match reference claims and measured geospatial attributes.",
    metrics: [
      { code: "DAP", name: "Direct Assessment Precision", unit: "%", direction: "Higher is better", detail: "Predicted hazard claims that also appear in the reference claims." },
      { code: "DDR", name: "Direct Discovery Recall", unit: "%", direction: "Higher is better", detail: "Reference hazard claims recovered by the generated report." },
      { code: "FC", name: "Factual correctness", unit: "%", direction: "Higher is better", detail: "Extracted numeric geo-claims within the evaluator's 10% relative tolerance." },
    ],
  },
  {
    id: "response",
    index: "04",
    title: "Response quality",
    description: "Evaluates recommended actions for necessity, coverage, and priority ordering.",
    metrics: [
      { code: "RN", name: "Recovery necessity", unit: "%", direction: "Higher is better", detail: "Share of proposed interventions included in the required reference actions." },
      { code: "SC", name: "Strategic completeness", unit: "%", direction: "Higher is better", detail: "Share of reference checklist actions covered by proposed interventions." },
      { code: "APP", name: "Action priority precision", unit: "0–1", direction: "Higher is better", detail: "Kendall rank agreement on overlapping actions, normalized to the 0–1 range." },
    ],
  },
  {
    id: "composite",
    index: "05",
    title: "Composite reasoning score",
    description: "A compact summary of report and response quality; it is not an overall model accuracy score.",
    metrics: [
      { code: "AVG", name: "Average reasoning score", unit: "0–1", direction: "Higher is better", detail: "Unweighted mean of DAP, DDR, factual correctness, recovery necessity, strategic completeness, and action priority precision." },
    ],
  },
  {
    id: "segmentation",
    index: "06",
    title: "Spatial footprint segmentation",
    description: "Measures pixel-level hazard footprints when a checkpoint contains trained segmentation-head weights.",
    metrics: [
      { code: "PA", name: "Pixel accuracy", unit: "%", direction: "Higher is better", detail: "Share of valid footprint pixels classified correctly." },
      { code: "IoU", name: "Intersection over Union", unit: "0–1", direction: "Higher is better", detail: "Overlap between the predicted and reference hazard footprint." },
      { code: "DICE", name: "Dice score", unit: "0–1", direction: "Higher is better", detail: "F1-style overlap score for the predicted footprint." },
    ],
  },
];

const EVALUATION_METRIC_KEYS = {
  ACC: "accuracy",
  P: "precision",
  R: "recall",
  F1: "f1",
  KM: "haversine_dist_km",
  MAE: "radius_abs_err_km",
  REL: "radius_rel_err",
  DAP: "dap",
  DDR: "ddr",
  FC: "factual_correctness",
  RN: "recovery_necessity",
  SC: "strategic_completeness",
  APP: "action_priority_precision",
  AVG: "average_score",
  PA: "pixel_accuracy",
  IoU: "iou",
  DICE: "dice",
};

function formatEvaluationValue(value, unit) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (unit === "%") return `${(value * 100).toFixed(1)}`;
  if (unit === "km") return value.toFixed(2);
  return value.toFixed(3);
}

function getUnsupervisedDiagnostics(prediction) {
  const scores = Object.entries(prediction?.hazard_probabilities || {})
    .map(([hazard, probability]) => ({ hazard, probability: Number(probability) }))
    .filter(({ probability }) => Number.isFinite(probability) && probability >= 0);
  const total = scores.reduce((sum, score) => sum + score.probability, 0);
  if (!scores.length || total <= 0) return null;

  const normalized = scores
    .map((score) => ({ ...score, probability: score.probability / total }))
    .sort((left, right) => right.probability - left.probability);
  const entropy = -normalized.reduce(
    (sum, score) => sum + (score.probability > 0 ? score.probability * Math.log(score.probability) : 0),
    0,
  );
  const maximumEntropy = normalized.length > 1 ? Math.log(normalized.length) : 1;

  return {
    predictedHazard: prediction.hazard_type || normalized[0].hazard,
    confidence: normalized[0].probability,
    margin: normalized[0].probability - (normalized[1]?.probability || 0),
    normalizedEntropy: entropy / maximumEntropy,
    probabilities: normalized,
  };
}

function EvaluationPage({ data, loading, error, currentPrediction, history }) {
  const [scope, setScope] = useState("overall");
  const metricCount = EVALUATION_SECTIONS.reduce((count, section) => count + section.metrics.length, 0);
  const resultMetrics = data?.results_by_hazard?.[scope] || {};
  const sampleCount = data?.sample_counts?.[scope];
  const unsupervised = getUnsupervisedDiagnostics(currentPrediction);
  const serviceStatus = loading
    ? "Loading evaluation status…"
    : error
      ? "Evaluation service unavailable"
      : data?.results_available
        ? "Saved evaluation report loaded"
        : "No evaluation report loaded";

  return (
    <div className="evaluation-page">
      <section className="evaluation-hero">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> RESEARCH / EVALUATION WORKSPACE</div>
          <h1>Measure what the model gets right.</h1>
          <p>Metric definitions from the project evaluator, with benchmark status reported separately from implementation status.</p>
        </div>
        <div className="evaluation-run-status">
          <span className={`status-pip ${data?.results_available ? "status-pip-ready" : ""}`} />
          <div><strong>{serviceStatus}</strong><small>{error || data?.detail || "Scores are shown only when returned by FastAPI."}</small></div>
        </div>
      </section>

      <section className="evaluation-summary" aria-label="Evaluation status">
        <div><span>IMPLEMENTED METRICS</span><strong>{String(metricCount).padStart(2, "0")}</strong><small>Defined in evaluation/metrics.py</small></div>
        <div><span>EVALUATED SAMPLES</span><strong>{Number.isFinite(sampleCount) ? sampleCount : "—"}</strong><small>{scope === "overall" ? "Overall test cases" : `${scope} test cases`}</small></div>
        <div><span>HAZARD BREAKDOWN</span><strong>3</strong><small>Flood · Wildfire · Landslide</small></div>
        <div><span>VALIDATION STATUS</span><strong className={data?.results_available ? "summary-live" : "summary-pending"}>{data?.results_available ? "Report loaded" : "Pending"}</strong><small>Requires independent labels</small></div>
      </section>

      <section className="evaluation-results-note">
        <div className="results-note-icon"><CircleAlert size={18} /></div>
        <div><strong>{error ? "Evaluation data could not be fetched." : data?.results_available ? "Showing metrics from the saved backend report." : "Metrics are implemented; benchmark values are not available."}</strong><p>{error || data?.detail || "The evaluation runner aggregates supplied prediction/reference test cases per hazard and overall. Scores are not estimated by this page."}</p></div>
        <span className="results-note-badge">{error ? "OFFLINE" : loading ? "LOADING" : data?.results_available ? "BACKEND" : "NOT RUN"}</span>
      </section>
      {data?.results_available && data.metadata?.evaluation_kind && <section className="evaluation-provenance">
        <div><span className="eyebrow compact">REPORT PROVENANCE</span><strong>{data.metadata.evaluation_kind.replaceAll("_", " ")}</strong></div>
        <p>{data.metadata.ground_truth_status}. This report contains {data.sample_counts?.overall || 0} labeled sample(s); it is not a historical event benchmark.</p>
        <small>{data.metadata.segmentation_note} {data.metadata.geospatial_regression_note}</small>
      </section>}
      <EvaluationDetails data={data} />
      <section className="prediction-history">
        <div className="unsupervised-heading">
          <div>
            <div className="eyebrow compact">PERSISTED ANALYSES / LIVE USE</div>
            <h2>Prediction history</h2>
            <p>Each successful point analysis is retained for review. These records describe model output and do not become accuracy scores without verified reference labels.</p>
          </div>
          <span className="unsupervised-status has-sample">{history?.count || 0} STORED</span>
        </div>
        {history?.predictions?.length ? <div className="history-table">
          {history.predictions.slice(0, 8).map((item, index) => (
            <article key={`${item.created_at}-${index}`}>
              <div><strong>{item.hazard_type}</strong><small>{item.created_at ? new Date(item.created_at).toLocaleString("en-IN") : "Unknown time"}</small></div>
              <span>{Number(item.latitude).toFixed(3)}, {Number(item.longitude).toFixed(3)}</span>
              <span>{Number(item.impact_radius_km).toFixed(1)} km radius</span>
              <small>{item.storage === "local_json_fallback" ? "Local fallback storage" : "Database storage"}</small>
            </article>
          ))}
        </div> : <div className="unsupervised-empty"><Crosshair size={17} /><span>Run a successful point analysis from the Situation map to start collecting prediction history.</span></div>}
      </section>

      <section className="unsupervised-section">
        <div className="unsupervised-heading">
          <div>
            <div className="eyebrow compact">LABEL-FREE / LATEST CASE</div>
            <h2>Unsupervised output diagnostics</h2>
            <p>Computed from the most recent FastAPI prediction probabilities. These describe confidence and uncertainty; they do not measure correctness.</p>
          </div>
          <span className={`unsupervised-status ${unsupervised ? "has-sample" : ""}`}>{unsupervised ? "LIVE MODEL OUTPUT" : "AWAITING PREDICTION"}</span>
        </div>
        {unsupervised ? <>
          <div className="unsupervised-cards">
            <article><span>TOP PREDICTION</span><strong>{unsupervised.predictedHazard}</strong><small>Unlabeled inference result</small></article>
            <article><span>TOP-CLASS CONFIDENCE</span><strong>{formatEvaluationValue(unsupervised.confidence, "%")}<small>%</small></strong><small>Largest normalized class probability</small></article>
            <article><span>TOP-TWO MARGIN</span><strong>{formatEvaluationValue(unsupervised.margin, "%")}<small> pp</small></strong><small>Top probability minus runner-up</small></article>
            <article><span>NORMALIZED ENTROPY</span><strong>{formatEvaluationValue(unsupervised.normalizedEntropy, "0–1")}<small> / 1</small></strong><small>Lower means a more concentrated distribution</small></article>
          </div>
          <div className="unsupervised-distribution">
            <div className="distribution-title"><strong>Class probability distribution</strong><span>Latest point prediction only</span></div>
            {unsupervised.probabilities.map(({ hazard, probability }) => (
              <div className="distribution-row" key={hazard}>
                <span>{HAZARDS.find((item) => item.id === hazard)?.label || hazard}</span>
                <div><i style={{ width: `${probability * 100}%`, background: HAZARDS.find((item) => item.id === hazard)?.color || "#829087" }} /></div>
                <strong>{formatEvaluationValue(probability, "%")}%</strong>
              </div>
            ))}
          </div>
        </> : <div className="unsupervised-empty"><Crosshair size={17} /><span>Run a point prediction from the Situation map, then return here to inspect its unlabeled probability diagnostics.</span></div>}
      </section>

      <div className="evaluation-content">
        <div className="metric-sections">
          <div className="evaluation-scope" role="group" aria-label="Evaluation result scope">
            <span>RESULT SCOPE</span>
            {["overall", "flood", "wildfire", "landslide"].map((value) => (
              <button key={value} type="button" className={scope === value ? "selected" : ""} aria-pressed={scope === value} onClick={() => setScope(value)}>{value === "overall" ? "Overall" : value}</button>
            ))}
          </div>
          {EVALUATION_SECTIONS.map((section) => (
            <section className="metric-section" key={section.id} aria-labelledby={`metric-${section.id}`}>
              <div className="metric-section-heading">
                <span className="metric-section-index">{section.index}</span>
                <div><h2 id={`metric-${section.id}`}>{section.title}</h2><p>{section.description}</p></div>
              </div>
              <div className="metric-cards">
                {section.metrics.map((metric) => (
                  <article className="metric-card" key={metric.code}>
                    <div className="metric-card-top"><span>{metric.code}</span><small>{metric.direction}</small></div>
                    <h3>{metric.name}</h3>
                    <p>{metric.detail}</p>
                    <div className="metric-card-value"><strong>{formatEvaluationValue(resultMetrics[EVALUATION_METRIC_KEYS[metric.code]], metric.unit)}</strong><span>{metric.unit}</span><small>{EVALUATION_METRIC_KEYS[metric.code] in resultMetrics ? "Backend result" : "Not measured"}</small></div>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>

        <aside className="research-notes">
          <div className="research-note-heading"><span className="eyebrow compact">BENCHMARK GATE</span><ShieldCheck size={17} /></div>
          <h2>Ground truth first.</h2>
          <p>Historical evaluation requires independent, machine-readable reference data. Official event reports or map-only PDFs do not qualify as quantitative masks.</p>
          <div className="readiness-status"><span className="readiness-dot" /><div><strong>Wayanad L1</strong><small>not_ready · map reference only</small></div></div>
          <div className="research-divider" />
          <span className="eyebrow compact">FOR A REPRODUCIBLE REPORT</span>
          <ul>
            <li>Report sample counts and class balance per hazard.</li>
            <li>Split by event or geography to avoid spatial leakage.</li>
            <li>Record source, dates, CRS, resolution, and label provenance.</li>
            <li>Include per-hazard scores and the aggregation method.</li>
          </ul>
          <div className="evaluation-implementation"><Activity size={15} /><span>API: <code>GET /api/v1/evaluation/metrics</code></span></div>
        </aside>
      </div>

      <footer className="evaluation-footer"><span>MultiHazard Predictor / Evaluation protocol</span><span>Blank values mean not measured, not zero.</span></footer>
    </div>
  );
}

function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [activePage, setActivePage] = useState("map");
  const [evaluationData, setEvaluationData] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [predictionHistory, setPredictionHistory] = useState(null);
  const [monitorStatus, setMonitorStatus] = useState(null);
  const [districtData, setDistrictData] = useState(null);
  const [stateData, setStateData] = useState(null);
  const [regionOptions, setRegionOptions] = useState([]);
  const [regionSearch, setRegionSearch] = useState("");
  const [regionMenuOpen, setRegionMenuOpen] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [showDistricts, setShowDistricts] = useState(true);
  const [showStates, setShowStates] = useState(true);
  const [selectedPoint, setSelectedPoint] = useState(INDIA_CENTER);
  const [latitude, setLatitude] = useState(INDIA_CENTER[0].toFixed(4));
  const [longitude, setLongitude] = useState(INDIA_CENTER[1].toFixed(4));
  const [prediction, setPrediction] = useState(null);
  const [savingPdf, setSavingPdf] = useState(false);
  const [pdfError, setPdfError] = useState("");
  const [weather, setWeather] = useState(null);
  const [weatherError, setWeatherError] = useState("");
  const [selectedLens, setSelectedLens] = useState("all");
  const selectedLensRef = useRef("all");
  const pdfSaveVersion = useRef(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [infoPanel, setInfoPanel] = useState(null);

  useEffect(() => {
    let active = true;
    async function loadBoundaries() {
      try {
        const [districts, states] = await Promise.all([
          fetch("/data/boundaries/india_districts.geojson"),
          fetch("/data/boundaries/india_states.geojson"),
        ]);
        if (!districts.ok || !states.ok) throw new Error("Boundary files could not be loaded.");
        const [districtJson, stateJson] = await Promise.all([districts.json(), states.json()]);
        const stateRegions = (stateJson.features || []).map((feature) => ({
          kind: "state",
          label: feature.properties?.ST_NM || feature.properties?.name,
          feature,
        }));
        const districtRegions = (districtJson.features || []).map((feature) => ({
          kind: "district",
          label: `${feature.properties?.DISTRICT || feature.properties?.district || "District"}, ${feature.properties?.ST_NM || feature.properties?.state || "India"}`,
          feature,
        }));
        if (active) {
          setDistrictData(districtJson);
          setStateData(stateJson);
          setRegionOptions([...stateRegions, ...districtRegions].filter((region) => region.label));
        }
      } catch {
        if (active) setError("Administrative map boundaries are unavailable. Basemap interaction is still enabled.");
      }
    }
    void loadBoundaries();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (activePage !== "evaluation") return undefined;

    const controller = new AbortController();
    setEvaluationLoading(true);
    setEvaluationError("");
    Promise.all([
      fetch(`${API_BASE}/evaluation/metrics`, { signal: controller.signal }).then((response) => readJsonResponse(response, "Could not load evaluation metrics")),
      fetch(`${API_BASE}/prediction-history`, { signal: controller.signal }).then((response) => readJsonResponse(response, "Could not load prediction history")),
    ])
      .then(([metrics, history]) => {
        setEvaluationData(metrics);
        setPredictionHistory(history);
      })
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setEvaluationError(requestError.message || "Could not load evaluation metrics.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setEvaluationLoading(false);
      });

    return () => controller.abort();
  }, [activePage]);

  useEffect(() => {
    let active = true;
    let retryTimer;
    async function refreshStatus() {
      let isOnline = false;
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) throw new Error("API health check failed.");
        isOnline = true;
        if (active) setApiStatus("online");
      } catch {
        if (active) setApiStatus("offline");
      }
      if (isOnline) {
        try {
          const response = await fetch(`${API_BASE}/monitor/status`);
          if (response.ok && active) setMonitorStatus(await response.json());
        } catch {
          if (active) setMonitorStatus(null);
        }
      } else if (active) {
        setMonitorStatus(null);
      }
      if (active) retryTimer = window.setTimeout(refreshStatus, isOnline ? 30000 : 3000);
    }
    void refreshStatus();
    return () => {
      active = false;
      window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    if (apiStatus !== "online") {
      setWeather(null);
      setWeatherError("Waiting for FastAPI connection");
      return undefined;
    }
    const controller = new AbortController();
    setWeather(null);
    setWeatherError("");
    const timer = window.setTimeout(async () => {
      try {
        const query = new URLSearchParams({ latitude: String(selectedPoint[0]), longitude: String(selectedPoint[1]) });
        const response = await fetch(`${API_BASE}/weather?${query}`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Live weather is unavailable.");
        setWeather(payload);
      } catch (requestError) {
        if (requestError.name !== "AbortError") setWeatherError("Weather temporarily unavailable");
      }
    }, 350);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [selectedPoint, apiStatus]);

  function updateSelectedPoint(point) {
    setSelectedPoint(point);
    setLatitude(point[0].toFixed(4));
    setLongitude(point[1].toFixed(4));
    setError("");
  }

  function chooseRegion(region) {
    setSelectedRegion(region);
    setRegionSearch(region.label);
    setRegionMenuOpen(false);
  }

  const matchingRegions = regionOptions
    .filter((region) => region.label.toLowerCase().includes(regionSearch.trim().toLowerCase()))
    .slice(0, 8);

  async function requestPointPrediction(lat, lon) {
    const area = findDistrictAtPoint(districtData, lat, lon);
    if (districtData && !area) {
      setError("Choose a point inside a mapped Indian district.");
      return;
    }
    setLoading(true);
    setError("");
    setPrediction(null);
    pdfSaveVersion.current += 1;
    setSavingPdf(false);
    setPdfError("");
    try {
      const response = await fetch(`${API_BASE}/predict-point`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ latitude: lat, longitude: lon, selected_hazard: selectedLensRef.current }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The prediction request could not be completed.");
      setPrediction(payload);
      if (payload.selected_hazard !== selectedLensRef.current) {
        await savePdfForLens(selectedLensRef.current, payload);
      }
    } catch (requestError) {
      setError(requestError.message || "The prediction service is unavailable.");
    } finally {
      setLoading(false);
    }
  }

  async function savePdfForLens(lens, current = prediction) {
    if (!current) return;
    const version = ++pdfSaveVersion.current;
    setSavingPdf(true);
    setPdfError("");
    try {
      const response = await fetch(`${API_BASE}/analysis-reports`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...current, selected_hazard: lens }),
      });
      const report = await readJsonResponse(response, "Could not save PDF locally");
      if (version === pdfSaveVersion.current) {
        setPrediction((latest) => latest === current ? { ...latest, selected_hazard: lens, pdf_report: report } : latest);
      }
    } catch (saveError) {
      if (version === pdfSaveVersion.current) setPdfError(saveError.message || "Could not save PDF locally. Please retry.");
    } finally {
      if (version === pdfSaveVersion.current) setSavingPdf(false);
    }
  }

  function retrySavePdf() {
    if (!savingPdf) void savePdfForLens(selectedLensRef.current);
  }

  function chooseHazardLens(lens) {
    if (selectedLensRef.current === lens) return;
    selectedLensRef.current = lens;
    setSelectedLens(lens);
    if (prediction) void savePdfForLens(lens);
  }

  function selectMapPoint(coordinates) {
    updateSelectedPoint(coordinates);
    void requestPointPrediction(coordinates[0], coordinates[1]);
  }

  async function analyzePoint(event) {
    event.preventDefault();
    const lat = Number(latitude);
    const lon = Number(longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setError("Enter a valid latitude and longitude.");
      return;
    }
    updateSelectedPoint([lat, lon]);
    await requestPointPrediction(lat, lon);
  }

  const activeHazard = prediction?.hazard_type?.toLowerCase();
  const predictionColor = HAZARDS.find((hazard) => hazard.id === activeHazard)?.color || "#267b65";
  const displayedHazard = selectedLens === "all" ? activeHazard : selectedLens;
  const displayedHazardInfo = HAZARDS.find((hazard) => hazard.id === displayedHazard);
  const confidenceHazard = selectedLens === "all" ? activeHazard : selectedLens;
  const confidence = prediction?.hazard_probabilities?.[confidenceHazard]
    ?? prediction?.hazard_probabilities?.[activeHazard]
    ?? 0;
  const confidenceLabel = selectedLens === "all"
    ? `${prediction?.hazard_type || "Primary"} confidence`
    : `${HAZARDS.find((hazard) => hazard.id === selectedLens)?.label || "Selected"} confidence`;
  const selectedArea = findDistrictAtPoint(districtData, selectedPoint[0], selectedPoint[1]);
  const monitorLabel = monitorStatus?.enabled
    ? (monitorStatus.status || "starting")
    : "monitor off";

  return (
    <div className="site-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="MultiHazard Predictor home">
          <span className="brand-mark"><Activity size={20} strokeWidth={2.3} /></span>
          <span className="brand-name">MultiHazard Predictor<small>HAZARD INTELLIGENCE</small></span>
        </a>
        <button className="mobile-menu" type="button" onClick={() => setMobileNavOpen(!mobileNavOpen)} aria-label="Toggle navigation">
          {mobileNavOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
        <nav className={`main-nav ${mobileNavOpen ? "is-open" : ""}`} aria-label="Main navigation">
          <button className={`nav-link ${activePage === "map" ? "active" : ""}`} type="button" onClick={() => { setActivePage("map"); setMobileNavOpen(false); }}>Situation map</button>
          <button className={`nav-link ${activePage === "reports" ? "active" : ""}`} type="button" onClick={() => { setActivePage("reports"); setMobileNavOpen(false); }}>View reports</button>
        </nav>
        <div className="topbar-right">
          <span className={`api-indicator ${apiStatus}`}><i />{apiStatus === "checking" ? "Connecting" : apiStatus === "online" ? "API operational" : "API offline · retrying"}</span>
          <a className="docs-link" href="http://localhost:8000/docs" target="_blank" rel="noreferrer" aria-label="Open API documentation">
            <span>API docs</span><ExternalLink size={14} />
          </a>
        </div>
      </header>
      {activePage === "map" ? <main id="top" className="app-main">
        <section className="dashboard-layout" id="map" aria-label="India hazard intelligence dashboard">
          <div className="map-column">
            <div className="map-toolbar">
              <div className="map-heading-copy">
                <span className="eyebrow compact">LIVE FIELD VIEW <span> / </span> INDIA</span>
                <h1>Situation map</h1>
                <p><MapPin size={13} /> {selectedPoint[0].toFixed(4)}° N, {selectedPoint[1].toFixed(4)}° E</p>
              </div>
              <div className="map-controls">
                <div className="region-picker">
                  <label htmlFor="region-search">Go to an Indian region</label>
                  <input
                    id="region-search"
                    type="search"
                    value={regionSearch}
                    placeholder="Search state or district"
                    autoComplete="off"
                    aria-expanded={regionMenuOpen && matchingRegions.length > 0}
                    aria-controls="region-suggestions"
                    onFocus={() => setRegionMenuOpen(true)}
                    onBlur={() => window.setTimeout(() => setRegionMenuOpen(false), 120)}
                    onChange={(event) => {
                      setRegionSearch(event.target.value);
                      setRegionMenuOpen(true);
                      if (selectedRegion && event.target.value !== selectedRegion.label) setSelectedRegion(null);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && matchingRegions[0]) {
                        event.preventDefault();
                        chooseRegion(matchingRegions[0]);
                      }
                      if (event.key === "Escape") setRegionMenuOpen(false);
                    }}
                  />
                  {regionMenuOpen && matchingRegions.length > 0 && <div className="region-suggestions" id="region-suggestions" role="listbox" aria-label="Indian regions">
                    {matchingRegions.map((region) => (
                      <button type="button" key={`${region.kind}:${region.label}`} role="option" aria-selected={selectedRegion?.label === region.label} onMouseDown={(event) => event.preventDefault()} onClick={() => chooseRegion(region)}>
                        <span>{region.label}</span><small>{region.kind}</small>
                      </button>
                    ))}
                  </div>}
                </div>
                <div className="layer-toggles" aria-label="Map layers">
                  <span className="layer-label"><Layers3 size={14} /> Layers</span>
                  <label><input type="checkbox" checked={showStates} onChange={(event) => setShowStates(event.target.checked)} /><span>States</span></label>
                  <label><input type="checkbox" checked={showDistricts} onChange={(event) => setShowDistricts(event.target.checked)} /><span>Districts</span></label>
                </div>
              </div>
            </div>
            <div className="map-screen" aria-label="Interactive map of India">
              <MapContainer
                center={INDIA_CENTER}
                zoom={4.7}
                minZoom={2}
                maxZoom={12}
                scrollWheelZoom
                zoomControl={false}
                className="india-map"
              >
                <FitIndiaMap stateCollection={stateData} />
                <RegionViewport region={selectedRegion} onSelectPoint={updateSelectedPoint} />
                <MapClickHandler onSelect={selectMapPoint} />
                {stateData && <GeoJSON data={stateData} style={{ color: "#476954", weight: 0.55, fillColor: "#e4edda", fillOpacity: 1 }} interactive={false} />}
                {showStates && stateData && <GeoJSON data={stateData} style={{ color: "#577863", weight: 1.1, fillOpacity: 0 }} onEachFeature={stateInteractions} />}
                {showDistricts && districtData && <GeoJSON data={districtData} style={{ color: "#9aa995", weight: 0.45, fillOpacity: 0.02 }} onEachFeature={districtInteractions} />}
                <CircleMarker center={selectedPoint} radius={7} pathOptions={{ color: "#ffffff", weight: 3, fillColor: "#e56e4d", fillOpacity: 1 }}>
                  <Popup>{`Selected point: ${selectedPoint[0].toFixed(4)}, ${selectedPoint[1].toFixed(4)}`}</Popup>
                </CircleMarker>
                {prediction && <>
                  <Circle center={prediction.predicted_coordinates} radius={prediction.radius_km * 1000} pathOptions={{ color: predictionColor, weight: 2, fillColor: predictionColor, fillOpacity: 0.2 }} />
                  <CircleMarker center={prediction.predicted_coordinates} radius={8} pathOptions={{ color: "#ffffff", weight: 3, fillColor: predictionColor, fillOpacity: 1 }}>
                    <Popup>
                      <div className="prediction-popup">
                        <strong>{prediction.hazard_type} impact center</strong>
                        <span>Radius: {formatNumber(prediction.radius_km)} km</span>
                        <span>Affected area: {formatNumber(prediction.affected_area_km2)} km²</span>
                        <span>{prediction.predicted_coordinates[0].toFixed(4)}, {prediction.predicted_coordinates[1].toFixed(4)}</span>
                      </div>
                    </Popup>
                  </CircleMarker>
                </>}
                <ZoomControl position="topright" />
              </MapContainer>
            </div>
            <div className="map-footer">
              <span><span className="hint-pulse" /> Click the map to set an assessment point</span>
              <div className="map-legend">
                {HAZARDS.map(({ id, label, color }) => <span className="legend-item" key={id}><i style={{ background: color }} />{label}</span>)}
                {prediction && <span className="legend-note">Circle shows estimated radius</span>}
              </div>
            </div>
          </div>

          <aside className="assessment-panel dashboard-rail" aria-label="Point assessment and map information">
            <div className="rail-heading">
              <div><span className="eyebrow compact">ASSESSMENT DESK <span> / </span> 01</span><h2>Investigate a point</h2></div>
              <span className="assessment-icon"><Crosshair size={17} /></span>
            </div>
            <p className="panel-description">Select a point on the map or enter coordinates to request a fresh analysis.</p>
            <div className="rail-status-row">
              <div className="weather-card">
                <span className="rail-card-label"><Wind size={12} /> CURRENT WEATHER</span>
                {weather ? <strong>{formatNumber(weather.temperature_c)}° <small>{weather.condition}</small></strong> : <strong className="weather-waiting">{weatherError || "Waiting for location"}</strong>}
                {weather && <span className="weather-footnote">Wind {formatNumber(weather.wind_speed_kmh)} km/h</span>}
              </div>
              <div className="monitor-card">
                <span className="rail-card-label"><Radio size={12} /> MONITOR</span>
                <strong className={monitorStatus?.enabled && monitorStatus.status === "healthy" ? "monitor-good" : "monitor-muted"}>{monitorLabel}</strong>
                <span className="weather-footnote">Background checks</span>
                {monitorStatus?.last_success_at && <span className="weather-footnote">Last success {formatAcquisition(monitorStatus.last_success_at)}</span>}
              </div>
            </div>
            <div className="rail-coordinate"><MapPin size={14} /><span>{selectedArea ? `${selectedArea.district}, ${selectedArea.state}` : "Selected location"}</span><strong>{selectedPoint[0].toFixed(3)}, {selectedPoint[1].toFixed(3)}</strong></div>
            <form onSubmit={analyzePoint}>
              <div className="coordinate-fields">
                <label>Latitude<input aria-label="Latitude" type="number" step="any" min="-90" max="90" value={latitude} onChange={(event) => setLatitude(event.target.value)} /></label>
                <label>Longitude<input aria-label="Longitude" type="number" step="any" min="-180" max="180" value={longitude} onChange={(event) => setLongitude(event.target.value)} /></label>
              </div>
              <button className="analyze-button" type="submit" disabled={loading || apiStatus !== "online"}>
                {loading ? <><LoaderCircle className="spin" size={17} /> Analyzing scenes</> : <>Run point analysis <ArrowUpRight size={17} /></>}
              </button>
            </form>
            <div className="rail-lens-control">
              <span>HAZARD LENS</span>
              <div className="lens-filter" role="group" aria-label="Hazard lens">
                {[{ id: "all", label: "All" }, ...HAZARDS.map(({ id, label }) => ({ id, label }))].map(({ id, label }) => (
                  <button type="button" key={id} className={selectedLens === id ? "selected" : ""} aria-pressed={selectedLens === id} onClick={() => chooseHazardLens(id)}>{label}</button>
                ))}
              </div>
            </div>
            {loading && <div className="loading-note"><span className="loading-bar" />Searching recent satellite coverage. This can take several minutes.</div>}
            {error && <div className="error-note"><CircleAlert size={16} /><span>{error}</span></div>}
            {apiStatus === "offline" && <div className="error-note"><CircleAlert size={16} /><span>FastAPI is not responding on port 8000. Open a terminal at the project root with your Python environment activated, then run: <code>python -m uvicorn api.app:app --host 0.0.0.0 --port 8000</code>. This page retries automatically.</span></div>}

            <div className="rail-section-heading"><span>MODEL OUTPUT</span><span className="result-live"><i /> {prediction ? "RECEIVED" : "READY"}</span></div>
            {prediction ? (
              <div className="results" aria-live="polite">
                <div className="primary-result">
                  <div className="hazard-symbol" style={{ color: displayedHazardInfo?.color || "#267b65" }}>
                    {displayedHazard === "flood" ? <Waves size={19} /> : displayedHazard === "wildfire" ? <Flame size={19} /> : <Mountain size={19} />}
                  </div>
                  <div><span>{selectedLens === "all" ? "PRIMARY SIGNAL" : "SELECTED HAZARD"}</span><strong>{displayedHazardInfo?.label || prediction.hazard_type}</strong><small className="confidence-label">{confidenceLabel}</small></div>
                  <div className="confidence-value">{formatNumber(confidence * 100)}<small>%</small></div>
                </div>
                <div className="probability-list">
                  {HAZARDS.filter(({ id }) => selectedLens === "all" || id === selectedLens).map(({ id, label, color }) => {
                    const probability = prediction.hazard_probabilities?.[id] || 0;
                    return <div className="probability-row" key={id}>
                      <div className="probability-label"><span>{label}</span><b>{formatNumber(probability * 100)}%</b></div>
                      <div className="probability-track"><i style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%`, background: color }} /></div>
                    </div>;
                  })}
                </div>
                <div className="metric-grid">
                  <div className="metric-cell"><span>IMPACT RADIUS · {prediction.hazard_type}</span><strong>{formatNumber(prediction.radius_km)} <small>km</small></strong></div>
                  <div className="metric-cell"><span>CIRCULAR AREA · {prediction.hazard_type}</span><strong>{formatNumber(prediction.affected_area_km2)} <small>km²</small></strong></div>
                </div>
                <div className="source-card">
                  <div className="source-icon"><Satellite size={15} /></div>
                  <div><span>IMAGERY SOURCE</span><strong title={prediction.data_source || "Source not reported"}>{prediction.data_source || "Source not reported"}</strong><small>{formatAcquisition(prediction.source_acquired_at)}{prediction.source_age_hours != null ? ` · ${formatNumber(prediction.source_age_hours)} hours ago` : ""}</small></div>
                </div>
                {prediction.inference_mode === "demo_synthetic" && <div className="error-note"><CircleAlert size={15} /><span>Demo mode: synthetic inputs are not for operational hazard decisions.</span></div>}
                {prediction.language_report && <blockquote className="report-quote">{prediction.language_report}</blockquote>}
                {savingPdf ? (
                  <p className="pdf-save-note" role="status">Saving {selectedLens === "all" ? "all hazards" : selectedLens} PDF locally...</p>
                ) : prediction.pdf_report?.status === "saved" && (prediction.pdf_report.selected_hazard || "all") === selectedLens ? (
                  <div className="saved-pdf">
                    <p className="pdf-save-note" title={prediction.pdf_report.local_path}>{selectedLens === "all" ? "All hazards" : displayedHazardInfo?.label} PDF saved locally and added to View reports.</p>
                    <a className="download-report" href={`${API_BASE}/${prediction.pdf_report.view_path}`} target="_blank" rel="noreferrer">
                      <ExternalLink size={16} /> View PDF <ChevronRight size={15} />
                    </a>
                    <a className="download-report" href={`${API_BASE}/${prediction.pdf_report.download_path}`}>
                      <ArrowDownToLine size={16} /> Download PDF <ChevronRight size={15} />
                    </a>
                  </div>
                ) : (
                  <div>
                    <div className="error-note" role="status"><CircleAlert size={15} /><span>{pdfError || prediction.pdf_report?.detail || "This result has no saved PDF yet. Save it locally below."}</span></div>
                    <button type="button" className="download-report pdf-retry-button" onClick={retrySavePdf} disabled={savingPdf}>
                      {savingPdf ? <LoaderCircle size={16} /> : <ArrowDownToLine size={16} />}{savingPdf ? "Saving PDF locally..." : "Save PDF locally"}
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon"><Crosshair size={20} /></div>
                <strong>Assessment appears here</strong>
                <span>Run an analysis to view hazard signals, estimated impact, and source details.</span>
              </div>
            )}
            <div className="disclaimer"><ShieldCheck size={15} /><span>Decision support only. Not an official warning or a validated event boundary.</span></div>
          </aside>
        </section>
      </main> : activePage === "reports" ? <main id="reports" className="evaluation-main"><ReportsPage apiBase={API_BASE} /></main> : activePage === "assessment" ? <main id="assessment" className="evaluation-main"><AssessmentPage prediction={prediction} apiBase={API_BASE} /></main> : <main id="evaluation" className="evaluation-main">      <EvaluationPage data={evaluationData} loading={evaluationLoading} error={evaluationError} currentPrediction={prediction} history={predictionHistory} /></main>}
      {infoPanel && <div className="info-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setInfoPanel(null); }}>
        <section className="info-dialog" role="dialog" aria-modal="true" aria-labelledby="info-title">
          <button className="info-close" type="button" onClick={() => setInfoPanel(null)} aria-label="Close dialog"><X size={18} /></button>
          {infoPanel === "method" ? <><span className="eyebrow compact">HOW TO READ THIS VIEW</span><h2 id="info-title">Signals, not certainties.</h2><p>Each point request checks for recent Sentinel-1 and Sentinel-2 coverage before model inference. The model returns hazard scores and an estimated circular impact radius; that circle is not a measured footprint polygon.</p><div className="info-callout"><Satellite size={17} /> Scene availability varies by location and date. Requests can take several minutes.</div></> : <><span className="eyebrow compact">DATA & VALIDATION POLICY</span><h2 id="info-title">Historical events stay fail-closed.</h2><p>Official reports confirm an event’s occurrence, but they do not replace local geospatial inputs or independent machine-readable ground truth. Wayanad L1 remains <strong>not_ready</strong> until those materials are available.</p><div className="info-callout"><ShieldCheck size={17} /> Model estimates are decision support only, not official warnings or validated event boundaries.</div></>}
        </section>
      </div>}
    </div>
  );
}

export default App;