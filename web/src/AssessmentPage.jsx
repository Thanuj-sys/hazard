import { useEffect, useRef, useState } from "react";
import "./assessment.css";

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const show = (value, digits = 2) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-IN", { maximumFractionDigits: digits }) : "Unavailable";

export default function AssessmentPage({ prediction, apiBase }) {
  const [selected, setSelected] = useState(prediction || null);
  const [exposure, setExposure] = useState(null);
  const [exposureName, setExposureName] = useState("");
  const [sourceName, setSourceName] = useState("");
  const [sourceReference, setSourceReference] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [configuration, setConfiguration] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const activeRequest = useRef(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/assessment/status`, { signal: controller.signal }).then((response) => response.ok ? response.json() : null).then(setConfiguration).catch(() => {});
    return () => { controller.abort(); activeRequest.current?.abort(); };
  }, [apiBase]);
  async function loadFile(event, kind) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(""); setResult(null);
    try {
      if (file.size > 3 * 1024 * 1024) throw new Error("Use a file smaller than 3 MB. Upload a regional subset.");
      const value = JSON.parse((await file.text()).replace(/^\uFEFF/, ""));
      if (kind === "prediction") {
        if (!value || typeof value !== "object" || !value.hazard_type) throw new Error("Upload a saved prediction response, not an evaluation metrics report. It needs hazard_type, hazard_probabilities, predicted_coordinates, and radius_km.");
        setSelected(value);
      } else {
        if (value?.type !== "FeatureCollection" || !Array.isArray(value.features)) throw new Error("Exposure input must be a WGS84 GeoJSON FeatureCollection.");
        if (value.features.length > 5000) throw new Error("Use at most 5000 exposure features per assessment.");
        setExposure(value); setExposureName(file.name);
        const suppliedSource = value.assessment_source;
        setSourceName(typeof suppliedSource?.name === "string" ? suppliedSource.name : "");
        setSourceReference(typeof suppliedSource?.reference === "string" ? suppliedSource.reference : "");
      }
    } catch (failure) { setError(failure.message); }
    event.target.value = "";
  }
  async function assess() {
    if (!selected) { setError("Run a Situation map prediction first, or upload a saved prediction response."); return; }
    if (exposure && (!sourceName.trim() || !sourceReference.trim())) { setError("Enter a source name and a dataset/evidence reference for the exposure layer."); return; }
    setBusy(true); setError(""); setResult(null);
    const controller = new AbortController(); activeRequest.current = controller;
    const timer = setTimeout(() => controller.abort(), 180000);
    try {
      const response = await fetch(`${apiBase}/assessment/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ prediction: selected, exposure, source: exposure ? { name: sourceName, reference: sourceReference } : null, use_llm: useLlm }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Assessment input was rejected. Check the prediction and exposure formats.");
      setResult(payload);
    } catch (failure) { setError(failure.name === "AbortError" ? "Stopped waiting for the assessment. A local model request may still be finishing; check the backend before retrying." : failure.message); }
    finally { clearTimeout(timer); activeRequest.current = null; setBusy(false); }
  }
  function downloadPng() {
    if (!result?.map_svg) return;
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas"); canvas.width = 1800; canvas.height = 1040;
      const context = canvas.getContext("2d");
      if (!context) { setError("Image export is unavailable in this browser. Download the SVG instead."); return; }
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) { setError("PNG export failed. Download the SVG instead."); return; }
        const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
        anchor.href = url; anchor.download = "assessment_footprint.png"; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }, "image/png");
    };
    image.onerror = () => setError("Could not render PNG. Download the SVG instead.");
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(result.map_svg)}`;
  }
  const validation = result?.validation;
  const vulnerability = result?.vulnerability;
  const report = result?.report;
  return <div className="assessment-page">
    <header className="as-heading"><span className="eyebrow compact">OUTPUT ASSESSMENT / EVIDENCE FIRST</span><h1>Validate. Assess vulnerability. Report.</h1><p>One workflow for checking a model prediction, calculating supplied-data exposure, and producing an auditable assessment.</p></header>
    <div className="as-input-grid"><section className="as-panel"><h2>Prediction input</h2><p>Use the latest Situation map output or upload its saved JSON response.</p><div className="as-controls"><button disabled={!prediction || busy} onClick={() => { setSelected(prediction); setResult(null); setError(""); }}>Use latest prediction</button><label className="as-file">Upload prediction JSON<input type="file" accept=".json,application/json" disabled={busy} onChange={(event) => loadFile(event,"prediction")} /></label></div>
      {selected ? <dl className="as-input-summary"><dt>Model hazard</dt><dd>{selected.hazard_type}</dd><dt>Model radius</dt><dd>{show(selected.radius_km)} km</dd><dt>Coordinates</dt><dd>{Array.isArray(selected.predicted_coordinates) ? selected.predicted_coordinates.join(", ") : "Unavailable"}</dd><dt>Source</dt><dd>{selected.data_source || "Unspecified"}</dd><dt>Mode</dt><dd>{selected.inference_mode || "Unspecified"}</dd></dl> : <p className="as-empty">No prediction selected. The workflow does not create a fake example.</p>}
      {selected?.inference_mode === "demo_synthetic" && <p className="as-warning">Generated-input demonstration. Any report produced here remains a demonstration.</p>}</section>
    <section className="as-panel"><h2>Exposure / vulnerability source</h2><p>Optional WGS84 GeoJSON: population polygons, buildings, hospitals, schools, roads, shelters, or other infrastructure.</p><div className="as-controls"><label className="as-file">Upload exposure GeoJSON<input type="file" accept=".geojson,.json,application/geo+json,application/json" disabled={busy} onChange={(event) => loadFile(event,"exposure")} /></label>{exposure && <button disabled={busy} onClick={() => { setExposure(null); setExposureName(""); setResult(null); }}>Remove layer</button>}</div><p>{exposure ? `${exposureName}: ${exposure.features.length} supplied features` : "No layer selected. Exposure will be reported as unavailable."}</p>
      <label className="as-field">Source name<input value={sourceName} disabled={busy} onChange={(event) => setSourceName(event.target.value)} placeholder="Dataset organization / inventory name" /></label><label className="as-field">Source reference<input value={sourceReference} disabled={busy} onChange={(event) => setSourceReference(event.target.value)} placeholder="Dataset URL, version, or annotation record" /></label><details><summary>Supported feature properties</summary><p><code>kind</code>: population, building, hospital, clinic, healthcare, school, college, road, shelter, infrastructure. HOT/OSM amenity, healthcare, highway and building tags are also interpreted automatically. Population polygons need <code>population</code>. Optional: unique <code>id</code>, <code>name</code>, and source-provided <code>vulnerability_class</code> (low / medium / high / critical). Overlapping population units are rejected.</p></details></section></div>
    <div className="as-run"><label><input type="checkbox" checked={useLlm} disabled={busy} onChange={(event) => setUseLlm(event.target.checked)} /> Add local Ollama wording</label><small>{configuration?.local_llm_configured ? `Configured model: ${configuration.local_llm_model}. Service availability is checked when used.` : "Optional LLM is not configured. Structured reports work without it."}</small><button disabled={busy || !selected} onClick={assess}>{busy ? "Assessing output…" : "Validate, assess & generate report"}</button></div>
    {error && <p role="alert" className="as-error">{error}</p>}
    {!result && !busy && <p className="as-empty">Results appear below after assessment. Output integrity checks are not a measure of prediction accuracy.</p>}
    {result && <>
      <section className="as-panel" id="output-validation"><span className="eyebrow compact">01 / VALIDATION</span><h2>Output validation</h2><div className="as-stats"><article><span>Integrity status</span><strong>{validation.status.replaceAll("_"," ")}</strong></article><article><span>Failures</span><strong>{validation.failure_count}</strong></article><article><span>Warnings</span><strong>{validation.warning_count}</strong></article><article><span>Independent accuracy</span><strong>Not evaluated</strong></article></div><div className="as-table-wrap"><table><thead><tr><th>Check</th><th>Status</th><th>Finding</th></tr></thead><tbody>{validation.checks.map((check) => <tr key={check.name}><th>{check.name.replaceAll("_"," ")}</th><td><span className={`as-badge as-${check.status}`}>{check.status.replaceAll("_"," ")}</span></td><td>{check.message}</td></tr>)}</tbody></table></div></section>
      <section className="as-panel" id="vulnerability"><span className="eyebrow compact">02 / VULNERABILITY ASSESSMENT</span><h2>Exposure and source vulnerability</h2><p><strong>{vulnerability.status.replaceAll("_"," ")}</strong> — {vulnerability.reason || vulnerability.footprint_method}</p><div className="as-stats"><article><span>Intersecting supplied features</span><strong>{show(vulnerability.intersecting_feature_count,0)}</strong></article><article><span>Population in fully contained units</span><strong>{show(vulnerability.population_fully_contained_units,0)}</strong></article><article><span>Partially intersecting population units</span><strong>{show(vulnerability.population_partial_units,0)}</strong></article><article><span>Supplied roads inside footprint</span><strong>{show(vulnerability.road_length_inside_km)} km</strong></article></div>
      <p className="as-warning">Counts represent source features, which may not be unique physical facilities. Overlap is potential exposure, not observed damage. The population figure is a subset sum, not total affected population. Vulnerability score: unavailable without a reviewed susceptibility/fragility model.</p>
      {Object.keys(vulnerability.feature_counts || {}).length > 0 && <div className="as-type-counts">{Object.entries(vulnerability.feature_counts).map(([kind,count]) => <span key={kind}>{kind}: <strong>{count}</strong></span>)}</div>}
      {vulnerability.feature_exposure && <div className="as-table-wrap"><table><thead><tr><th>Source feature type</th><th>Supplied</th><th>Intersects model footprint</th><th>Share of supplied inventory</th></tr></thead><tbody>{Object.entries(vulnerability.feature_exposure).map(([kind,counts]) => <tr key={kind}><th>{kind}</th><td>{counts.supplied}</td><td>{counts.intersecting}</td><td>{show(counts.fraction*100,1)}%</td></tr>)}</tbody></table></div>}
      {typeof vulnerability.mapped_building_overlap_km2 === "number" && <p>Mapped building footprint overlap: <strong>{show(vulnerability.mapped_building_overlap_km2*1000000,0)} m²</strong>. This is intersecting source building area, not damaged floor area.</p>}
      {Object.keys(vulnerability.source_vulnerability_classes || {}).length > 0 && <p>Source-provided vulnerability classes: {Object.entries(vulnerability.source_vulnerability_classes).map(([kind,count]) => `${kind}: ${count}`).join(" · ")}. These classes were not independently verified here.</p>}
      {vulnerability.source && <p>Source: {vulnerability.source.name} · {vulnerability.source.reference}</p>}
      {result.map_svg && <><img className="as-map" src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(result.map_svg)}`} alt="Model search footprint and intersecting supplied exposure features, north up" /><div className="as-controls"><button onClick={downloadPng}>Download map PNG</button><button onClick={() => download("assessment_footprint.svg",result.map_svg,"image/svg+xml")}>Download map SVG</button></div></>}
      {vulnerability.exposed_features?.length > 0 && <details><summary>Intersecting source features ({vulnerability.exposed_features.length})</summary><div className="as-table-wrap"><table><thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Fully contained</th></tr></thead><tbody>{vulnerability.exposed_features.map((feature) => <tr key={feature.id}><td>{feature.id}</td><td>{feature.name}</td><td>{feature.kind}</td><td>{feature.fully_contained ? "Yes" : "No"}</td></tr>)}</tbody></table></div></details>}
      {(vulnerability.limitations || []).map((item) => <p className="as-note" key={item}>{item}</p>)}</section>
      <section className="as-panel" id="report-generation"><span className="eyebrow compact">03 / REPORT GENERATION</span><h2>Assessment report</h2><p>{report.notice}</p><div className="as-controls"><button onClick={() => download("hazard_assessment.txt",report.text,"text/plain")}>Download report TXT</button><button onClick={() => download("hazard_assessment.html",report.html,"text/html")}>Download printable HTML</button><button onClick={() => download("hazard_assessment.json",JSON.stringify(result,null,2),"application/json")}>Download complete JSON</button></div><p className="as-note">Open the downloaded HTML and choose Print → Save as PDF. The report includes the map, sources, validation checks and unavailable fields.</p>
      <h3>Evidence and calculated findings</h3>{report.facts.map((fact) => <article className="as-fact" key={fact.id}><strong>[{fact.id}] {fact.name}</strong><p>{typeof fact.value === "object" ? JSON.stringify(fact.value) : String(fact.value)}</p><small>{fact.basis}</small></article>)}
      <h3>Local LLM wording</h3><p>Status: {report.llm.status.replaceAll("_"," ")}. {report.llm.reason || report.llm.verification}</p>{report.llm.paragraphs.map((paragraph,index) => <p className="as-llm" key={index}>{paragraph.text} <small>[{paragraph.fact_ids.join(", ")}]</small></p>)}
      <h3>Information still unavailable</h3><ul>{report.unknowns.map((item) => <li key={item}>{item}</li>)}</ul><p className="as-note">Generated: {report.created_at} · Scope: {report.scope}<br />Input SHA-256: {report.input_sha256}</p></section>
    </>}
  </div>;
}
