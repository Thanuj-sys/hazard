import { useState } from "react";
import "./evaluation-details.css";

const display = (value, percent = false) => Number.isFinite(value) ? (percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)) : "Not evaluated";
const title = (value) => value.replaceAll("_", " ");
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function CurvePlot({ curves, mode }) {
  const colors = ["#217b51", "#cd6938", "#466fba", "#94713c"];
  const available = Object.entries(curves || {}).filter(([, curve]) => curve.status === "available");
  if (!available.length) return <p className="ed-empty">Not evaluated. Full class probabilities and both positive and negative references are required.</p>;
  return <><svg className="ed-plot" viewBox="0 0 360 250" role="img" aria-label={mode === "roc" ? "One versus rest ROC curves" : "One versus rest precision recall curves"}>
    {[0, .25, .5, .75, 1].map((v) => <g key={v}><line x1="42" x2="335" y1={208-v*180} y2={208-v*180} stroke="#e3e9e4" /><text x="35" y={212-v*180} textAnchor="end">{v}</text><text x={42+v*293} y="225" textAnchor="middle">{v}</text></g>)}
    {mode === "roc" && <line x1="42" y1="208" x2="335" y2="28" stroke="#9eaaa2" strokeDasharray="4 4" />}
    {available.map(([label, curve], i) => {
      const x = mode === "roc" ? curve.roc.fpr : curve.precision_recall.recall;
      const y = mode === "roc" ? curve.roc.tpr : curve.precision_recall.precision;
      return <polyline key={label} fill="none" stroke={colors[i % colors.length]} strokeWidth="2.5" points={x.map((v, j) => `${42+v*293},${208-y[j]*180}`).join(" ")} />;
    })}
    <text x="188" y="246" textAnchor="middle">{mode === "roc" ? "False positive rate" : "Recall"}</text>
    <text transform="translate(12 120) rotate(-90)" textAnchor="middle">{mode === "roc" ? "True positive rate" : "Precision"}</text>
  </svg><div className="ed-legend">{available.map(([label, curve], i) => <span key={label} style={{ color: colors[i % colors.length] }}>{label}: {mode === "roc" ? "AUC" : "AP"} {display(mode === "roc" ? curve.roc_auc : curve.average_precision)}</span>)}</div></>;
}

export default function EvaluationDetails({ data }) {
  const [scope, setScope] = useState("overall");
  const [normalized, setNormalized] = useState(false);
  const details = data?.detailed_evaluation;
  const classification = details?.classification?.[scope];
  const probability = classification?.probability_diagnostics;
  const perClass = classification?.per_class || [];
  const summary = classification?.summary || {};
  const smoke = details?.provenance === "generated_sample_smoke_test" || data?.metadata?.evaluation_kind === "generated_sample_smoke_test";
  const coverage = data?.metric_coverage?.[scope] || {};
  const distributions = details?.metric_distributions?.[scope] || {};
  const matrix = normalized ? classification?.confusion_matrix_row_normalized : classification?.confusion_matrix;
  const labels = classification?.labels || [];
  const n = classification?.sample_count || 0;
  const csv = () => {
    const keys = ["class", "support", "predicted_count", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "specificity", "negative_predictive_value"];
    const quote = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
    const rows = [keys, ...perClass.map((row) => keys.map((key) => row[key]))];
    download(`${smoke ? "SMOKE_ONLY" : "evaluation"}_${scope}_classes.csv`, rows.map((row) => row.map(quote).join(",")).join("\r\n"), "text/csv");
  };
  return <section className="evaluation-details">
    <div className="ed-heading"><div><span className="eyebrow compact">DETAILED EVALUATION / AUDITABLE RESULTS</span><h2>Classification, uncertainty & error analysis</h2><p>Every chart uses saved reference/prediction pairs. Missing measurements remain unavailable.</p></div><div className="ed-actions"><button onClick={() => window.print()}>Print / Save PDF</button><button disabled={!n} onClick={csv}>Export class CSV</button><button disabled={!data} onClick={() => download("evaluation_report.json", JSON.stringify(data, null, 2), "application/json")}>Export report JSON</button></div></div>
    <div className={`ed-notice ${smoke ? "ed-smoke" : ""}`}><strong>{smoke ? "GENERATED-DATA SMOKE TEST — NOT BENCHMARK PERFORMANCE" : "REFERENCE COVERAGE DETERMINES WHICH SCORES CAN BE REPORTED"}</strong><p>{details?.notice || "Regenerate a detailed report, then refresh this page. No scores are estimated by the interface."}</p>{smoke && <p>These samples cannot substantiate real-world accuracy. A perfect matrix here only means the generated sample labels were matched.</p>}</div>
    <div className="ed-toolbar"><label>Result scope <select value={scope} onChange={(event) => setScope(event.target.value)}>{["overall", "flood", "wildfire", "landslide"].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><span>{n} scored classification cases · {summary.error_count ?? 0} errors</span></div>
    <div className="ed-scores">{[["Accuracy", summary.accuracy, true], ["Balanced accuracy", summary.balanced_accuracy, true], ["Weighted F1", summary.weighted_f1, true], ["Micro F1", summary.micro_f1, true], ["Cohen’s kappa", summary.cohens_kappa], ["Multiclass MCC", summary.multiclass_mcc]].map(([name, value, pct]) => <article key={name}><span>{name}</span><strong>{display(value, pct)}</strong></article>)}</div>
    <p className="ed-footnote">Accuracy 95% Wilson interval: {summary.accuracy_ci95 ? `${display(summary.accuracy_ci95.lower, true)} – ${display(summary.accuracy_ci95.upper, true)}` : "Not evaluated"}. This interval assumes independent cases; event-clustered samples require event-level uncertainty analysis. Balanced accuracy averages recall over supported classes. Undefined class ratios remain unavailable.</p>
    <div className="ed-grid"><article className="ed-panel"><div className="ed-panel-heading"><h3>Confusion matrix</h3><label><input type="checkbox" checked={normalized} onChange={(event) => setNormalized(event.target.checked)} /> Row normalized</label></div><p>Rows: actual class. Columns: predicted class.</p>{n ? <div className="ed-scroll"><table className="ed-matrix"><thead><tr><th>Actual ↓ / Predicted →</th>{labels.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{matrix.map((row, i) => <tr key={labels[i]}><th>{labels[i]}<small>n = {perClass[i]?.support ?? 0}</small></th>{row.map((value, j) => <td key={labels[j]} style={{ background: value === null ? "#f1f3f1" : `rgba(36,123,81,${.05+.6*(normalized ? value : value/Math.max(1,n))})` }}>{value === null ? "—" : normalized ? `${(value*100).toFixed(1)}%` : value}</td>)}</tr>)}</tbody></table></div> : <p className="ed-empty">Not evaluated: no eligible saved classification pairs.</p>}</article>
    <article className="ed-panel"><h3>Precision, recall & F1 by class</h3><p>Inspect support alongside each score; one sample cannot establish class performance.</p>{n ? perClass.map((row) => <div className="ed-class-bars" key={row.class}><strong>{row.class} <small>support {row.support}</small></strong>{["precision", "recall", "f1"].map((key) => <div className={`ed-bar ed-${key}`} key={key}><span>{key}</span><div><i style={{ width: `${Number.isFinite(row[key]) ? row[key]*100 : 0}%` }} /></div><b>{display(row[key], true)}</b></div>)}</div>) : <p className="ed-empty">Not evaluated.</p>}</article></div>
    <article className="ed-panel"><h3>Per-class classification report</h3><div className="ed-scroll"><table><thead><tr>{["Class", "Support", "Predicted", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Specificity", "NPV"].map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{perClass.map((row) => <tr key={row.class}><th>{row.class}</th>{["support", "predicted_count", "tp", "fp", "fn", "tn"].map((key) => <td key={key}>{row[key]}</td>)}{["precision", "recall", "f1", "specificity", "negative_predictive_value"].map((key) => <td key={key}>{display(row[key], true)}</td>)}</tr>)}</tbody></table></div>{!n && <p className="ed-empty">No eligible class references or predictions.</p>}</article>
    <div className="ed-grid"><article className="ed-panel"><h3>ROC curves · one versus rest</h3><CurvePlot curves={probability?.curves} mode="roc" /></article><article className="ed-panel"><h3>Precision–recall curves · one versus rest</h3><CurvePlot curves={probability?.curves} mode="pr" /><p className="ed-footnote">AP means average precision; the class prevalence is the no-skill precision baseline.</p></article></div>
    <article className="ed-panel"><h3>Probability quality & calibration</h3><p>{probability?.eligible_count || 0} of {n} classified cases have usable full probability vectors. Calibration uses the probability argmax.</p><div className="ed-scores">{[["Log loss", probability?.log_loss], ["Multiclass Brier score (0–2)", probability?.multiclass_brier_score], ["Calibration error · 10 bins", probability?.expected_calibration_error]].map(([name, value]) => <article key={name}><span>{name}</span><strong>{display(value)}</strong></article>)}</div>{probability?.status !== "available" ? <p className="ed-empty">{probability?.reason || "Not evaluated: complete probabilities are unavailable."}</p> : <div className="ed-scroll"><table><thead><tr><th>Confidence bin</th><th>Cases</th><th>Mean confidence</th><th>Observed accuracy</th><th>Absolute gap</th></tr></thead><tbody>{probability.calibration_bins.map((bin) => <tr key={bin.lower}><th>{(bin.lower*100).toFixed(0)}–{(bin.upper*100).toFixed(0)}%</th><td>{bin.count}</td><td>{display(bin.confidence,true)}</td><td>{display(bin.accuracy,true)}</td><td>{bin.count ? display(Math.abs(bin.confidence-bin.accuracy),true) : "—"}</td></tr>)}</tbody></table></div>}</article>
    <article className="ed-panel"><h3>Metric coverage & error distributions</h3><p>Coverage is scored cases / test cases. Quartiles and p95 describe eligible per-case values; they are not confidence intervals.</p><div className="ed-scroll"><table><thead><tr><th>Metric</th><th>Coverage</th><th>Mean</th><th>Std. dev.</th><th>Median</th><th>p25</th><th>p75</th><th>p95</th></tr></thead><tbody>{Object.entries(coverage).map(([key, value]) => { const stats = distributions[key] || {}; return <tr key={key}><th title={value.reason || ""}>{title(key)}</th><td>{value.eligible_count}/{value.test_count}</td>{["mean", "std_population", "median", "p25", "p75", "p95"].map((name) => <td key={name}>{display(stats[name])}</td>)}</tr>; })}</tbody></table></div>{!Object.keys(coverage).length && <p className="ed-empty">Coverage was not recorded in this report. The generated smoke report does not supply benchmark coverage.</p>}</article>
    <article className="ed-panel"><h3>Misclassified cases</h3>{classification?.per_case?.some((row) => row.true_hazard !== row.predicted_hazard) ? <div className="ed-scroll"><table><thead><tr><th>Case</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>{classification.per_case.filter((row) => row.true_hazard !== row.predicted_hazard).map((row, i) => <tr key={`${row.case_id}-${i}`}><th>{row.case_id}</th><td>{row.true_hazard}</td><td>{row.predicted_hazard}</td></tr>)}</tbody></table></div> : <p className="ed-empty">{n ? `No mismatches among these ${n} cases. This does not establish generalization.` : "Not evaluated: no eligible cases."}</p>}</article>
  </section>;
}
