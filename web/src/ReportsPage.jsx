import { useEffect, useState } from "react";
import { ArrowDownToLine, CircleAlert, ExternalLink, FileText, FolderOpen, LoaderCircle, RefreshCw, Search, X } from "lucide-react";
import "./reports.css";

function dateLabel(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Date unavailable" : new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium", timeStyle: "short",
  }).format(date);
}

function sizeLabel(value) {
  return value < 1024 * 1024 ? `${Math.max(1, Math.round(value / 1024))} KB` : `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ReportsPage({ apiBase }) {
  const [library, setLibrary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    async function load() {
      try {
        const response = await fetch(`${apiBase}/analysis-reports`, { signal: controller.signal, cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not load saved reports.");
        setLibrary(payload);
        setSelected((current) => current ? payload.reports.find((report) => report.filename === current.filename) || null : null);
      } catch (requestError) {
        if (requestError.name !== "AbortError") setError(requestError.message || "Could not load saved reports.");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [apiBase, refresh]);

  const reports = library?.reports || [];
  const visible = reports.filter((report) => [report.filename, report.hazard_type, report.selected_hazard, ...(report.predicted_coordinates || []), dateLabel(report.saved_at)]
    .join(" ").toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <div className="reports-page">
      <div className="reports-heading">
        <div><span className="eyebrow compact">LOCAL REPORT LIBRARY</span><h1>View reports</h1><p>Every saved analysis, ready to revisit. Preview a PDF or download a copy.</p></div>
        <button className="reports-button" type="button" onClick={() => setRefresh((value) => value + 1)} disabled={loading}>
          {loading ? <LoaderCircle size={16} className="reports-spinner" /> : <RefreshCw size={16} />} Refresh
        </button>
      </div>
      <div className="reports-storage"><FolderOpen size={20} /><div><strong>{reports.length} {reports.length === 1 ? "report" : "reports"} saved locally</strong><p>PDFs stay on this computer after you close or restart the app.</p>{library?.storage_directory && <code>{library.storage_directory}</code>}</div></div>
      <label className="reports-search"><Search size={17} /><span className="reports-sr-only">Search reports</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search hazard, coordinates, date or filename" /></label>
      {error && <div className="error-note" role="alert"><CircleAlert size={17} /><span>{error} Use Refresh to try again.</span></div>}
      {loading && <p className="reports-status" role="status"><LoaderCircle size={16} className="reports-spinner" /> Loading local reports...</p>}
      {!loading && !error && reports.length === 0 && <div className="reports-empty"><FileText size={32} /><h2>Your reports will appear here</h2><p>Run Analyze on the situation map. Its PDF is automatically saved and added to this library.</p></div>}
      {!loading && reports.length > 0 && visible.length === 0 && <p className="reports-status">No reports match your search.</p>}
      {visible.length > 0 && <div className={`reports-layout ${selected ? "has-preview" : ""}`}>
        <div className="reports-list" aria-label="Saved PDF reports">
          {visible.map((report) => {
            const coordinates = report.predicted_coordinates;
            const focus = report.selected_hazard === "all" ? "All hazards" : report.selected_hazard || report.hazard_type;
            return <article className={`report-file ${selected?.filename === report.filename ? "is-selected" : ""}`} key={report.filename}>
              <div className="report-file-heading"><span className="report-file-icon"><FileText size={20} /></span><div><span className="report-file-kind">POINT ANALYSIS</span><h2>{focus ? `${focus} assessment` : "Saved analysis report"}</h2></div><span className="report-file-format">PDF</span></div>
              {report.hazard_type && <p className="report-file-location">Model primary signal: {report.hazard_type}</p>}
              <div className="report-file-details"><span>{dateLabel(report.saved_at)}</span><span>{sizeLabel(report.size_bytes)}</span></div>
              {coordinates?.length === 2 && <p className="report-file-location">Location: {coordinates.map((value) => Number(value).toFixed(5)).join(", ")}</p>}
              <p className="report-file-name" title={report.filename}>{report.filename}</p>
              <div className="report-file-actions">
                <button className="reports-button primary" type="button" aria-pressed={selected?.filename === report.filename} onClick={() => setSelected(report)}><FileText size={14} /> View PDF</button>
                <a className="reports-button" href={`${apiBase}/${report.download_path}`}><ArrowDownToLine size={14} /> Download</a>
              </div>
            </article>;
          })}
        </div>
        {selected && <section className="report-preview" aria-label="PDF preview">
          <div className="report-preview-heading"><strong>Report preview</strong><a href={`${apiBase}/${selected.view_path}`} target="_blank" rel="noreferrer" aria-label="Open PDF in a new tab"><ExternalLink size={17} /></a><button type="button" onClick={() => setSelected(null)} aria-label="Close PDF preview"><X size={18} /></button></div>
          <iframe key={selected.filename} title={`PDF preview: ${selected.filename}`} src={`${apiBase}/${selected.view_path}`} />
          <p>If the preview is unavailable, <a href={`${apiBase}/${selected.view_path}`} target="_blank" rel="noreferrer">open the PDF in a new tab</a>.</p>
        </section>}
      </div>}
    </div>
  );
}
