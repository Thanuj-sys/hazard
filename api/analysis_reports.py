"""Durable local analysis PDFs and the report library API."""
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from typing import Literal
from xml.sax.saxutils import escape
import json
import os
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

router = APIRouter(prefix="/api/v1/analysis-reports", tags=["Saved analysis PDFs"])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_NAME = re.compile(r"analysis-\d{8}T\d{12}Z-[a-f0-9]{32}\.pdf")
IST = timezone(timedelta(hours=5, minutes=30))
GREEN = colors.HexColor("#267b65")
INK = colors.HexColor("#193b30")
MUTED = colors.HexColor("#63786e")
PALE = colors.HexColor("#f1f6f3")
HAZARD_COLORS = {"flood": "#1687a7", "wildfire": "#de704c", "landslide": "#8170b5"}


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    predicted_coordinates: list[float] = Field(min_length=2, max_length=2)
    hazard_type: str = Field(min_length=1, max_length=100)
    selected_hazard: Literal["all", "flood", "wildfire", "landslide"] = "all"
    hazard_probabilities: dict[str, float]
    radius_km: float = Field(ge=0)
    affected_area_km2: float | None = Field(default=None, ge=0)
    language_report: str = Field(default="", max_length=100000)


def report_directory():
    configured = Path(os.getenv("ANALYSIS_REPORTS_DIR", "reports/analyses"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def report_metadata(path, saved_at):
    return {"status": "saved", "filename": path.name, "saved_at": saved_at,
            "local_path": str(path), "view_path": f"analysis-reports/{path.name}",
            "download_path": f"analysis-reports/{path.name}?download=true"}


def save_analysis_pdf(prediction):
    """Write the PDF to disk before returning a saved status; never auto-delete reports."""
    generated = datetime.now(timezone.utc)
    report_id = uuid4().hex
    filename = f"analysis-{generated:%Y%m%dT%H%M%S%fZ}-{report_id}.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=27,
                              leading=31, textColor=INK, spaceAfter=12))
    styles.add(ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=13,
                              leading=18, textColor=GREEN, spaceBefore=17, spaceAfter=9,
                              keepWithNext=True))
    styles.add(ParagraphStyle("ReportBody", fontName="Helvetica", fontSize=10,
                              leading=15, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("SmallNote", parent=styles["ReportBody"], fontSize=8,
                              leading=12, textColor=MUTED))
    styles.add(ParagraphStyle("TableLabel", parent=styles["SmallNote"], spaceAfter=0))
    styles.add(ParagraphStyle("TableValue", parent=styles["ReportBody"], spaceAfter=0))
    story = []
    width = A4[0] - 96

    def p(value, style="ReportBody"):
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[style])

    def paragraph(value, style="ReportBody"):
        story.append(p(value, style))

    def number(value):
        return "Not reported" if value is None else f"{float(value):,.2f}"

    def details(rows):
        table = Table([[p(label, "TableLabel"), p(value, "TableValue")] for label, value in rows],
                      colWidths=[width * 0.32, width * 0.68], hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("BACKGROUND", (0, 0), (0, -1), PALE),
                                   ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dce7df")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                   ("TOPPADDING", (0, 0), (-1, -1), 9),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
        story.append(table)

    coords = prediction.get("predicted_coordinates") or []
    location = f"{float(coords[0]):.5f}, {float(coords[1]):.5f}" if len(coords) == 2 else "Not reported"
    hazard = str(prediction.get("hazard_type") or "Not reported")
    selected = prediction.get("selected_hazard") or "all"
    if selected not in {"all", "flood", "wildfire", "landslide"}:
        raise ValueError("Invalid selected hazard")
    focus = "All hazards" if selected == "all" else selected.title()
    scores = sorted((prediction.get("hazard_probabilities") or {}).items(), key=lambda item: item[1], reverse=True)
    paragraph("POINT ANALYSIS / INDIA", "SmallNote")
    paragraph(f"{focus} assessment report", "ReportTitle")
    paragraph(f"Report ID: {report_id[:12].upper()} | Generated: {generated.astimezone(IST):%d %b %Y, %I:%M %p} IST", "SmallNote")
    paragraph(f"Generated (UTC): {generated:%Y-%m-%d %H:%M:%S}", "SmallNote")
    paragraph("01  Executive summary", "Section")
    summary = ""
    if selected == "all":
        probabilities = prediction.get("hazard_probabilities") or {}
        labels = []
        for name in ("flood", "wildfire", "landslide"):
            value = probabilities.get(name)
            labels.append(f"{name} {float(value) * 100:.2f}%" if value is not None else f"{name} not reported")
        summary = "Report focus: all hazards. Model scores: " + ", ".join(labels) + ". "
    else:
        score = (prediction.get("hazard_probabilities") or {}).get(selected)
        score_label = f"{float(score) * 100:.2f}%" if score is not None else "not reported"
        summary = f"Report focus: {selected}, with a model score of {score_label}. "
    summary += f"Primary signal: {hazard}. The selected point at {location} was assessed for flood, wildfire and landslide signals. "
    if scores and selected != "all":
        summary += f"The highest model score is {scores[0][0]} at {float(scores[0][1]) * 100:.2f}%. "
        if len(scores) > 1:
            summary += f"Its separation from the next score is {(float(scores[0][1]) - float(scores[1][1])) * 100:.2f} percentage points. "
    summary += "These scores describe the model output for this imagery; they do not establish whether an event occurred."
    paragraph(summary)
    details([("Report focus", focus),
             ("Location (latitude, longitude)", location),
             ("Primary model signal", hazard),
             (f"Impact radius ({hazard})", f"{number(prediction.get('radius_km'))} km"),
             (f"Circular area ({hazard})", f"{number(prediction.get('affected_area_km2'))} km2")])
    paragraph("02  Hazard signal breakdown", "Section")
    rows = [[p("Hazard", "TableLabel"), p("Model score", "TableLabel"), p("Relative signal", "TableLabel")]]
    for label, score in scores:
        bar = Drawing(180, 12)
        bar.add(Rect(0, 2, 180, 8, fillColor=colors.HexColor("#e6ede8"), strokeColor=None))
        bar.add(Rect(0, 2, 180 * max(0, min(1, float(score))), 8,
                     fillColor=colors.HexColor(HAZARD_COLORS.get(label, "#267b65")), strokeColor=None))
        rows.append([p(label.title()), p(f"{float(score) * 100:.2f}%"), bar])
    table = Table(rows, colWidths=[width * 0.29, width * 0.27, width * 0.44], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PALE),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dce7df"))]))
    story.append(table)
    story.append(Spacer(1, 10))
    paragraph("Model scores are not calibrated event probabilities or measured hazard severity. A higher score is not a verified warning.", "SmallNote")

    def page_frame(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(GREEN)
        canvas.setLineWidth(2)
        canvas.line(48, A4[1] - 35, A4[0] - 48, A4[1] - 35)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(48, A4[1] - 26, "MULTIHAZARD PREDICTOR")
        canvas.drawRightString(A4[0] - 48, A4[1] - 26, "POINT ASSESSMENT")
        canvas.setStrokeColor(colors.HexColor("#dce7df"))
        canvas.setLineWidth(0.5)
        canvas.line(48, 42, A4[0] - 48, 42)
        canvas.drawString(48, 28, f"{report_id[:12].upper()}  |  Decision support only")
        canvas.drawRightString(A4[0] - 48, 28, f"Page {document.page}")
        canvas.restoreState()

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title="MultiHazard point analysis", author="MultiHazard Predictor",
                                 rightMargin=48, leftMargin=48, topMargin=55, bottomMargin=56)
    document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    directory = report_directory()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    temporary = destination.with_suffix(".tmp")
    try:
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {**report_metadata(destination, generated.isoformat()),
                "selected_hazard": selected, "report_hazard": selected,
                "primary_hazard": hazard}
    # A separate snapshot supports catalogue details without depending on a database.
    # The PDF remains discoverable even if this optional metadata write fails.
    sidecar = destination.with_suffix(".json")
    temporary_metadata = sidecar.with_suffix(".json.tmp")
    try:
        snapshot = {key: value for key, value in prediction.items() if key != "pdf_report"}
        snapshot["selected_hazard"] = selected
        temporary_metadata.write_text(json.dumps({"report": metadata, "analysis": snapshot}, ensure_ascii=False), encoding="utf-8")
        temporary_metadata.replace(sidecar)
    except (OSError, TypeError, ValueError):
        logger.exception("PDF saved, but catalogue metadata could not be written")
    finally:
        try:
            temporary_metadata.unlink(missing_ok=True)
        except OSError:
            pass
    return metadata


def attach_analysis_pdf(prediction, selected_hazard=None):
    """Keep a successful model result visible and allow a retry if disk storage fails."""
    result = dict(prediction)
    if selected_hazard is not None:
        result["selected_hazard"] = selected_hazard
    try:
        result["pdf_report"] = save_analysis_pdf(result)
    except Exception:
        logger.exception("Could not save the analysis PDF")
        result["pdf_report"] = {"status": "failed", "detail": "Analysis completed, but its PDF could not be saved locally. Retry saving the report."}
    return result


@router.get("")
def list_analysis_reports():
    directory = report_directory()
    entries = []
    try:
        paths = list(directory.glob("*.pdf")) if directory.exists() else []
        for path in paths:
            if not REPORT_NAME.fullmatch(path.name) or path.is_symlink() or not path.is_file():
                continue
            saved_at = datetime.strptime(path.name.split("-")[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).isoformat()
            entry = {**report_metadata(path, saved_at), "size_bytes": path.stat().st_size,
                     "hazard_type": None, "selected_hazard": None, "predicted_coordinates": None, "radius_km": None}
            try:
                snapshot = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                analysis = snapshot.get("analysis", {})
                for key in ("hazard_type", "selected_hazard", "predicted_coordinates", "radius_km"):
                    entry[key] = analysis.get(key)
            except (OSError, ValueError, AttributeError, TypeError):
                pass  # Older PDFs and PDFs with missing metadata still belong in the library.
            entries.append(entry)
    except OSError as exc:
        logger.exception("Could not read locally saved reports")
        raise HTTPException(status_code=503, detail="The local reports folder could not be read.") from exc
    entries.sort(key=lambda entry: entry["filename"], reverse=True)
    return {"reports": entries, "count": len(entries), "storage_directory": str(directory)}


@router.post("")
def create_analysis_report(body: ReportRequest):
    try:
        return save_analysis_pdf(body.model_dump(exclude_unset=True))
    except Exception as exc:
        logger.exception("Could not save requested analysis report")
        raise HTTPException(status_code=503, detail="The PDF could not be saved locally. Check the reports folder and retry.") from exc


@router.get("/{filename}")
def view_analysis_pdf(filename: str, download: bool = False):
    if not REPORT_NAME.fullmatch(filename):
        raise HTTPException(status_code=404, detail="Saved PDF not found.")
    path = report_directory() / filename
    if not path.is_file() or path.is_symlink():
        raise HTTPException(status_code=404, detail="Saved PDF not found.")
    return FileResponse(path, media_type="application/pdf", filename=filename,
                        content_disposition_type="attachment" if download else "inline")
