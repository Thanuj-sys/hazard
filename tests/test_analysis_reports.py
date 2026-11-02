"""PDF persistence and point-analysis integration checks (no model inference)."""
from pathlib import Path
from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader

from api import analysis_reports as reports


@pytest.fixture
def prediction():
    return {
        "status": "success", "predicted_coordinates": [11.6854, 76.1320],
        "hazard_type": "landslide", "radius_km": 2.5, "affected_area_km2": 19.63,
        "hazard_probabilities": {"flood": 0.1, "wildfire": 0.2, "landslide": 0.7},
        "language_report": "Check slopes < 30 degrees & verify locally.",
        "data_source": "Test satellite", "source_acquired_at": "2026-10-27T12:00:00Z",
        "source_age_hours": 3, "data_freshness": "recent", "inference_mode": "planetary_computer",
        "cached": False,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_REPORTS_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(reports.router)
    return TestClient(app)


def test_saved_pdf_contains_result_and_supports_view_and_download(client, prediction):
    result = reports.attach_analysis_pdf(prediction)
    metadata = result["pdf_report"]
    path = Path(metadata["local_path"])
    assert metadata["status"] == "saved"
    assert path.is_file()
    response = client.get("/api/v1/" + metadata["view_path"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content == path.read_bytes()
    reader = PdfReader(BytesIO(response.content))
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in ["11.68540", "76.13200", "landslide", "70.00%", "2.50 km", "19.63 km2",
                     "Executive summary", "Hazard signal breakdown"]:
        assert expected in text
    download = client.get("/api/v1/" + metadata["download_path"])
    assert download.headers["content-disposition"].startswith("attachment;")
    assert download.content == response.content
    assert "pdf_report" not in prediction


def test_long_observation_still_generates_one_page_and_unique_files(client, prediction):
    prediction["language_report"] = "Verify this generated note against local observations. " * 700
    first = reports.save_analysis_pdf(prediction)
    second = reports.save_analysis_pdf(prediction)
    assert first["filename"] != second["filename"]
    reader = PdfReader(first["local_path"])
    assert len(reader.pages) == 1
    assert "Decision support only" in reader.pages[-1].extract_text()


def test_missing_and_invalid_report_paths_are_rejected(client):
    assert client.get("/api/v1/analysis-reports/not-a-report.pdf").status_code == 404
    missing = "analysis-20261028T123456000000Z-" + "a" * 32 + ".pdf"
    assert client.get("/api/v1/analysis-reports/" + missing).status_code == 404
    assert client.get("/api/v1/analysis-reports/%2e%2e%2fREADME.md").status_code == 404


def test_storage_failure_preserves_analysis(tmp_path, monkeypatch, prediction):
    blocked = tmp_path / "file-not-directory"
    blocked.write_text("occupied")
    monkeypatch.setenv("ANALYSIS_REPORTS_DIR", str(blocked))
    result = reports.attach_analysis_pdf(prediction)
    assert result["hazard_probabilities"] == prediction["hazard_probabilities"]
    assert result["pdf_report"]["status"] == "failed"
    assert "pdf_report" not in prediction


@pytest.mark.parametrize("selected", ["all", "flood", "wildfire", "landslide"])
@pytest.mark.parametrize("cached", [False, True])
def test_predict_point_automatically_saves_pdf(tmp_path, monkeypatch, prediction, cached, selected):
    from api import app as gateway
    monkeypatch.setenv("ANALYSIS_REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(gateway, "validate_point", lambda lat, lon: (lat, lon))
    monkeypatch.setattr(gateway, "INFERENCE_MODE", "demo_synthetic")
    monkeypatch.setattr(gateway, "ALLOW_SYNTHETIC_INFERENCE", True)
    monkeypatch.setattr(gateway, "get_cached_prediction", lambda *args: {**prediction, "cached": True} if cached else None)
    build = Mock(return_value=prediction)
    monkeypatch.setattr(gateway, "build_demo_prediction", build)
    monkeypatch.setattr(gateway, "save_prediction", Mock())
    monkeypatch.setattr(gateway, "cache_prediction", Mock())
    client = TestClient(gateway.app)
    response = client.post("/api/v1/predict-point", json={"latitude": 11.6854, "longitude": 76.132, "selected_hazard": selected})
    assert response.status_code == 200
    payload = response.json()
    assert payload["pdf_report"]["status"] == "saved"
    assert payload["cached"] is cached
    assert payload["selected_hazard"] == selected
    assert payload["pdf_report"]["selected_hazard"] == selected
    assert payload["hazard_type"] == prediction["hazard_type"]
    assert Path(payload["pdf_report"]["local_path"]).is_file()
    assert client.get("/api/v1/" + payload["pdf_report"]["view_path"]).status_code == 200
    assert build.call_count == (0 if cached else 1)


def test_library_retains_new_and_legacy_pdfs_after_restart(client, prediction):
    first = reports.save_analysis_pdf(prediction)
    second = reports.save_analysis_pdf(prediction)
    Path(first["local_path"]).with_suffix(".json").unlink()
    # Start a separate app: catalogue state must come from disk, not process memory.
    app = FastAPI()
    app.include_router(reports.router)
    restarted = TestClient(app)
    response = restarted.get("/api/v1/analysis-reports")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert [row["filename"] for row in payload["reports"]] == [second["filename"], first["filename"]]
    assert payload["reports"][0]["hazard_type"] == "landslide"
    assert payload["reports"][0]["predicted_coordinates"] == prediction["predicted_coordinates"]
    assert payload["reports"][1]["hazard_type"] is None
    for row in payload["reports"]:
        assert row["size_bytes"] > 0
        assert restarted.get("/api/v1/" + row["view_path"]).status_code == 200


def test_report_library_handles_missing_folder_and_corrupt_metadata(client, prediction):
    assert client.get("/api/v1/analysis-reports").json()["reports"] == []
    metadata = reports.save_analysis_pdf(prediction)
    Path(metadata["local_path"]).with_suffix(".json").write_text("broken JSON")
    response = client.get("/api/v1/analysis-reports")
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["reports"][0]["filename"] == metadata["filename"]


def test_save_retry_creates_structured_local_report(client, prediction):
    response = client.post("/api/v1/analysis-reports", json=prediction)
    assert response.status_code == 200
    metadata = response.json()
    assert Path(metadata["local_path"]).is_file()
    reader = PdfReader(metadata["local_path"])
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) == 1
    for section in ["Executive summary", "Hazard signal breakdown"]:
        assert section in text
    for removed in ["Imagery and analysis provenance", "Model-generated observation", "Review checklist"]:
        assert removed not in text
    assert all(f"Page {i + 1}" in page.extract_text() for i, page in enumerate(reader.pages))
    assert client.get("/api/v1/analysis-reports").json()["count"] == 1
    invalid = client.post("/api/v1/analysis-reports", json={"language_report": "missing analysis fields"})
    assert invalid.status_code == 422
    assert client.get("/api/v1/analysis-reports").json()["count"] == 1


def test_save_retry_reports_disk_failure(client, tmp_path, monkeypatch, prediction):
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("occupied")
    monkeypatch.setenv("ANALYSIS_REPORTS_DIR", str(blocked))
    response = client.post("/api/v1/analysis-reports", json=prediction)
    assert response.status_code == 503
    assert "could not be saved locally" in response.json()["detail"]


@pytest.mark.parametrize("selected", ["all", "flood", "wildfire", "landslide"])
def test_pdf_focus_and_library_follow_lens_without_changing_prediction(client, prediction, selected):
    prediction["hazard_type"] = "wildfire"
    prediction["hazard_probabilities"] = {"flood": 0.25, "wildfire": 0.45, "landslide": 0.3}
    response = client.post("/api/v1/analysis-reports", json={**prediction, "selected_hazard": selected})
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["selected_hazard"] == selected
    assert metadata["primary_hazard"] == "wildfire"
    reader = PdfReader(metadata["local_path"])
    assert len(reader.pages) == 1
    text = " ".join(reader.pages[0].extract_text().split())
    assert "Primary signal: wildfire" in text
    if selected == "all":
        assert "All hazards assessment report" in text
        assert "Report focus: all hazards" in text
        assert "flood 25.00%, wildfire 45.00%, landslide 30.00%" in text
        for label in ("Flood", "Wildfire", "Landslide"):
            assert label in text
    else:
        assert f"{selected.title()} assessment report" in text
        assert f"Report focus: {selected}" in text
    assert "25.00%" in text and "45.00%" in text and "30.00%" in text
    entry = client.get("/api/v1/analysis-reports").json()["reports"][0]
    assert entry["selected_hazard"] == selected
    assert entry["hazard_type"] == "wildfire"


def test_report_rejects_invalid_hazard_lens(client, prediction):
    response = client.post("/api/v1/analysis-reports", json={**prediction, "selected_hazard": "unknown"})
    assert response.status_code == 422
    assert client.get("/api/v1/analysis-reports").json()["count"] == 0
