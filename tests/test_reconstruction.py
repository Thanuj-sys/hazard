import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

from evaluation import reconstruction
from evaluation.reconstruction import (
    EVENT_MANIFEST_FIELDS,
    EventNotReadyError,
    check_event_readiness,
    get_event,
    load_event_inputs,
    load_event_manifest,
    load_ground_truth,
    reconstruct_event,
    route_hazard_logits,
    select_first_data_ready_event,
    write_prediction_artifacts,
)


TEST_EVENT_ID = "TST-F"
EVENT_START = "2022-08-02"
EVENT_END = "2022-08-03"
TRANSFORM = from_origin(500000, 2500000, 10, 10)
CRS = "EPSG:32643"
IMAGE_DESCRIPTIONS = (
    "VV", "VH", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "Band9", "Band10"
)


def write_raster(path, array, *, descriptions=None, acquisition_date=None, nodata=None):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=array.dtype,
        crs=CRS,
        transform=TRANSFORM,
        nodata=nodata,
    ) as dataset:
        dataset.write(array)
        if descriptions:
            for index, description in enumerate(descriptions, 1):
                dataset.set_band_description(index, description)
        if acquisition_date:
            dataset.update_tags(acquisition_date=acquisition_date)
    return path


def make_manifest_row(input_paths=None, ground_truth="not_available"):
    input_paths = input_paths or {}
    row = {field: "not_verified" for field in EVENT_MANIFEST_FIELDS}
    row.update({
        "event_id": TEST_EVENT_ID,
        "hazard": "Flood",
        "event_name": "Synthetic unit-test fixture",
        "country": "Test fixture only",
        "state": "Test fixture only",
        "district": "Test fixture only",
        "start_date": EVENT_START,
        "end_date": EVENT_END,
        "latitude": "not_available",
        "longitude": "not_available",
        "aoi": "local projected test raster",
        "pre_event_data": str(input_paths.get("pre_event_data", "not_available")),
        "during_event_data": str(input_paths.get("during_event_data", "not_available")),
        "post_event_data": "not_available",
        "rainfall_data": "not_available",
        "dem_data": str(input_paths.get("dem_data", "not_available")),
        "slope_data": str(input_paths.get("slope_data", "not_available")),
        "geology_data": str(input_paths.get("geology_data", "not_available")),
        "landcover_data": "not_available",
        "ground_truth": str(ground_truth),
        "ground_truth_area": "not_available",
        "ground_truth_runout": "not_applicable",
        "ground_truth_severity": "not_available",
        "source": "Synthetic unit-test fixture; not a real event source",
        "source_url": "urn:test:synthetic-fixture",
        "data_status": "verified",
        "notes": "Temporary synthetic fixture used only by a unit test.",
    })
    return row


def write_manifest(path, row):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=EVENT_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return path


def make_ready_fixture(tmp_path, *, ground_truth="not_available"):
    root = Path(tmp_path)
    input_paths = {}
    image = np.ones((10, 16, 16), dtype=np.float32)
    image[4] = 0.2
    image[5] = 0.6
    input_paths["pre_event_data"] = write_raster(
        root / "pre.tif", image,
        descriptions=IMAGE_DESCRIPTIONS,
        acquisition_date="2022-08-01",
    )
    input_paths["during_event_data"] = write_raster(
        root / "during.tif", image,
        descriptions=IMAGE_DESCRIPTIONS,
        acquisition_date="2022-08-02",
    )
    input_paths["dem_data"] = write_raster(root / "dem.tif", np.arange(256, dtype=np.float32).reshape(16, 16))
    input_paths["slope_data"] = write_raster(root / "slope.tif", np.full((16, 16), 3.0, dtype=np.float32))
    input_paths["geology_data"] = write_raster(root / "geology.tif", np.full((16, 16), 4, dtype=np.uint8))
    manifest_path = write_manifest(root / "event_manifest.csv", make_manifest_row(input_paths, ground_truth))
    return root, manifest_path, input_paths


class StubSegmentationModel:
    segmentation_enabled = True
    segmentation_output_size = (4, 4)
    config = {"model": {"segmentation_threshold": 0.5}}

    def __init__(self, logits=None):
        self.logits = logits if logits is not None else torch.full((1, 1, 4, 4), 20.0)
        self.calls = 0
        self.inputs = None

    def to(self, device):
        self.logits = self.logits.to(device)
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls += 1
        self.inputs = inputs
        return {
            "segmentation": {
                "flood": self.logits,
                "landslide": -self.logits,
                "wildfire": self.logits * 0.0,
            }
        }


def test_manifest_loading_and_unknown_event_error(tmp_path):
    _, manifest_path, _ = make_ready_fixture(tmp_path)

    event = get_event(TEST_EVENT_ID, manifest_path)
    assert event["event_name"] == "Synthetic unit-test fixture"
    assert set(load_event_manifest(manifest_path)) == {TEST_EVENT_ID}
    with pytest.raises(KeyError, match="Unknown event ID"):
        get_event("NO-SUCH-EVENT", manifest_path)


def test_missing_event_input_is_reported_and_no_prediction_runs(tmp_path):
    root = Path(tmp_path)
    manifest_path = write_manifest(
        root / "missing.csv",
        make_manifest_row({"pre_event_data": root / "absent.tif"}),
    )
    readiness = check_event_readiness(TEST_EVENT_ID, manifest_path=manifest_path, data_root=root)
    model = StubSegmentationModel()

    assert not readiness["inputs_available"]
    assert readiness["status"] == "not_ready"
    assert any("pre_event_data" in item for item in readiness["missing_inputs"])
    result = reconstruct_event(
        TEST_EVENT_ID,
        manifest_path=manifest_path,
        data_root=root,
        results_root=root / "results",
        model=model,
    )
    assert result["summary"]["status"] == "not_ready"
    assert result["summary"]["prediction"] is None
    assert model.calls == 0
    assert Path(result["artifacts"]["reconstruction_summary"]).is_file()


def test_ground_truth_path_is_excluded_from_model_input_construction(tmp_path, monkeypatch):
    root, manifest_path, _ = make_ready_fixture(
        tmp_path,
        ground_truth=str(Path(tmp_path) / "independent_truth.geojson"),
    )
    monkeypatch.setattr(
        reconstruction,
        "load_ground_truth",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GT read during input build")),
    )

    bundle = load_event_inputs(TEST_EVENT_ID, manifest_path=manifest_path, data_root=root)

    assert "ground_truth" not in bundle["model_inputs"]
    assert str(root / "independent_truth.geojson") not in str(bundle["model_inputs"])
    assert bundle["metadata"]["event_date"] == EVENT_START
    assert {record["acquisition_date"] for record in bundle["metadata"]["input_sources"] if record["acquisition_date"]} == {
        "2022-08-01", "2022-08-02"
    }


def test_historical_phase_dates_are_enforced(tmp_path):
    root, manifest_path, paths = make_ready_fixture(tmp_path)
    image = np.ones((10, 16, 16), dtype=np.float32)
    write_raster(
        paths["pre_event_data"], image,
        descriptions=IMAGE_DESCRIPTIONS,
        acquisition_date="2022-08-04",
    )

    readiness = check_event_readiness(TEST_EVENT_ID, manifest_path=manifest_path, data_root=root)

    assert not readiness["inputs_available"]
    assert any("not before the event start date" in item for item in readiness["missing_inputs"])


def test_unconsumed_declared_rainfall_is_not_silently_ignored(tmp_path):
    root, manifest_path, paths = make_ready_fixture(tmp_path)
    rainfall_path = write_raster(
        root / "rainfall.tif", np.ones((16, 16), dtype=np.float32)
    )
    row = make_manifest_row(paths)
    row["rainfall_data"] = str(rainfall_path)
    write_manifest(manifest_path, row)

    readiness = check_event_readiness(
        TEST_EVENT_ID, manifest_path=manifest_path, data_root=root
    )

    assert not readiness["inputs_available"]
    assert "rainfall_data" in readiness["declared_but_not_consumed"]
    assert any("declared_but_not_consumed" in item for item in readiness["missing_inputs"])


def test_l1_wayanad_metadata_contract_uses_verified_cartosat_and_risat_sources():
    metadata_path = Path(__file__).resolve().parents[1] / "data" / "events" / "L1" / "metadata" / "event_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["historical_event_date"] == "2024-07-30"
    assert payload["pre_event_sensor"] == "Cartosat-3"
    assert payload["pre_event_acquisition_date"] == "2023-05-22"
    assert payload["post_event_sensor"] == "RISAT SAR"
    assert payload["post_event_acquisition_date"] == "2024-07-31"
    assert payload["ground_truth_status"] == "map_reference_only"

    readiness = check_event_readiness("L1", manifest_path=Path(__file__).resolve().parents[1] / "data" / "events" / "event_manifest.csv", data_root=Path(__file__).resolve().parents[1])
    assert readiness["status"] == "not_ready"
    assert readiness["ground_truth_status"] in {"not_validated", "map_reference_only"}


def test_hazard_routing_selects_matching_segmentation_head():
    outputs = {
        "flood": torch.tensor([1.0]),
        "landslide": torch.tensor([2.0]),
        "wildfire": torch.tensor([3.0]),
    }
    assert route_hazard_logits(outputs, "flood").item() == 1.0
    assert route_hazard_logits(outputs, "landslide").item() == 2.0
    assert route_hazard_logits(outputs, "wildfire").item() == 3.0


def test_hazard_schemas_are_specific_to_hazard_and_do_not_force_10_raw_bands():
    config = reconstruction._event_data_config()
    flood = config["hazards"]["flood"]
    landslide = config["hazards"]["landslide"]
    wildfire = config["hazards"]["wildfire"]

    assert set(flood["required_inputs"]) == {"pre_event_data", "during_event_data", "dem_data", "slope_data"}
    assert "rainfall_data" in flood["unsupported_declared_inputs"]
    assert set(landslide["required_inputs"]) == {"pre_event_data", "post_event_data", "dem_data", "slope_data"}
    assert set(wildfire["required_inputs"]) == {"pre_event_data", "post_event_data", "dem_data", "slope_data"}
    assert config["model_input_channels"] == 10
    assert flood["channels"][0]["name"] == "pre_vv"
    assert landslide["channels"][0]["name"] == "pre_b02"
    assert wildfire["channels"][0]["name"] == "pre_b04"
    assert route_hazard_logits({"flood": torch.tensor([1.0]), "landslide": torch.tensor([2.0]), "wildfire": torch.tensor([3.0])}, "flood").item() == 1.0


def test_synthetic_logits_create_prediction_artifacts_without_ground_truth(tmp_path):
    root, manifest_path, _ = make_ready_fixture(tmp_path)
    model = StubSegmentationModel()
    result = reconstruct_event(
        TEST_EVENT_ID,
        manifest_path=manifest_path,
        data_root=root,
        results_root=root / "results",
        model=model,
    )
    summary = result["summary"]

    assert model.calls == 1
    assert "target_mask" not in model.inputs
    assert "ground_truth" not in model.inputs
    assert summary["status"] == "demo_only"
    assert summary["ground_truth"]["available"] is False
    assert result["readiness"]["inputs_available"] is True
    assert Path(result["artifacts"]["input_metadata"]).is_file()
    assert Path(result["artifacts"]["logits_raster"]).is_file()
    assert Path(result["artifacts"]["probability_raster"]).is_file()
    assert Path(result["artifacts"]["raw_mask_raster"]).is_file()
    assert Path(result["artifacts"]["processed_mask_raster"]).is_file()
    assert Path(result["artifacts"]["polygon_geojson"]).is_file()
    assert Path(result["artifacts"]["summary_json"]).is_file()
    with rasterio.open(result["artifacts"]["logits_raster"]) as logits_raster:
        assert logits_raster.crs.to_string() == CRS
        assert logits_raster.width == 4 and logits_raster.height == 4


def test_ground_truth_geojson_loads_separately_and_area_only_is_not_spatial(tmp_path):
    root, manifest_path, paths = make_ready_fixture(tmp_path)
    truth_path = root / "truth.geojson"
    truth_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": mapping(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        }],
    }), encoding="utf-8")
    row = make_manifest_row(paths, ground_truth=str(truth_path))
    write_manifest(manifest_path, row)

    ground_truth = load_ground_truth(TEST_EVENT_ID, manifest_path=manifest_path, data_root=root)

    assert ground_truth["available"] is True
    assert ground_truth["status"] == "spatial_polygon"
    assert ground_truth["crs"] == "EPSG:4326"

    row["ground_truth"] = "not_available"
    row["ground_truth_area"] = "12.5"
    write_manifest(manifest_path, row)
    area_only = load_ground_truth(TEST_EVENT_ID, manifest_path=manifest_path, data_root=root)
    assert area_only["available"] is False
    assert area_only["status"] == "area_only"
    assert area_only["area_km2"] == 12.5


def test_current_benchmark_events_are_not_marked_ready():
    assert select_first_data_ready_event() is None
    readiness = check_event_readiness("L1")
    assert readiness["status"] == "not_ready"
    assert readiness["inputs_available"] is False
    assert readiness["ground_truth_available"] is False
