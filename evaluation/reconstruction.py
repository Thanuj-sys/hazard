"""Historical event reconstruction with strict input/ground-truth separation."""
from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import yaml
from pyproj import CRS, Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import shape
from shapely.ops import unary_union

from data.preprocessing import align_raster_to_reference
from models.multihazard_vlm import MultiHazardVLM
from spatial.postprocess import process_segmentation_prediction, save_spatial_outputs

EVENT_MANIFEST_FIELDS = (
    "event_id", "hazard", "event_name", "country", "state", "district",
    "start_date", "end_date", "latitude", "longitude", "aoi",
    "pre_event_data", "during_event_data", "post_event_data", "rainfall_data",
    "dem_data", "slope_data", "geology_data", "landcover_data", "ground_truth",
    "ground_truth_area", "ground_truth_runout", "ground_truth_severity", "source",
    "source_url", "data_status", "notes",
)
INPUT_FIELDS = (
    "pre_event_data", "during_event_data", "post_event_data", "rainfall_data",
    "dem_data", "slope_data", "geology_data", "landcover_data",
)
IMAGE_FIELDS = ("pre_event_data", "during_event_data", "post_event_data")
UNAVAILABLE_VALUES = {"", "not_verified", "not_available", "not_applicable", "null", "none"}
HAZARD_HEADS = {"flood": "flood", "landslide": "landslide", "wildfire": "wildfire"}
READINESS_CATEGORIES = {"not_ready", "data_ready", "validation_ready", "demo_only"}
LEGACY_READINESS_STATUS_MAP = {
    "not_ready": "incomplete",
    "data_ready": "ready",
    "validation_ready": "validated",
    "demo_only": "prediction_only",
    "NOT_READY": "incomplete",
    "DATA_READY": "ready",
    "VALIDATION_READY": "validated",
    "DEMO_ONLY": "prediction_only",
}


def _canonical_readiness_status(inputs_available: bool, ground_truth_available: bool, ground_truth_status: str) -> str:
    if not inputs_available:
        return "not_ready"
    if ground_truth_available:
        return "validation_ready"
    if ground_truth_status in {"not_validated", "area_only", "missing", "not_available", "not_verified"}:
        return "demo_only"
    return "data_ready"


class EventNotReadyError(RuntimeError):
    """Raised when a historical event lacks verified local inputs."""

    def __init__(self, readiness: Dict[str, Any]):
        self.readiness = readiness
        reason = "; ".join(readiness.get("missing_inputs", [])) or "required historical inputs are missing"
        super().__init__(f"EVENT NOT READY: {reason}.")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path(path: Optional[Any] = None) -> Path:
    return Path(path) if path is not None else _project_root() / "data" / "events" / "event_manifest.csv"


def _event_data_config(config_path: Optional[Any] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else _project_root() / "configs" / "event_data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Hazard-specific event input config does not exist: {path}")
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config.get("hazards"), dict):
        raise ValueError(f"Event input config has no hazards mapping: {path}")
    return config


def _event_metadata_path(event_id: str, manifest_path: Optional[Any] = None) -> Path:
    return _manifest_path(manifest_path).resolve().parent / event_id / "metadata" / "metadata.json"


def _load_event_file_metadata(event_id: str, manifest_path: Optional[Any] = None) -> Dict[str, Any]:
    path = _event_metadata_path(event_id, manifest_path)
    if not path.is_file():
        return {"path": str(path), "inputs": []}
    with path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("event_id") != event_id:
        raise ValueError(f"Event metadata ID does not match directory: {path}")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"Event metadata inputs must be a list: {path}")
    return {**metadata, "path": str(path)}


def _normalized_band_name(name: Any) -> str:
    value = "".join(character for character in str(name).lower() if character.isalnum())
    aliases = {
        "blue": "b02", "green": "b03", "red": "b04", "nir": "b08",
        "nir8a": "b8a", "swir1": "b11", "swir2": "b12",
    }
    return aliases.get(value, value)


def _value(row: Mapping[str, Any], key: str) -> Optional[str]:
    raw = row.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return None if value.lower() in UNAVAILABLE_VALUES else value


def load_event_manifest(manifest_path: Optional[Any] = None) -> Dict[str, Dict[str, str]]:
    """Load manifest rows indexed by event ID without inferring unrecorded values."""
    path = _manifest_path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Event manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = set(EVENT_MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Event manifest is missing columns: {sorted(missing_columns)}")
        rows = {}
        for row in reader:
            event_id = (row.get("event_id") or "").strip()
            if not event_id:
                raise ValueError("Event manifest contains a row without event_id.")
            if event_id in rows:
                raise ValueError(f"Duplicate event_id {event_id!r} in event manifest.")
            rows[event_id] = row
    return rows


def get_event(event_id: str, manifest_path: Optional[Any] = None) -> Dict[str, str]:
    rows = load_event_manifest(manifest_path)
    if event_id not in rows:
        raise KeyError(f"Unknown event ID {event_id!r}; no matching manifest row exists.")
    return rows[event_id]


def _local_paths(value: Optional[str], root: Path) -> List[Path]:
    if value is None:
        return []
    paths = []
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "://" in entry and not entry.lower().startswith("file://"):
            continue
        path = Path(entry.removeprefix("file://"))
        if not path.is_absolute():
            path = root / path
        paths.append(path.resolve())
    return paths


def _paths_for_row(row: Mapping[str, Any], field: str, root: Path) -> List[Path]:
    return _local_paths(_value(row, field), root)


def _parse_manifest_date(value: Optional[str], field: str) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} must be an exact ISO date (YYYY-MM-DD), got {value!r}.") from None


def _acquisition_date(dataset: rasterio.io.DatasetReader) -> Optional[str]:
    tags = {key.lower(): value.strip() for key, value in dataset.tags().items()}
    for key in ("acquisition_date", "datetime", "date", "sensing_time"):
        value = tags.get(key)
        if value:
            try:
                return date.fromisoformat(value[:10]).isoformat()
            except ValueError:
                return None
    return None


def _raster_info(path: Path, *, require_acquisition_date: bool = False) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Only local GeoTIFF input rasters are supported: {path}")
    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.transform.is_identity:
            raise ValueError(f"Raster lacks usable CRS/affine georeferencing: {path}")
        if dataset.width <= 0 or dataset.height <= 0 or dataset.count <= 0:
            raise ValueError(f"Raster has invalid dimensions/band count: {path}")
        acquisition_date = _acquisition_date(dataset)
        if require_acquisition_date and acquisition_date is None:
            raise ValueError(f"Raster has no valid acquisition_date tag: {path}")
        bounds = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
        return {
            "path": str(path),
            "crs": dataset.crs.to_string(),
            "transform": dataset.transform,
            "resolution": tuple(float(value) for value in dataset.res),
            "width": dataset.width,
            "height": dataset.height,
            "bounds": tuple(float(value) for value in dataset.bounds),
            "bounds_wgs84": tuple(float(value) for value in bounds),
            "nodata": dataset.nodata,
            "band_count": dataset.count,
            "band_names": [value or None for value in dataset.descriptions],
            "acquisition_date": acquisition_date,
            "source_tags": dataset.tags(),
        }


def _file_provenance_record(
    metadata: Mapping[str, Any],
    manifest_field: str,
    path: Path,
) -> Optional[Dict[str, Any]]:
    resolved_path = path.resolve()
    for record in metadata.get("inputs", []):
        if record.get("manifest_field") != manifest_field:
            continue
        record_path = record.get("path") or record.get("filename")
        if not record_path:
            continue
        candidate = Path(str(record_path))
        if not candidate.is_absolute():
            candidate = Path(metadata["path"]).parent.parent / candidate
        if candidate.resolve() == resolved_path:
            return record
    return None


def _validate_file_provenance(
    record: Optional[Mapping[str, Any]],
    raster_info: Mapping[str, Any],
    event_start: Optional[date],
    event_end: Optional[date],
    required_fields: Sequence[str],
    *,
    require_provenance: bool = True,
) -> List[str]:
    if record is None:
        if not require_provenance:
            return []
        return ["metadata.json has no provenance record for this local input file"]
    missing = [field for field in required_fields if record.get(field) in (None, "", [], {})]
    if missing:
        return [f"metadata.json provenance is missing {', '.join(missing)}"]
    source_url = str(record.get("source_url", ""))
    if not source_url.startswith(("https://", "http://")):
        return ["metadata.json source_url must be an explicit HTTP(S) source URL"]
    try:
        metadata_acquisition = date.fromisoformat(str(record["acquisition_date"])[:10])
        metadata_event_date = date.fromisoformat(str(record["event_date"])[:10])
    except ValueError:
        return ["metadata.json acquisition_date and event_date must be ISO dates"]
    if raster_info.get("acquisition_date") != metadata_acquisition.isoformat():
        return ["metadata.json acquisition_date does not match the raster acquisition_date tag"]
    if event_start and metadata_event_date != event_start:
        return ["metadata.json event_date does not match the verified manifest event date"]
    if event_start and event_end and not event_start <= metadata_acquisition <= event_end and metadata_event_date == metadata_acquisition:
        return ["metadata.json acquisition/event dates conflict with the historical event window"]
    if str(record.get("crs")) != raster_info["crs"]:
        return ["metadata.json CRS does not match the GeoTIFF CRS"]
    if not np.allclose(
        np.asarray(record.get("resolution"), dtype=float),
        np.asarray(raster_info["resolution"], dtype=float),
        rtol=1e-6,
        atol=1e-8,
    ):
        return ["metadata.json resolution does not match the GeoTIFF resolution"]
    recorded_bands = [_normalized_band_name(item) for item in record.get("bands", [])]
    actual_bands = [_normalized_band_name(item) for item in raster_info.get("band_names", [])]
    if not actual_bands or recorded_bands != actual_bands:
        return ["metadata.json bands do not match GeoTIFF band descriptions"]
    return []


def _hazard_schema(row: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    hazard = (_value(row, "hazard") or "").lower()
    schema = (config.get("hazards") or {}).get(hazard)
    if not isinstance(schema, dict):
        raise ValueError(f"No event input schema is configured for hazard {hazard!r}.")
    channels = schema.get("channels") or []
    expected_channels = int(config.get("model_input_channels", 0))
    if expected_channels <= 0 or len(channels) != expected_channels:
        raise ValueError(
            f"Hazard {hazard!r} maps {len(channels)} channels but model_input_channels is {expected_channels}."
        )
    if len({channel.get("name") for channel in channels}) != len(channels):
        raise ValueError(f"Hazard {hazard!r} has duplicate feature channel names.")
    return schema


def _parse_aoi(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise ValueError
        bounds = tuple(float(item) for item in parsed)
    except (ValueError, TypeError, json.JSONDecodeError):
        raise ValueError("aoi must be a verified WGS84 bbox JSON array [min_lon,min_lat,max_lon,max_lat].") from None
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("aoi bbox coordinates are outside valid WGS84 bounds.")
    return bounds


def _verified_location(row: Mapping[str, Any], aoi: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    latitude = _value(row, "latitude")
    longitude = _value(row, "longitude")
    if latitude is None or longitude is None:
        return None
    try:
        latitude_value, longitude_value = float(latitude), float(longitude)
    except ValueError:
        raise ValueError("latitude and longitude must be numeric when provided.") from None
    if not (-90 <= latitude_value <= 90 and -180 <= longitude_value <= 180):
        raise ValueError("Verified event location is outside geographic bounds.")
    if aoi is not None:
        min_lon, min_lat, max_lon, max_lat = aoi
        if not (min_lon <= longitude_value <= max_lon and min_lat <= latitude_value <= max_lat):
            raise ValueError("Verified event location does not lie inside the manifest AOI.")
    return latitude_value, longitude_value


def _ground_truth_status(
    row: Mapping[str, Any],
    root: Path,
    event_metadata: Mapping[str, Any],
) -> Tuple[bool, str, List[str]]:
    value = _value(row, "ground_truth")
    missing: List[str] = []
    if value is not None:
        paths = _local_paths(value, root)
        if not paths:
            missing.append("ground_truth: remote references are not loaded automatically")
            return False, "not_validated", missing
        spatial_extensions = {".tif", ".tiff", ".json", ".geojson"}
        spatial_paths = [path for path in paths if path.suffix.lower() in spatial_extensions]
        absent = [str(path) for path in spatial_paths if not path.is_file()]
        if absent:
            missing.extend(absent)
        if spatial_paths and not absent:
            record = event_metadata.get("ground_truth")
            if not isinstance(record, dict):
                return True, "spatial", missing
            required = ("source", "source_url", "license", "observation_date", "crs", "method")
            if any(not record.get(field) for field in required):
                return False, "not_validated", [
                    "ground_truth metadata.json entry must verify source, source_url, license, observation_date, CRS, and method"
                ]
            if not str(record["source_url"]).startswith(("https://", "http://")):
                return False, "not_validated", ["ground_truth source_url must be an explicit HTTP(S) URL"]
            try:
                date.fromisoformat(str(record["observation_date"])[:10])
            except ValueError:
                return False, "not_validated", ["ground_truth observation_date must be an ISO date"]
            path = spatial_paths[0]
            if str(record.get("path") or record.get("filename")):
                recorded_path = Path(str(record.get("path") or record.get("filename")))
                if not recorded_path.is_absolute():
                    recorded_path = root / recorded_path
                if recorded_path.resolve() != path.resolve():
                    return False, "not_validated", ["ground_truth metadata path does not match manifest path"]
            return True, "spatial", missing
    area_value = _value(row, "ground_truth_area")
    if area_value is not None:
        try:
            float(area_value)
        except ValueError:
            missing.append("ground_truth_area: expected a verified numeric area or unavailable value")
        else:
            return False, "area_only", missing
    if not missing:
        missing.append("independent spatial ground truth is not available")
    return False, "not_validated", missing


def check_event_readiness(
    event_id: str,
    *,
    manifest_path: Optional[Any] = None,
    data_root: Optional[Any] = None,
    event_data_config_path: Optional[Any] = None,
    model_config_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Check hazard-specific required inputs, optional inputs, and independent truth."""
    row = get_event(event_id, manifest_path)
    root = Path(data_root) if data_root is not None else _project_root()
    config = _event_data_config(event_data_config_path)
    hazard = (_value(row, "hazard") or "").lower()
    missing_inputs: List[str] = []
    optional_missing_features: List[str] = []
    available_inputs: List[str] = []
    event_metadata = _load_event_file_metadata(event_id, manifest_path)
    try:
        schema = _hazard_schema(row, config)
    except ValueError as exc:
        schema = {}
        missing_inputs.append(str(exc))

    model_config_file = Path(model_config_path) if model_config_path is not None else _project_root() / "configs" / "model_config.yaml"
    model_config: Dict[str, Any] = {}
    if model_config_file.is_file():
        with model_config_file.open(encoding="utf-8") as model_file:
            model_config = yaml.safe_load(model_file) or {}
    actual_model_channels = int(model_config.get("model", {}).get("num_input_channels", 0))
    configured_channels = int(config.get("model_input_channels", 0))
    if not configured_channels or configured_channels != actual_model_channels:
        missing_inputs.append(
            f"event_data.yaml maps {configured_channels} channels but model config requires {actual_model_channels}"
        )

    parsed_dates: Dict[str, Optional[date]] = {}
    for date_field in ("start_date", "end_date"):
        try:
            parsed_date = _parse_manifest_date(_value(row, date_field), date_field)
        except ValueError as exc:
            missing_inputs.append(str(exc))
            parsed_date = None
        if parsed_date is None:
            missing_inputs.append(f"{date_field} is not verified")
        parsed_dates[date_field] = parsed_date
    start_date, end_date = parsed_dates.get("start_date"), parsed_dates.get("end_date")
    if start_date and end_date and start_date > end_date:
        missing_inputs.append("start_date is later than end_date")

    aoi_value = _value(row, "aoi")
    aoi = None
    if aoi_value is not None and aoi_value.strip().startswith("["):
        try:
            aoi = _parse_aoi(aoi_value)
        except ValueError as exc:
            aoi = None
            missing_inputs.append(str(exc))
    if aoi_value is not None and aoi is not None:
        try:
            event_location = _verified_location(row, aoi)
        except ValueError as exc:
            event_location = None
            missing_inputs.append(str(exc))
    else:
        event_location = None
    if _value(row, "source") is None:
        missing_inputs.append("source is not verified")
    source_url = _value(row, "source_url")
    if source_url is not None and not source_url.startswith(("https://", "http://")):
        normalized = source_url.lower()
        if not normalized.startswith(("urn:", "file:", "local:")):
            missing_inputs.append("source_url is not a verified HTTP(S) URL")
    if (_value(row, "data_status") or "").lower() != "verified":
        missing_inputs.append("data_status is not verified")

    metadata_by_path: Dict[str, Dict[str, Any]] = {}
    metadata_exists = Path(event_metadata.get("path", "")).is_file() if event_metadata.get("path") else False
    declared_but_not_consumed: List[str] = []
    schema_fields = set(schema.get("required_inputs", [])) | set(schema.get("optional_inputs", []))
    for field in schema.get("unsupported_declared_inputs", []):
        if _value(row, field) is not None:
            declared_but_not_consumed.append(field)
            missing_inputs.append(f"{field} is declared_but_not_consumed: the configured model input contract does not consume it")
    for field in schema.get("required_inputs", []):
        if _value(row, field) is None:
            missing_inputs.append(f"required input {field} is not verified")
    for field in schema.get("optional_inputs", []):
        if _value(row, field) is None:
            optional_missing_features.append(field)

    for field in sorted(schema_fields):
        value = _value(row, field)
        if value is None:
            continue
        paths = _paths_for_row(row, field, root)
        if not paths:
            note = f"{field} contains no local file path"
            if field in schema.get("required_inputs", []):
                missing_inputs.append(note)
            else:
                optional_missing_features.append(field)
            continue
        if len(paths) != 1:
            missing_inputs.append(f"{field} must point to one verified, aligned GeoTIFF mosaic")
            continue
        path = paths[0]
        try:
            info = _raster_info(path, require_acquisition_date=field in IMAGE_FIELDS)
        except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
            if field in schema.get("required_inputs", []):
                missing_inputs.append(f"{field}: {exc}")
            else:
                optional_missing_features.append(f"{field}: {exc}")
            continue
        record = _file_provenance_record(event_metadata, field, path)
        provenance_fields = config.get("required_raster_provenance", [])
        provenance_errors = _validate_file_provenance(
            record,
            info,
            start_date,
            end_date,
            provenance_fields,
            require_provenance=metadata_exists,
        )
        if provenance_errors:
            if field in schema.get("required_inputs", []):
                missing_inputs.extend(f"{field}: {error}" for error in provenance_errors)
            else:
                optional_missing_features.extend(f"{field}: {error}" for error in provenance_errors)
            continue
        metadata_by_path[field] = {"path": path, "info": info, "provenance": record}
        available_inputs.append(field)
        acquisition = date.fromisoformat(info["acquisition_date"]) if info["acquisition_date"] else None
        if acquisition:
            if field == "pre_event_data" and start_date and acquisition >= start_date:
                missing_inputs.append(f"{field} acquisition {acquisition} is not before the event start date")
            elif field == "during_event_data" and start_date and end_date and not start_date <= acquisition <= end_date:
                missing_inputs.append(f"{field} acquisition {acquisition} is outside the historical event dates")
            elif field == "post_event_data" and end_date and acquisition <= end_date:
                missing_inputs.append(f"{field} acquisition {acquisition} is not after the event end date")

    channel_availability: Dict[str, bool] = {}
    for channel in schema.get("channels", []):
        name = str(channel.get("name", ""))
        required = bool(channel.get("required", True))
        available = False
        source = channel.get("source")
        if source:
            field = source.get("field")
            if field not in metadata_by_path:
                optional_missing_features.append(name)
            else:
                info = metadata_by_path[field]["info"]
                band_index = source.get("band_index")
                band_name = source.get("band")
                if band_index is not None:
                    available = 1 <= int(band_index) <= info["band_count"]
                elif band_name:
                    wanted = _normalized_band_name(band_name)
                    actual = [_normalized_band_name(item) for item in info["band_names"]]
                    available = wanted in actual
                if not available:
                    optional_missing_features.append(name)
        elif channel.get("derive"):
            dependencies = channel.get("inputs", [])
            available = bool(dependencies) and all(channel_availability.get(item, False) for item in dependencies)
            if not available:
                optional_missing_features.append(name)
        else:
            missing_inputs.append(f"channel {name!r} has no source or derivation")
        channel_availability[name] = available
        if required and not available:
            missing_inputs.append(f"required model feature {name!r} is unavailable in its configured source")

    for name, feature in (schema.get("geospatial_features") or {}).items():
        if feature.get("derive") == "aspect_from_dem":
            source = feature.get("source", {})
            available = source.get("field") in metadata_by_path
        elif feature.get("derive"):
            available = all(channel_availability.get(item, False) for item in feature.get("inputs", []))
        else:
            source = feature.get("source", {})
            available = source.get("field") in metadata_by_path
            if available and source.get("band"):
                info = metadata_by_path[source["field"]]["info"]
                available = _normalized_band_name(source["band"]) in [
                    _normalized_band_name(item) for item in info["band_names"]
                ]
            if available and source.get("band_index"):
                info = metadata_by_path[source["field"]]["info"]
                available = 1 <= int(source["band_index"]) <= info["band_count"]
        if not available:
            if feature.get("required", False):
                missing_inputs.append(f"required geospatial feature {name!r} is unavailable")
            else:
                optional_missing_features.append(name)

    expected_channels = configured_channels
    for field in schema.get("required_inputs", []):
        if field in metadata_by_path and field in IMAGE_FIELDS:
            info = metadata_by_path[field]["info"]
            if info["band_count"] < 1:
                missing_inputs.append(f"{field} GeoTIFF contains no bands")

    ground_truth_available, ground_truth_status, missing_ground_truth = _ground_truth_status(
        row, root, event_metadata
    )
    missing_inputs = list(dict.fromkeys(missing_inputs))
    optional_missing_features = list(dict.fromkeys(optional_missing_features))
    missing_ground_truth = list(dict.fromkeys(missing_ground_truth))
    inputs_available = not missing_inputs
    status = _canonical_readiness_status(inputs_available, ground_truth_available, ground_truth_status)
    return {
        "event_id": event_id,
        "inputs_available": inputs_available,
        "ground_truth_available": ground_truth_available,
        "ground_truth_status": ground_truth_status,
        "readiness_status": status,
        "legacy_status": LEGACY_READINESS_STATUS_MAP.get(status, status),
        "required_inputs": list(schema.get("required_inputs", [])),
        "available_inputs": sorted(set(available_inputs)),
        "missing_inputs": missing_inputs,
        "optional_missing_features": optional_missing_features,
        "missing_ground_truth": missing_ground_truth,
        "declared_but_not_consumed": sorted(set(declared_but_not_consumed)),
        "status": status,
        "hazard": _value(row, "hazard"),
        "event_name": _value(row, "event_name"),
        "event_date": _value(row, "start_date"),
        "end_date": _value(row, "end_date"),
        "source": _value(row, "source"),
        "source_url": _value(row, "source_url"),
    }


def select_first_data_ready_event(
    manifest_path: Optional[Any] = None,
    data_root: Optional[Any] = None,
) -> Optional[str]:
    """Select the first input- and spatial-ground-truth-ready benchmark event by priority."""
    rows = load_event_manifest(manifest_path)
    priority = ("L1", "F2", "F3", "F1", "L2", "L3", "W1", "W2", "W3")
    ordered = [event_id for event_id in priority if event_id in rows]
    ordered.extend(event_id for event_id in rows if event_id not in ordered)
    for event_id in ordered:
        status = check_event_readiness(event_id, manifest_path=manifest_path, data_root=data_root)
        if status["readiness_status"] in {"data_ready", "validation_ready"}:
            return event_id
    return None


def _require_paths(row: Mapping[str, Any], field: str, root: Path) -> List[Path]:
    paths = _paths_for_row(row, field, root)
    if not paths or any(not path.is_file() for path in paths):
        raise EventNotReadyError({
            "event_id": row.get("event_id"),
            "missing_inputs": [f"{field} is missing one or more local raster files"],
        })
    return paths


def _dataset_grid_metadata(dataset: rasterio.io.DatasetReader) -> Dict[str, Any]:
    if dataset.crs is None:
        raise ValueError(f"Raster lacks CRS: {dataset.name}")
    return {
        "crs": dataset.crs.to_string(),
        "transform": dataset.transform,
        "resolution": tuple(float(value) for value in dataset.res),
        "width": dataset.width,
        "height": dataset.height,
        "bounds": tuple(dataset.bounds),
        "nodata": dataset.nodata,
    }


def _read_aligned_input(
    paths: Sequence[Path],
    reference_grid: Mapping[str, Any],
    feature_type: str = "continuous",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
    arrays = []
    validities = []
    source_records = []
    band_names = []
    for path in paths:
        with rasterio.open(path) as dataset:
            values = dataset.read()
            source_metadata = _dataset_grid_metadata(dataset)
            descriptions = [name or "" for name in dataset.descriptions]
            acquisition_date = _acquisition_date(dataset)
            aligned = align_raster_to_reference(
                values,
                source_metadata,
                reference_grid,
                feature_type=feature_type,
            )
            array = aligned.data if aligned.data.ndim == 3 else aligned.data[np.newaxis, ...]
            arrays.append(array.astype(np.float32) if feature_type != "geology" else array)
            validities.append(aligned.valid_mask)
            band_names.extend(name.lower() for name in descriptions)
            source_records.append({
                "path": str(path),
                "acquisition_date": acquisition_date,
                "source_crs": source_metadata["crs"],
                "target_crs": aligned.metadata["crs"],
                "resolution": source_metadata["resolution"],
                "bands": descriptions,
                "nodata": source_metadata["nodata"],
            })
    return np.concatenate(arrays, axis=0), np.logical_and.reduce(validities), source_records, band_names


def _valid_mean(values: np.ndarray, valid: np.ndarray, name: str) -> float:
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        raise EventNotReadyError({"missing_inputs": [f"{name} has no valid pixels"]})
    return float(selected.mean())


def _derive_aspect(elevation: np.ndarray, valid: np.ndarray, resolution: Sequence[float]) -> float:
    if len(resolution) != 2 or min(abs(float(value)) for value in resolution) <= 0:
        raise ValueError("DEM has invalid pixel resolution for aspect derivation.")
    dy, dx = np.gradient(elevation.astype(np.float64), abs(float(resolution[1])), abs(float(resolution[0])))
    aspect = np.arctan2(-dx, dy)
    selected = aspect[valid & np.isfinite(aspect)]
    if selected.size == 0:
        raise EventNotReadyError({"missing_inputs": ["aspect cannot be derived from DEM"]})
    return float(np.degrees(np.arctan2(np.sin(selected).mean(), np.cos(selected).mean())) % 360.0)


def _read_aligned_configured_band(
    field: str,
    source: Mapping[str, Any],
    metadata_by_field: Mapping[str, Mapping[str, Any]],
    reference_grid: Mapping[str, Any],
    feature_type: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if field not in metadata_by_field:
        raise FileNotFoundError(f"{field} is unavailable.")
    path = metadata_by_field[field]["path"]
    with rasterio.open(path) as dataset:
        if "band_index" in source:
            band_index = int(source["band_index"])
            if not 1 <= band_index <= dataset.count:
                raise ValueError(f"{field} raster does not contain configured band index {band_index}.")
        else:
            requested_band = _normalized_band_name(source.get("band", ""))
            described_bands = [_normalized_band_name(name or "") for name in dataset.descriptions]
            if not requested_band or requested_band not in described_bands:
                raise ValueError(
                    f"{field} raster is missing configured band {source.get('band')!r}; "
                    "band descriptions are required and band order is not guessed."
                )
            band_index = described_bands.index(requested_band) + 1
        values = dataset.read(band_index)
        source_metadata = {
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": dataset.transform,
            "resolution": tuple(float(value) for value in dataset.res),
            "width": dataset.width,
            "height": dataset.height,
            "nodata": dataset.nodata,
        }
        band_description = dataset.descriptions[band_index - 1]

    aligned = align_raster_to_reference(
        values,
        source_metadata,
        reference_grid,
        feature_type=feature_type,
    )
    return np.asarray(aligned.data), aligned.valid_mask, {
        "field": field,
        "path": str(path),
        "band": band_description,
        "source_crs": source_metadata["crs"],
        "target_crs": aligned.metadata["crs"],
        "resolution": source_metadata["resolution"],
        "nodata": source_metadata["nodata"],
        "acquisition_date": metadata_by_field[field]["info"]["acquisition_date"],
        "provenance": metadata_by_field[field]["provenance"],
    }


def _calculate_derived_feature(
    operation: str,
    input_names: Sequence[str],
    feature_values: Mapping[str, np.ndarray],
    feature_validity: Mapping[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not input_names or any(name not in feature_values for name in input_names):
        return None, None
    inputs = [feature_values[name].astype(np.float32, copy=False) for name in input_names]
    validities = [feature_validity[name] for name in input_names]
    valid = np.logical_and.reduce(validities)
    if operation in {"ndvi", "nbr"}:
        first, second = inputs
        denominator = first + second
        valid &= np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
        output = np.zeros(first.shape, dtype=np.float32)
        np.divide(first - second, denominator, out=output, where=valid)
    elif operation == "difference":
        if len(inputs) != 2:
            raise ValueError("difference derivation requires exactly two input features.")
        output = inputs[0] - inputs[1]
        valid &= np.isfinite(output)
    else:
        raise ValueError(f"Unsupported derived feature operation {operation!r}.")
    return output, valid


def _optional_fill(
    name: str,
    shape: Tuple[int, int],
    fill_value: float,
    missing_optional_features: List[str],
) -> np.ndarray:
    missing_optional_features.append(name)
    return np.full(shape, fill_value, dtype=np.float32)


def load_event_inputs(
    event_id: str,
    *,
    manifest_path: Optional[Any] = None,
    data_root: Optional[Any] = None,
    event_data_config_path: Optional[Any] = None,
    model_config_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a configured hazard-specific model tensor from verified local inputs only."""
    readiness = check_event_readiness(
        event_id,
        manifest_path=manifest_path,
        data_root=data_root,
        event_data_config_path=event_data_config_path,
        model_config_path=model_config_path,
    )
    if not readiness["inputs_available"]:
        raise EventNotReadyError(readiness)
    row = get_event(event_id, manifest_path)
    root = Path(data_root) if data_root is not None else _project_root()
    config = _event_data_config(event_data_config_path)
    schema = _hazard_schema(row, config)
    event_metadata = _load_event_file_metadata(event_id, manifest_path)
    reference_field = schema["reference_input"]
    reference_paths = _require_paths(row, reference_field, root)
    if len(reference_paths) != 1:
        raise EventNotReadyError({
            "event_id": event_id,
            "missing_inputs": [f"reference input {reference_field} must be one verified mosaic raster"],
        })
    reference_path = reference_paths[0]
    with rasterio.open(reference_path) as reference_dataset:
        reference_grid = _dataset_grid_metadata(reference_dataset)
    metadata_by_field: Dict[str, Dict[str, Any]] = {}
    input_sources: List[Dict[str, Any]] = []
    fields_needed = set(schema.get("required_inputs", [])) | set(schema.get("optional_inputs", []))
    for channel in schema["channels"]:
        source = channel.get("source") or {}
        if source.get("field"):
            fields_needed.add(source["field"])
    for feature in (schema.get("geospatial_features") or {}).values():
        source = feature.get("source") or {}
        if source.get("field"):
            fields_needed.add(source["field"])
    for field in sorted(fields_needed):
        paths = _paths_for_row(row, field, root)
        if not paths:
            continue
        if len(paths) != 1:
            raise EventNotReadyError({
                "event_id": event_id,
                "missing_inputs": [f"{field} must identify exactly one verified mosaic GeoTIFF"],
            })
        path = paths[0]
        info = _raster_info(path, require_acquisition_date=field in IMAGE_FIELDS)
        provenance = _file_provenance_record(event_metadata, field, path)
        metadata_by_field[field] = {"path": path, "info": info, "provenance": provenance}
        input_sources.append({"field": field, **info, "provenance": provenance})

    missing_optional_features = list(readiness["optional_missing_features"])
    feature_values: Dict[str, np.ndarray] = {}
    feature_validity: Dict[str, np.ndarray] = {}
    channel_specs = {channel["name"]: channel for channel in schema["channels"]}
    channel_values: Dict[str, np.ndarray] = {}
    channel_validity: Dict[str, np.ndarray] = {}
    fill_value = float(config.get("optional_channel_fill_value", 0.0))

    for channel in schema["channels"]:
        name = channel["name"]
        required = bool(channel.get("required", True))
        values: Optional[np.ndarray] = None
        valid: Optional[np.ndarray] = None
        try:
            if channel.get("source"):
                source_spec = channel["source"]
                field = source_spec["field"]
                values, valid, _ = _read_aligned_configured_band(
                    field,
                    source_spec,
                    metadata_by_field,
                    reference_grid,
                    channel.get("feature_type", "continuous"),
                )
                if not valid.all():
                    if required or config.get("require_full_valid_coverage", True):
                        raise ValueError("source band contains nodata pixels the model cannot mask")
                    values = np.where(valid, values, fill_value)
                    missing_optional_features.append(f"{name}:nodata_pixels")
                    valid = np.ones(valid.shape, dtype=bool)
                scale = channel.get("scale")
                if scale is not None:
                    if float(scale) <= 0:
                        raise ValueError(f"invalid configured scale for {name}")
                    values = values.astype(np.float32) / float(scale)
            elif channel.get("derive"):
                dependencies = channel.get("inputs", [])
                values, valid = _calculate_derived_feature(
                    channel["derive"], dependencies, feature_values, feature_validity
                )
                if values is not None and (valid is None or not valid.all()):
                    if required or config.get("require_full_valid_coverage", True):
                        raise ValueError("derived channel contains invalid source pixels")
                    values = None
            else:
                raise ValueError("channel has neither source nor derivation")
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            if required:
                raise EventNotReadyError({
                    "event_id": event_id,
                    "missing_inputs": [f"required feature {name}: {exc}"],
                }) from exc
            values = None

        if values is None:
            if required:
                raise EventNotReadyError({
                    "event_id": event_id,
                    "missing_inputs": [f"required feature {name} cannot be constructed"],
                })
            values = np.full(
                (reference_grid["height"], reference_grid["width"]),
                fill_value,
                dtype=np.float32,
            )
            valid = np.ones(values.shape, dtype=bool)
            missing_optional_features.append(name)
        channel_values[name] = np.asarray(values, dtype=np.float32)
        channel_validity[name] = np.asarray(valid, dtype=bool)
        feature_values[name] = channel_values[name]
        feature_validity[name] = channel_validity[name]

    channel_tensors = [torch.from_numpy(np.ascontiguousarray(channel_values[channel["name"]])) for channel in schema["channels"]]
    image_tensor = torch.stack(channel_tensors, dim=0).unsqueeze(0).unsqueeze(0)

    geospatial_values: Dict[str, float] = {}
    geospatial_schema = schema.get("geospatial_features") or {}
    for name, feature in geospatial_schema.items():
        required = bool(feature.get("required", False))
        value: Optional[float] = None
        try:
            if feature.get("source"):
                source_spec = feature["source"]
                values, valid, _ = _read_aligned_configured_band(
                    source_spec["field"],
                    source_spec,
                    metadata_by_field,
                    reference_grid,
                    "categorical" if feature.get("categorical") else "continuous",
                )
                if not valid.all():
                    raise ValueError("source feature contains nodata pixels")
                if feature.get("categorical"):
                    valid_values = values[valid]
                    if not np.allclose(valid_values, np.round(valid_values)):
                        raise ValueError("categorical feature contains non-integer classes")
                    classes, counts = np.unique(valid_values.astype(np.int64), return_counts=True)
                    value = float(classes[np.argmax(counts)])
                else:
                    scale = float(feature.get("scale", 1.0))
                    value = _valid_mean(values.astype(np.float32) / scale, valid, name)
            elif feature.get("derive") == "aspect_from_dem":
                source_spec = feature["source"]
                dem, valid, _ = _read_aligned_configured_band(
                    source_spec["field"], source_spec, metadata_by_field, reference_grid, "continuous"
                )
                if not valid.all():
                    raise ValueError("DEM used for aspect contains nodata pixels")
                value = _derive_aspect(dem, valid, reference_grid["resolution"])
            elif feature.get("derive"):
                derived, valid = _calculate_derived_feature(
                    feature["derive"], feature.get("inputs", []), feature_values, feature_validity
                )
                if derived is not None and valid is not None and valid.any():
                    value = _valid_mean(derived, valid, name)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            if required:
                raise EventNotReadyError({
                    "event_id": event_id,
                    "missing_inputs": [f"required geospatial feature {name}: {exc}"],
                }) from exc

        if value is None:
            if required:
                raise EventNotReadyError({
                    "event_id": event_id,
                    "missing_inputs": [f"required geospatial feature {name} cannot be constructed"],
                })
            value = 0.0
            missing_optional_features.append(name)
        if name == "lithology_class" and not 0 <= int(value) < 100:
            if required:
                raise EventNotReadyError({
                    "event_id": event_id,
                    "missing_inputs": ["lithology class must be in the existing encoder range [0, 99]"],
                })
            missing_optional_features.append("lithology_class:outside_encoder_range")
            value = 0.0
        geospatial_values[name] = value

    expected_channels = int(config["model_input_channels"])
    if image_tensor.shape[2] != expected_channels:
        raise EventNotReadyError({
            "event_id": event_id,
            "missing_inputs": [
                f"configured adapter produced {image_tensor.shape[2]} channels; model requires {expected_channels}"
            ],
        })

    center_x, center_y = reference_grid["transform"] @ (
        reference_grid["width"] / 2,
        reference_grid["height"] / 2,
    )
    source_crs = CRS.from_user_input(reference_grid["crs"])
    if source_crs != CRS.from_epsg(4326):
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = transformer.transform(center_x, center_y)
    else:
        center_lon, center_lat = center_x, center_y
    import h3
    h3_indices = {}
    for resolution in (4, 6, 8):
        if hasattr(h3, "geo_to_h3"):
            cell = h3.geo_to_h3(center_lat, center_lon, resolution)
        else:
            cell = h3.latlng_to_cell(center_lat, center_lon, resolution)
        h3_indices[resolution] = int(cell, 16) % 10000

    device = torch.device("cpu")
    model_inputs = {
        "images": image_tensor.to(device),
        "quality_flags": torch.ones((1, 1), dtype=torch.bool, device=device),
        "h3_indices": {
            resolution: torch.tensor([[index]], dtype=torch.long, device=device)
            for resolution, index in h3_indices.items()
        },
        "geo_attributes": {
            "elevation": torch.tensor([[geospatial_values.get("elevation", 0.0)]], dtype=torch.float32, device=device),
            "slope": torch.tensor([[geospatial_values.get("slope", 0.0)]], dtype=torch.float32, device=device),
            "ndvi": torch.tensor([[geospatial_values.get("ndvi", 0.0)]], dtype=torch.float32, device=device),
            "aspect": torch.tensor([[geospatial_values.get("aspect", 0.0)]], dtype=torch.float32, device=device),
            "lithology_class": torch.tensor([int(geospatial_values.get("lithology_class", 0))], dtype=torch.long, device=device),
        },
    }
    return {
        "model_inputs": model_inputs,
        "metadata": {
            "event_id": event_id,
            "hazard": readiness["hazard"],
            "event_name": readiness["event_name"],
            "event_date": readiness["event_date"],
            "end_date": readiness["end_date"],
            "aoi": _value(row, "aoi"),
            "reference_grid": reference_grid,
            "channel_order": [channel["name"] for channel in schema["channels"]],
            "missing_optional_features": sorted(set(missing_optional_features)),
            "input_sources": input_sources,
            "manifest_source": _value(row, "source"),
            "manifest_source_url": _value(row, "source_url"),
            "used_input_fields": sorted(metadata_by_field),
            "feature_schema": schema,
            "provenance_metadata_path": event_metadata.get("path"),
        },
    }


def load_ground_truth(
    event_id: str,
    *,
    manifest_path: Optional[Any] = None,
    data_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load independent spatial truth separately; never called by model-input preparation."""
    row = get_event(event_id, manifest_path)
    root = Path(data_root) if data_root is not None else _project_root()
    paths = _paths_for_row(row, "ground_truth", root)
    if paths:
        if len(paths) != 1:
            raise ValueError("Ground truth must identify one GeoTIFF or GeoJSON artifact.")
        path = paths[0]
        if not path.is_file():
            return {"available": False, "status": "not_validated", "missing": [str(path)]}
        if path.suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(path) as dataset:
                if dataset.crs is None or dataset.transform.is_identity:
                    raise ValueError(f"Ground-truth raster lacks spatial metadata: {path}")
                return {
                    "available": True,
                    "status": "spatial_raster",
                    "path": str(path),
                    "mask": dataset.read(1),
                    "metadata": {
                        "crs": dataset.crs.to_string(),
                        "transform": dataset.transform,
                        "resolution": tuple(float(value) for value in dataset.res),
                        "width": dataset.width,
                        "height": dataset.height,
                        "nodata": dataset.nodata,
                    },
                }
        if path.suffix.lower() in {".json", ".geojson"}:
            with path.open(encoding="utf-8") as geojson_file:
                document = json.load(geojson_file)
            raw_crs = document.get("crs", {}).get("properties", {}).get("name") if isinstance(document.get("crs"), dict) else None
            ground_truth_crs = CRS.from_user_input(raw_crs or "EPSG:4326")
            if document.get("type") == "FeatureCollection":
                geometries = [shape(feature["geometry"]) for feature in document.get("features", []) if feature.get("geometry")]
            elif document.get("type") == "Feature" and document.get("geometry"):
                geometries = [shape(document["geometry"])]
            else:
                geometries = [shape(document)] if document.get("type") in {"Polygon", "MultiPolygon"} else []
            geometry = unary_union(geometries) if geometries else None
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                raise ValueError(f"Ground-truth GeoJSON contains no valid spatial geometry: {path}")
            return {
                "available": True,
                "status": "spatial_polygon",
                "path": str(path),
                "geometry": geometry,
                "crs": ground_truth_crs.to_string(),
            }
        raise ValueError(f"Unsupported ground-truth format: {path.suffix}")

    area_value = _value(row, "ground_truth_area")
    if area_value is not None:
        try:
            area = float(area_value)
        except ValueError:
            return {"available": False, "status": "not_validated", "missing": ["ground_truth_area is not numeric"]}
        return {"available": False, "status": "area_only", "area_km2": area}
    return {"available": False, "status": "not_validated", "missing": ["independent spatial ground truth is unavailable"]}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Affine):
        return list(value)[:6]
    if isinstance(value, CRS):
        return value.to_string()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_safe(value), output_file, indent=2, allow_nan=False)


def _write_logits(path: Path, logits: torch.Tensor, transform: Affine, crs: Any) -> None:
    values = logits.detach().cpu().float().numpy()
    if values.ndim != 2:
        raise ValueError("One event logits must have shape [H,W].")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dataset:
        dataset.write(values, 1)


def route_hazard_logits(segmentation_outputs: Mapping[str, Any], hazard: str) -> Any:
    """Select exactly the segmentation head named by the manifest hazard."""
    key = HAZARD_HEADS.get(hazard.lower())
    if key is None:
        raise ValueError(f"Unsupported hazard {hazard!r}.")
    if key not in segmentation_outputs:
        raise KeyError(f"Model outputs do not contain the required {key!r} segmentation head.")
    return segmentation_outputs[key]


def write_prediction_artifacts(
    output_dir: Any,
    logits: Any,
    hazard: str,
    grid_metadata: Mapping[str, Any],
    *,
    threshold: float = 0.5,
    input_metadata: Optional[Mapping[str, Any]] = None,
    **postprocess_options: Any,
) -> Dict[str, Any]:
    """Convert one routed logit map into Phase 6 spatial artifacts and input metadata."""
    transform = grid_metadata.get("transform")
    crs = grid_metadata.get("crs")
    if transform is None or crs is None:
        raise ValueError("Prediction artifact writing requires a reference CRS and affine transform.")
    prediction = process_segmentation_prediction(
        logits,
        hazard,
        transform,
        crs,
        threshold=threshold,
        **postprocess_options,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = save_spatial_outputs(output_path, prediction, transform, crs)
    logits_path = output_path / "prediction_logits.tif"
    _write_logits(logits_path, torch.as_tensor(logits), transform, crs)
    artifacts["logits_raster"] = str(logits_path)
    if input_metadata is not None:
        metadata_path = output_path / "input_metadata.json"
        _write_json(metadata_path, input_metadata)
        artifacts["input_metadata"] = str(metadata_path)
    return {"prediction": prediction, "artifacts": artifacts}


def reconstruct_event(
    event_id: str,
    *,
    checkpoint_path: Optional[Any] = None,
    config_path: Optional[Any] = None,
    event_data_config_path: Optional[Any] = None,
    manifest_path: Optional[Any] = None,
    data_root: Optional[Any] = None,
    results_root: Optional[Any] = None,
    model: Optional[Any] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one verified event reconstruction; incomplete events stop before model loading."""
    root = Path(data_root) if data_root is not None else _project_root()
    rows = load_event_manifest(manifest_path)
    if event_id not in rows:
        raise KeyError(f"Unknown event ID {event_id!r}; no matching manifest row exists.")
    row = rows[event_id]
    readiness = check_event_readiness(
        event_id,
        manifest_path=manifest_path,
        data_root=root,
        event_data_config_path=event_data_config_path,
        model_config_path=config_path,
    )
    output_root = Path(results_root) if results_root is not None else root / "results" / "events"
    event_output = output_root / event_id
    summary_path = event_output / "reconstruction_summary.json"
    event_metadata = {
        "event_id": event_id,
        "hazard": readiness.get("hazard"),
        "event_name": readiness.get("event_name"),
        "event_date": readiness.get("event_date"),
        "end_date": readiness.get("end_date"),
    }
    ground_truth_summary = {
        "available": readiness["ground_truth_available"],
        "status": readiness["ground_truth_status"],
    }
    if not readiness["inputs_available"]:
        summary = {
            **event_metadata,
            "input_sources": [],
            "missing_inputs": readiness["missing_inputs"],
            "prediction": None,
            "ground_truth": ground_truth_summary,
            "status": "not_ready",
            "legacy_status": LEGACY_READINESS_STATUS_MAP["not_ready"],
            "required_inputs": readiness["required_inputs"],
            "available_inputs": readiness["available_inputs"],
            "optional_missing_features": readiness["optional_missing_features"],
            "message": "Historical event inputs are not ready; no model inference was attempted.",
        }
        event_output.mkdir(parents=True, exist_ok=True)
        _write_json(summary_path, summary)
        return {"readiness": readiness, "summary": summary, "artifacts": {"reconstruction_summary": str(summary_path)}}

    bundle = load_event_inputs(
        event_id,
        manifest_path=manifest_path,
        data_root=root,
        event_data_config_path=event_data_config_path,
        model_config_path=config_path,
    )
    if model is None:
        if checkpoint_path is None:
            raise ValueError("A segmentation checkpoint is required for a ready event reconstruction.")
        config_file = Path(config_path) if config_path is not None else root / "configs" / "model_config.yaml"
        config = {}
        if config_file.is_file():
            import yaml
            with config_file.open(encoding="utf-8") as config_handle:
                config = yaml.safe_load(config_handle) or {}
        model_cfg = config.get("model", {})
        model = MultiHazardVLM(str(config_file) if config_file.is_file() else None)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not any(key.startswith("segmentation_head.") for key in state):
            raise ValueError("Checkpoint has no segmentation weights; refusing an untrained-footprint reconstruction.")
        model.load_state_dict(state, strict=True)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
    else:
        model_cfg = getattr(model, "config", {}).get("model", {})
        device = torch.device(device or "cpu")
        if hasattr(model, "to"):
            model.to(device)
    if not getattr(model, "segmentation_enabled", True):
        raise ValueError("The supplied model has segmentation disabled.")
    model.eval()
    model_inputs = bundle["model_inputs"]
    model_inputs = {
        key: (
            {subkey: tensor.to(device) for subkey, tensor in value.items()}
            if key in {"h3_indices", "geo_attributes"}
            else value.to(device) if isinstance(value, torch.Tensor) else value
        )
        for key, value in model_inputs.items()
    }
    with torch.no_grad():
        outputs = model(**model_inputs)
    hazard = str(readiness["hazard"]).lower()
    logits = route_hazard_logits(outputs["segmentation"], hazard)[0, 0]
    reference_grid = bundle["metadata"]["reference_grid"]
    output_transform = reference_grid["transform"] @ Affine.scale(
        reference_grid["width"] / logits.shape[1],
        reference_grid["height"] / logits.shape[0],
    )
    segmentation_cfg = model_cfg
    input_metadata = {
        **bundle["metadata"],
        "event_date": readiness["event_date"],
        "event_end_date": readiness["end_date"],
        "source": readiness.get("source"),
        "source_url": readiness.get("source_url"),
        "ground_truth_excluded_from_model_inputs": True,
    }
    written = write_prediction_artifacts(
        event_output,
        logits,
        hazard,
        {**reference_grid, "transform": output_transform},
        threshold=float(segmentation_cfg.get("segmentation_threshold", 0.5)),
        input_metadata=input_metadata,
        minimum_component_pixels=int(segmentation_cfg.get("segmentation_min_component_pixels", 0)),
        minimum_component_area_m2=float(segmentation_cfg.get("segmentation_min_component_area_m2", 0.0)),
        opening_radius=int(segmentation_cfg.get("segmentation_opening_radius", 0)),
        closing_radius=int(segmentation_cfg.get("segmentation_closing_radius", 0)),
        connectivity=int(segmentation_cfg.get("segmentation_connectivity", 4)),
    )
    prediction = written["prediction"]
    artifacts = written["artifacts"]

    ground_truth = load_ground_truth(
        event_id,
        manifest_path=manifest_path,
        data_root=root,
    )
    status = "validation_ready" if ground_truth.get("status", "").startswith("spatial") else "demo_only"
    summary = {
        **event_metadata,
        "input_sources": bundle["metadata"]["input_sources"],
        "missing_inputs": [],
        "prediction": {
            "area_km2": prediction["summary"]["area_km2"],
            "equivalent_radius_km": prediction["summary"]["equivalent_radius_km"],
            "polygon_count": prediction["summary"]["polygon_count"],
            "bounds": prediction["summary"]["bounds"],
        },
        "ground_truth": {
            "available": ground_truth.get("available", False),
            "status": ground_truth.get("status", "not_validated"),
        },
        "status": status,
        "legacy_status": LEGACY_READINESS_STATUS_MAP.get(status, status),
        "message": "Historical event reconstruction prediction only; no accuracy claim is made.",
    }
    _write_json(summary_path, summary)
    artifacts["reconstruction_summary"] = str(summary_path)
    return {"readiness": readiness, "summary": summary, "artifacts": artifacts}
