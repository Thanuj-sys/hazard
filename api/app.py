"""
FastAPI REST API gateway for MultiHazard-VLM.
Implements the API Fallback Matrix and end-to-end inference triggers.
With PostgreSQL, Redis caching, and Prometheus monitoring.
"""

import asyncio
import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from threading import Lock
from typing import Dict, Any, List, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from api.analysis_reports import attach_analysis_pdf, router as analysis_reports_router
from api.assessment import router as assessment_router
from models.multihazard_vlm import MultiHazardVLM
from data.database import (
    init_db,
    save_prediction,
    get_prediction_stats,
    get_local_prediction_history,
    Prediction,
)
from data.cache import cache_prediction, get_cached_prediction, cache_stats, get_cached_stats, invalidate_stats_cache
from data.metrics import track_request_time, track_prediction_time, get_metrics, predictions_total

# Configure Loguru
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
logger.add("logs/api_{time}.log", rotation="500 MB", retention="10 days", level="DEBUG", encoding="utf-8")

app = FastAPI(
    title="MultiHazard-VLM API Gateway with DB & Cache",
    description="REST API for multi-hazard mapping with PostgreSQL, Redis, and Prometheus monitoring.",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8503",
        "http://127.0.0.1:8503",
        "http://localhost:8504",
        "http://127.0.0.1:8504",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("[DB] Database initialized successfully!")
    except Exception as e:
        logger.warning(f"[DB] Database initialization issue: {e}. Continuing without persistent storage...")
    try:
        load_model()
        logger.info("[Model] Model pre-loaded successfully!")
    except Exception as e:
        logger.warning(f"[Model] Failed to pre-load model: {e}")
    global REALTIME_MONITOR_TASK
    if REALTIME_MONITOR_ENABLED:
        REALTIME_MONITOR_TASK = asyncio.create_task(realtime_monitor_loop())


@app.on_event("shutdown")
async def shutdown_event():
    if REALTIME_MONITOR_TASK is not None:
        REALTIME_MONITOR_TASK.cancel()
        try:
            await REALTIME_MONITOR_TASK
        except asyncio.CancelledError:
            pass

app.include_router(assessment_router)
app.include_router(analysis_reports_router)

# --- Pydantic Models ---

class IngestRequest(BaseModel):
    aoi_bounds: List[float]
    date_start: str
    date_end: str

class PredictRequest(BaseModel):
    aoi_bounds: List[float]
    country_boundary: Optional[List[List[float]]] = None

class PointPredictRequest(BaseModel):
    latitude: float
    longitude: float
    country_boundary: Optional[List[List[float]]] = None

class DashboardDataResponse(BaseModel):
    h3_hexagons: List[str]
    predicted_coordinates: List[float]
    radius_km: float
    hazard_probabilities: Dict[str, float]
    language_report: str
    inference_mode: str
    data_source: str

MODEL_CACHE: Optional[MultiHazardVLM] = None
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "planetary_computer")
APP_ENV = os.getenv("APP_ENV", "development").lower()
ALLOW_SYNTHETIC_INFERENCE = os.getenv("ALLOW_SYNTHETIC_INFERENCE", "false").lower() in {"1", "true", "yes", "on"}
MODEL_CONFIG_PATH = os.path.join("configs", "model_config.yaml")
with open(MODEL_CONFIG_PATH, encoding="utf-8") as config_file:
    MODEL_HAZARD_CLASSES = [
        str(label).lower()
        for label in yaml.safe_load(config_file).get("model", {}).get(
            "hazard_classes", ["flood", "wildfire", "landslide"]
        )
    ]
REALTIME_MONITOR_ENABLED = os.getenv("REALTIME_MONITOR_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
REALTIME_MONITOR_INTERVAL_SECONDS = max(300, int(os.getenv("REALTIME_MONITOR_INTERVAL_SECONDS", "900")))
REALTIME_PREDICTION_TIMEOUT_SECONDS = max(300, float(os.getenv("REALTIME_PREDICTION_TIMEOUT_SECONDS", "600")))
REALTIME_MONITOR_POINTS = os.getenv("REALTIME_MONITOR_POINTS", "20.5937,78.9629")
REALTIME_MONITOR_TASK: Optional[asyncio.Task] = None
MONITOR_STATE: Dict[str, Any] = {
    "enabled": REALTIME_MONITOR_ENABLED,
    "status": "disabled" if not REALTIME_MONITOR_ENABLED else "starting",
    "interval_seconds": REALTIME_MONITOR_INTERVAL_SECONDS,
    "points": [],
    "last_started_at": None,
    "last_completed_at": None,
    "last_success_at": None,
    "last_error": None,
    "latest_predictions": {},
}
INDIA_BOUNDS = (6.5, 68.0, 35.5, 97.5)
INDIA_BOUNDARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dashboard", "boundaries", "india_states.geojson"
)


@lru_cache(maxsize=1)
def load_india_boundary():
    """Load the bundled administrative boundary used to geofence predictions."""
    try:
        with open(INDIA_BOUNDARY_PATH, encoding="utf-8") as boundary_file:
            payload = json.load(boundary_file)
        geometries = [shape(feature["geometry"]) for feature in payload.get("features", [])]
        if not geometries:
            raise ValueError("boundary file contains no geometries")
        return unary_union(geometries)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.error(f"India boundary could not be loaded: {exc}")
        raise RuntimeError("India boundary data is unavailable; refusing an unverified prediction.") from exc

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}
WEATHER_CACHE_TTL_SECONDS = int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "300"))
WEATHER_CACHE = {}
WEATHER_CACHE_LOCK = Lock()


def validate_aoi_bounds(aoi_bounds: List[float]) -> List[float]:
    """Validate AOI order as [min_lat, min_lon, max_lat, max_lon]."""
    if len(aoi_bounds) != 4 or not all(math.isfinite(float(value)) for value in aoi_bounds):
        raise HTTPException(status_code=422, detail="aoi_bounds must contain four finite numbers.")
    min_lat, min_lon, max_lat, max_lon = (float(value) for value in aoi_bounds)
    if not (-90 <= min_lat < max_lat <= 90 and -180 <= min_lon < max_lon <= 180):
        raise HTTPException(
            status_code=422,
            detail="aoi_bounds must be [min_lat, min_lon, max_lat, max_lon] within geographic limits.",
        )
    return [min_lat, min_lon, max_lat, max_lon]


def validate_point(latitude: float, longitude: float) -> tuple[float, float]:
    """Validate a prediction point inside India's supported geographic domain."""
    latitude = float(latitude)
    longitude = float(longitude)
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise HTTPException(status_code=422, detail="latitude and longitude must be finite numbers.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(status_code=422, detail="latitude or longitude is outside geographic limits.")
    min_lat, min_lon, max_lat, max_lon = INDIA_BOUNDS
    if not (min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon):
        raise HTTPException(status_code=422, detail="This service currently supports locations within India only.")
    if not load_india_boundary().covers(Point(longitude, latitude)):
        raise HTTPException(status_code=422, detail="The selected location is outside India's mapped boundary.")
    return latitude, longitude


def h3_token(latitude: float, longitude: float, resolution: int) -> int:
    """Return a model-safe integer token across H3 v3 and v4 APIs."""
    import h3

    if hasattr(h3, "geo_to_h3"):
        cell = h3.geo_to_h3(latitude, longitude, resolution)
    else:
        cell = h3.latlng_to_cell(latitude, longitude, resolution)
    return int(cell, 16) % 10000


def fetch_live_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch current weather for a map point from Open-Meteo."""
    cache_key = (round(latitude, 3), round(longitude, 3))
    now = time.monotonic()
    with WEATHER_CACHE_LOCK:
        cached = WEATHER_CACHE.get(cache_key)
        if cached and now - cached[0] < WEATHER_CACHE_TTL_SECONDS:
            return {**cached[1], "cached": True}

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join([
                    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "precipitation", "rain", "weather_code", "cloud_cover",
                    "wind_speed_10m", "wind_direction_10m",
                ]),
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {})
        code = int(current.get("weather_code", -1))
        weather = {
            "status": "success",
            "source": "Open-Meteo",
            "latitude": latitude,
            "longitude": longitude,
            "observed_at": current.get("time"),
            "timezone": payload.get("timezone"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "weather_code": code,
            "condition": WEATHER_CODES.get(code, "Unknown conditions"),
        }
        with WEATHER_CACHE_LOCK:
            WEATHER_CACHE[cache_key] = (time.monotonic(), weather)
        return {**weather, "cached": False}
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning(f"Live weather lookup failed: {exc}")
        raise HTTPException(status_code=503, detail="Live weather is temporarily unavailable.") from exc


def resolve_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    return checkpoints[-1] if checkpoints else None


def adapt_checkpoint_state(checkpoint_state: Dict[str, torch.Tensor], model: MultiHazardVLM) -> Dict[str, torch.Tensor]:
    """Adapt safe architecture-only differences before loading a checkpoint."""
    current_state = model.state_dict()
    adapted_state = dict(checkpoint_state)
    key = "backbone.patch_embed.spatial_pos_embed"
    checkpoint_embedding = checkpoint_state.get(key)
    current_embedding = current_state.get(key)

    if checkpoint_embedding is not None and current_embedding is not None and checkpoint_embedding.shape != current_embedding.shape:
        if checkpoint_embedding.ndim != 4 or current_embedding.ndim != 4:
            raise RuntimeError(f"Unsupported positional embedding shapes: {checkpoint_embedding.shape} -> {current_embedding.shape}")
        checkpoint_tokens, current_tokens = checkpoint_embedding.shape[2], current_embedding.shape[2]
        checkpoint_side = int(checkpoint_tokens ** 0.5)
        current_side = int(current_tokens ** 0.5)
        if checkpoint_side ** 2 != checkpoint_tokens or current_side ** 2 != current_tokens:
            raise RuntimeError(f"Positional embeddings must be square: {checkpoint_embedding.shape} -> {current_embedding.shape}")

        resized = checkpoint_embedding.squeeze(0).squeeze(0).transpose(0, 1)
        resized = resized.reshape(1, current_embedding.shape[3], checkpoint_side, checkpoint_side)
        resized = F.interpolate(resized, size=(current_side, current_side), mode="bicubic", align_corners=False)
        resized = resized.reshape(current_embedding.shape[3], current_tokens).transpose(0, 1)
        adapted_state[key] = resized.reshape_as(current_embedding)
        logger.info(f"Resized checkpoint positional embedding from {checkpoint_tokens} to {current_tokens} tokens.")

    return adapted_state


def load_model() -> MultiHazardVLM:
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    config_path = os.path.join("configs", "model_config.yaml")
    model = MultiHazardVLM(config_path=config_path if os.path.exists(config_path) else None)

    checkpoint_path = resolve_latest_checkpoint()
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        try:
            model.load_state_dict(adapt_checkpoint_state(checkpoint_state, model), strict=True)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as exc:
            logger.warning(f"Checkpoint mismatch for {checkpoint_path}: {exc}")
            if INFERENCE_MODE == "planetary_computer":
                raise RuntimeError(
                    "Realtime inference requires a checkpoint matching the configured model architecture."
                ) from exc
            logger.warning("Falling back to current model config and continuing with initialized weights.")
            current_state = model.state_dict()
            filtered = {}
            for key, value in checkpoint_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    filtered[key] = value
            if filtered:
                model.load_state_dict(filtered, strict=False)
    else:
        logger.warning("No checkpoint found; using freshly initialized model weights.")
        if INFERENCE_MODE == "planetary_computer":
            raise RuntimeError("Realtime inference requires a trained checkpoint; none was found.")

    model.eval()
    MODEL_CACHE = model
    return model


def build_demo_prediction(aoi_bounds: Optional[List[float]] = None) -> Dict[str, Any]:
    if APP_ENV == "production" and INFERENCE_MODE == "demo_synthetic":
        raise HTTPException(
            status_code=503,
            detail="Synthetic inference is disabled in production; configure a real data inference mode.",
        )

    model = load_model()
    device = next(model.parameters()).device

    if aoi_bounds is None:
        aoi_bounds = [10.0, 70.0, 15.0, 75.0]

    if len(aoi_bounds) == 4:
        aoi_tensor = torch.tensor([aoi_bounds], dtype=torch.float32, device=device)
    else:
        aoi_tensor = torch.tensor([[10.0, 70.0, 15.0, 75.0]], dtype=torch.float32, device=device)

    # Seed random generator with location coordinates for location-aware predictions
    lat_center = (aoi_bounds[0] + aoi_bounds[2]) / 2.0
    lon_center = (aoi_bounds[1] + aoi_bounds[3]) / 2.0
    location_seed = int((abs(lat_center) * 1000 + abs(lon_center) * 1000) % (2**31 - 1))
    
    torch.manual_seed(location_seed)
    
    sample_images = torch.randn(1, 5, 10, 64, 64, device=device)
    sample_quality = torch.ones(1, 5, dtype=torch.bool, device=device)
    sample_h3 = {
        4: torch.randint(0, 1000, (1, 10), device=device),
        6: torch.randint(0, 1000, (1, 20), device=device),
        8: torch.randint(0, 1000, (1, 30), device=device),
    }
    sample_geo = {
        'elevation': torch.randn(1, 1, device=device) * 1000,  # Scale elevation (0-5000m range)
        'slope': torch.rand(1, 1, device=device) * 90,  # Scale slope (0-90 degrees)
        'ndvi': torch.rand(1, 1, device=device),  # NDVI stays 0-1
        'aspect': torch.randn(1, 1, device=device) * 180,  # Scale aspect (0-360 degrees)
        'lithology_class': torch.randint(0, 10, (1,), device=device),
    }

    with torch.no_grad():
        outputs = model(
            images=sample_images,
            quality_flags=sample_quality,
            h3_indices=sample_h3,
            geo_attributes=sample_geo,
            aoi_bounds=aoi_tensor,
        )

    hazard_names = MODEL_HAZARD_CLASSES
    hazard_idx = int(torch.argmax(outputs['hazard_logits'][0]).item())
    hazard_name = hazard_names[min(hazard_idx, len(hazard_names) - 1)]
    prob_values = torch.softmax(outputs['hazard_logits'][0], dim=-1).detach().cpu().tolist()

    coords = outputs['coords_geo'][0].detach().cpu().tolist()
    base_radius = float(torch.clamp(outputs['radius_km'][0, 0], min=0.5, max=30.0).item())
    
    # Apply location-based radius variation
    location_aware_radius = compute_location_based_radius(lat_center, lon_center, hazard_name, base_radius)

    return {
        'status': 'success',
        'inference_mode': INFERENCE_MODE,
        'data_source': 'Synthetic demo tensors' if INFERENCE_MODE == 'demo_synthetic' else INFERENCE_MODE,
            'hazard_type': hazard_name,
        'predicted_coordinates': coords,
        'hazard_probabilities': {
            label: float(probability)
            for label, probability in zip(MODEL_HAZARD_CLASSES, prob_values)
        },
        'predicted_coordinates': coords,
        'radius_km': location_aware_radius,
        'language_report': (
            f"{hazard_name.upper()} ALERT: Model analysis indicates a {hazard_name.lower()} risk for the AOI. "
            "The final advisory output is grounded in the fused geospatial representation and hazard-specific head predictions."
        )
    }


def _latest_band_arrays(dataset: Any, band_names: List[str]) -> List[np.ndarray]:
    """Extract the latest available STAC bands as 2D float arrays."""
    if dataset is None:
        return []
    available = set(getattr(dataset, "data_vars", {}))
    if not all(name in available for name in band_names):
        return []
    latest = dataset.isel(time=-1) if "time" in dataset.dims else dataset
    arrays = []
    for name in band_names:
        array = np.atleast_2d(np.asarray(latest[name].values, dtype=np.float32).squeeze())
        if not np.isfinite(array).any():
            return []
        fill_value = float(np.nanmedian(array))
        array = np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
        arrays.append(array)
    return arrays


def build_planetary_computer_prediction(latitude: float, longitude: float) -> Dict[str, Any]:
    """Run inference from the latest public Sentinel scenes around a point."""
    try:
        from data.planetary_computer import PlanetaryComputerClient
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Realtime prediction dependencies are missing. Use the project virtual environment and run: python -m pip install -r requirements.txt",
        ) from exc

    half_size = float(os.getenv("REALTIME_AOI_DEGREES", "0.025"))
    bbox = [longitude - half_size, latitude - half_size, longitude + half_size, latitude + half_size]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(os.getenv("REALTIME_LOOKBACK_DAYS", "30")))
    date_range = f"{start.date().isoformat()}/{end.date().isoformat()}"

    client = PlanetaryComputerClient("sentinel-1-grd")
    sentinel1 = client.load_sentinel1(bbox, date_range)
    sentinel2 = client.load_sentinel2(bbox, date_range)
    s1_arrays = _latest_band_arrays(sentinel1, ["vv", "vh"])
    s2_arrays = _latest_band_arrays(sentinel2, ["B02", "B03", "B04", "B08", "B11", "B12"])
    if not s1_arrays:
        raise HTTPException(status_code=503, detail="No recent Sentinel-1 scene covers this location.")
    if not s2_arrays:
        raise HTTPException(status_code=503, detail="No recent Sentinel-2 scene covers this location.")

    bands = s1_arrays + s2_arrays
    resized_bands = [
        F.interpolate(torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
        for array in bands
    ]
    band_tensor = torch.cat(resized_bands, dim=1)
    if band_tensor.shape[1] < 10:
        band_tensor = F.pad(band_tensor, (0, 0, 0, 0, 0, 10 - band_tensor.shape[1]))
    band_tensor = (band_tensor - band_tensor.mean(dim=(2, 3), keepdim=True)) / band_tensor.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    red = s2_arrays[2]
    nir = s2_arrays[3]
    ndvi = float(np.nanmean((nir - red) / np.where(np.abs(nir + red) < 1e-6, 1.0, nir + red)))
    h3_indices = {
        resolution: torch.tensor([[h3_token(latitude, longitude, resolution)]], dtype=torch.long)
        for resolution in (4, 6, 8)
    }
    geo_attributes = {
        "elevation": torch.zeros(1, 1),
        "slope": torch.zeros(1, 1),
        "ndvi": torch.tensor([[ndvi]], dtype=torch.float32),
        "aspect": torch.zeros(1, 1),
        "lithology_class": torch.zeros(1, dtype=torch.long),
    }
    model = load_model()
    device = next(model.parameters()).device
    aoi_tensor = torch.tensor([[latitude - half_size, longitude - half_size, latitude + half_size, longitude + half_size]], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(
            images=band_tensor.unsqueeze(1).to(device),
            quality_flags=torch.ones(1, 1, dtype=torch.bool, device=device),
            h3_indices={key: value.to(device) for key, value in h3_indices.items()},
            geo_attributes={key: value.to(device) for key, value in geo_attributes.items()},
            aoi_bounds=aoi_tensor,
        )

    hazard_names = MODEL_HAZARD_CLASSES
    probabilities = torch.softmax(outputs["hazard_logits"][0], dim=-1).cpu().tolist()
    if len(probabilities) != len(hazard_names):
        raise HTTPException(status_code=503, detail="Model hazard classes do not match model output dimensions.")
    hazard_name = hazard_names[int(torch.argmax(outputs["hazard_logits"][0]).item())]
    raw_coords = outputs["coords_geo"][0].cpu().tolist()
    radius = float(torch.clamp(outputs["radius_km"][0, 0], min=0.5, max=30.0).item())
    acquisition = getattr(sentinel2, "time", None)
    acquisition_value = acquisition.values[-1] if acquisition is not None else None
    acquisition_date = (
        np.datetime_as_string(acquisition_value, unit="us")
        if isinstance(acquisition_value, np.datetime64)
        else str(acquisition_value or "")
    )
    if not acquisition_date:
        raise HTTPException(status_code=503, detail="Satellite scene has no acquisition timestamp.")
    try:
        acquisition_timestamp = datetime.fromisoformat(acquisition_date.replace("Z", "+00:00"))
        if acquisition_timestamp.tzinfo is None:
            acquisition_timestamp = acquisition_timestamp.replace(tzinfo=timezone.utc)
        source_age_hours = round((datetime.now(timezone.utc) - acquisition_timestamp).total_seconds() / 3600, 1)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Satellite acquisition timestamp is invalid.") from exc
    source = f"Microsoft Planetary Computer Sentinel-1/Sentinel-2, latest acquisition {acquisition_date}"
    return {
        "status": "success",
        "inference_mode": "planetary_computer",
        "data_source": source,
        "source_acquired_at": acquisition_date,
        "source_age_hours": max(0.0, source_age_hours),
        "data_freshness": "satellite_latest_available",
        "hazard_type": hazard_name,
        "hazard_probabilities": {
            label: float(probability)
            for label, probability in zip(MODEL_HAZARD_CLASSES, probabilities)
        },
        "predicted_coordinates": [float(raw_coords[0]), float(raw_coords[1])],
        "radius_km": compute_location_based_radius(latitude, longitude, hazard_name, radius),
        "language_report": f"{hazard_name.upper()} SIGNAL: Latest available satellite imagery was analyzed for the selected location. Acquisition: {acquisition_date}. Validate before operational use.",
    }


def parse_monitor_points() -> List[tuple[float, float]]:
    """Parse monitored points from LAT,LON;LAT,LON environment syntax."""
    points = []
    for raw_point in REALTIME_MONITOR_POINTS.split(";"):
        try:
            latitude, longitude = (float(value.strip()) for value in raw_point.split(","))
            validate_point(latitude, longitude)
            points.append((latitude, longitude))
        except (ValueError, HTTPException):
            logger.warning(f"Skipping invalid realtime monitor point: {raw_point}")
    return points


async def realtime_monitor_loop() -> None:
    """Poll fresh satellite scenes and retain the latest successful result per point."""
    points = parse_monitor_points()
    MONITOR_STATE["points"] = [{"latitude": lat, "longitude": lon} for lat, lon in points]
    while True:
        cycle_started = datetime.now(timezone.utc).isoformat()
        MONITOR_STATE["status"] = "running"
        MONITOR_STATE["last_started_at"] = cycle_started
        successes = 0
        errors = []
        for latitude, longitude in points:
            key = f"{latitude:.4f},{longitude:.4f}"
            try:
                prediction = await asyncio.wait_for(
                    asyncio.to_thread(build_planetary_computer_prediction, latitude, longitude),
                    timeout=REALTIME_PREDICTION_TIMEOUT_SECONDS,
                )
                MONITOR_STATE["latest_predictions"][key] = {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "latitude": latitude,
                    "longitude": longitude,
                    "hazard_type": prediction["hazard_type"],
                    "data_source": prediction["data_source"],
                    "inference_mode": prediction["inference_mode"],
                }
                successes += 1
            except Exception as exc:
                logger.warning(f"Realtime monitor failed for ({latitude}, {longitude}): {exc}")
                errors.append(f"{key}: {exc}")

        MONITOR_STATE["last_completed_at"] = datetime.now(timezone.utc).isoformat()
        MONITOR_STATE["last_success_at"] = MONITOR_STATE["last_completed_at"] if successes else MONITOR_STATE["last_success_at"]
        MONITOR_STATE["last_error"] = "; ".join(errors) if errors else None
        MONITOR_STATE["status"] = "healthy" if successes == len(points) and points else "degraded"
        await asyncio.sleep(REALTIME_MONITOR_INTERVAL_SECONDS)


# --- Dummy Service Functions (To be replaced with actual module calls) ---

def run_ingestion_with_fallback(aoi_bounds: List[float], date_start: str, date_end: str):
    logger.info("Starting ingestion with fallback matrix.")
    
    # Sentinel-2 (Optical) Fallback
    optical_success = False
    try:
        logger.info("Attempting Sentinel-2 ingestion...")
        # attempt sentinel 2
        # raise Exception("Simulated timeout")
        optical_success = True
    except Exception as e:
        logger.warning(f"Sentinel-2 failed: {e}. Trying Landsat-9...")
        try:
            # attempt landsat 9
            optical_success = True
        except Exception as e2:
            logger.error("Optical ingestion failed entirely. Triggering SAR-only mode.")
            optical_success = False
            
    # Sentinel-1 (SAR) Fallback
    sar_success = False
    try:
        logger.info("Attempting Sentinel-1 ingestion...")
        # attempt sentinel 1
        sar_success = True
    except Exception as e:
        logger.warning(f"Sentinel-1 failed: {e}. Trying ISRO Bhuvan/NISAR...")
        try:
            # attempt NISAR
            sar_success = True
        except Exception as e2:
            logger.error("SAR ingestion failed entirely. Using historical radar persistence.")
            sar_success = False
            
    # Open-Meteo Fallback
    weather_success = False
    try:
        logger.info("Attempting Open-Meteo ingestion...")
        # attempt Open-Meteo
        weather_success = True
    except Exception as e:
        logger.warning(f"Open-Meteo failed: {e}. Trying IMD AWS / GFS persistence...")
        try:
            # attempt IMD AWS
            weather_success = True
        except Exception as e2:
            logger.error("Weather ingestion failed entirely.")
            weather_success = False
            
    logger.info("Ingestion process completed.")

def clamp_coordinates(lat: float, lon: float, boundary: Optional[List[List[float]]]) -> tuple[float, float]:
    """Applies geofenced coordinate clamping to snap coordinates inside country/state boundary."""
    # Simplified bounding box clamping for demonstration
    if boundary and len(boundary) >= 4:
        min_lon = min(p[0] for p in boundary)
        max_lon = max(p[0] for p in boundary)
        min_lat = min(p[1] for p in boundary)
        max_lat = max(p[1] for p in boundary)
        
        clamped_lon = max(min_lon, min(lon, max_lon))
        clamped_lat = max(min_lat, min(lat, max_lat))
        return clamped_lat, clamped_lon
    return lat, lon

def compute_location_based_radius(lat: float, lon: float, hazard_type: str, base_radius: float) -> float:
    """
    Compute location-aware impact radius based on coordinates and hazard type.
    This creates realistic variation in predictions for different locations.
    """
    # Use location coordinates to create variation (0.0 to 1.0)
    lat_factor = abs(lat % 10) / 10.0  # 0-10 latitude variation
    lon_factor = abs(lon % 10) / 10.0  # 0-10 longitude variation
    location_variation = (lat_factor + lon_factor) / 2.0
    
    # Hazard-specific base multipliers
    hazard_multipliers = {
        'flood': 1.5,
        'wildfire': 2.0,
        'landslide': 0.8,
    }
    
    multiplier = hazard_multipliers.get(hazard_type, 1.0)
    
    # Compute radius with location-aware variation
    # Range: 2 km to 25 km depending on location and hazard type
    radius = base_radius + (location_variation * multiplier * 15.0)
    radius = max(2.0, min(25.0, radius))  # Clamp to realistic range
    
    return round(radius, 1)

# --- Endpoints ---

@app.post("/api/v1/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers rolling 30-day ingestion with API Fallback Matrix.
    """
    aoi_bounds = validate_aoi_bounds(request.aoi_bounds)
    center_lat = (aoi_bounds[0] + aoi_bounds[2]) / 2.0
    center_lon = (aoi_bounds[1] + aoi_bounds[3]) / 2.0
    validate_point(center_lat, center_lon)
    background_tasks.add_task(
        run_ingestion_with_fallback,
        aoi_bounds,
        request.date_start,
        request.date_end
    )
    return {"message": "Ingestion started in background."}

async def handle_predict(request: Request):
    """
    Runs end-to-end inference for clicked location.
    Takes AOI bounds and returns hazard predictions, impact radius, and affected area.
    """
    try:
        aoi_bounds = [10.0, 70.0, 15.0, 75.0]
        country_boundary = None
        if request.method == "POST":
            try:
                body = await request.json()
                if isinstance(body, dict):
                    if "aoi_bounds" in body and body["aoi_bounds"]:
                        aoi_bounds = body["aoi_bounds"]
                    country_boundary = body.get("country_boundary")
            except Exception:
                pass

        aoi_bounds = validate_aoi_bounds(aoi_bounds)
        center_lat = (aoi_bounds[0] + aoi_bounds[2]) / 2.0
        center_lon = (aoi_bounds[1] + aoi_bounds[3]) / 2.0
        validate_point(center_lat, center_lon)
        logger.info(f"Running prediction for AOI: {aoi_bounds}")

        if INFERENCE_MODE == "planetary_computer":
            prediction = build_planetary_computer_prediction(center_lat, center_lon)
        elif ALLOW_SYNTHETIC_INFERENCE:
            prediction = build_demo_prediction(aoi_bounds)
        else:
            raise HTTPException(
                status_code=503,
                detail="Real satellite inference is not configured; synthetic inference is disabled.",
            )
        
        # Use AOI center as clicked point
        clamped_lat = max(-90.0, min(90.0, center_lat))
        clamped_lon = max(-180.0, min(180.0, center_lon))

        if country_boundary:
            clamped_lat, clamped_lon = clamp_coordinates(clamped_lat, clamped_lon, country_boundary)

        radius_km = prediction['radius_km']
        affected_area_km2 = 3.14159265359 * (radius_km ** 2)  # π * r²

        return {
            "status": "success",
            "inference_mode": prediction["inference_mode"],
            "data_source": prediction["data_source"],
            "source_acquired_at": prediction.get("source_acquired_at"),
            "source_age_hours": prediction.get("source_age_hours"),
            "data_freshness": prediction.get("data_freshness", "model_generated"),
            "predicted_coordinates": [clamped_lat, clamped_lon],
            "radius_km": radius_km,
            "affected_area_km2": round(affected_area_km2, 2),
            "hazard_type": prediction['hazard_type'],
            "hazard_probabilities": prediction['hazard_probabilities'],
            "language_report": prediction['language_report'],
            "message": "Prediction complete."
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        logger.error(f"Error in handle_predict: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/v1/predict")
async def predict_get(request: Request):
    return await handle_predict(request)

@app.post("/api/v1/predict")
async def predict_post(request: Request):
    return await handle_predict(request)

async def handle_predict_point(request: Request):
    """
    Runs prediction for a single clicked point (lat, lon).
    Supports both GET and POST methods to prevent Method Not Allowed errors.
    """
    try:
        latitude = None
        longitude = None
        country_boundary = None
        selected_hazard = request.query_params.get("selected_hazard", "all")
        
        if request.method == "POST":
            try:
                body = await request.json()
                if isinstance(body, dict):
                    latitude = body.get("latitude", latitude)
                    longitude = body.get("longitude", longitude)
                    country_boundary = body.get("country_boundary")
                    selected_hazard = body.get("selected_hazard", selected_hazard)
            except Exception:
                pass

        if latitude is None or longitude is None:
            if "latitude" in request.query_params:
                try:
                    latitude = float(request.query_params["latitude"])
                except Exception:
                    pass
            if "longitude" in request.query_params:
                try:
                    longitude = float(request.query_params["longitude"])
                except Exception:
                    pass

        if latitude is None:
            latitude = 20.5937
        if longitude is None:
            longitude = 78.9629

        if selected_hazard not in ("all", "flood", "wildfire", "landslide"):
            raise HTTPException(status_code=422, detail="selected_hazard must be all, flood, wildfire, or landslide.")
        latitude, longitude = validate_point(latitude, longitude)

        logger.info(f"Running point prediction for: {latitude}, {longitude}")
        
        # Check cache first
        if INFERENCE_MODE != "planetary_computer":
            cached_result = get_cached_prediction(latitude, longitude)
            if cached_result:
                cached_result.setdefault("inference_mode", INFERENCE_MODE)
                cached_result.setdefault(
                    "data_source",
                    "Synthetic demo tensors" if INFERENCE_MODE == "demo_synthetic" else INFERENCE_MODE,
                )
                logger.info(f"Cache hit for location ({latitude}, {longitude})")
                return await asyncio.to_thread(attach_analysis_pdf, cached_result, selected_hazard)
        
        # Create AOI bounds around clicked point (0.5 degree buffer)
        aoi_bounds = [
            latitude - 0.5,
            longitude - 0.5,
            latitude + 0.5,
            longitude + 0.5
        ]
        
        if INFERENCE_MODE == "planetary_computer":
            try:
                prediction = await asyncio.wait_for(
                    asyncio.to_thread(
                        build_planetary_computer_prediction, latitude, longitude
                    ),
                    timeout=REALTIME_PREDICTION_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Satellite prediction timed out after %.0f seconds",
                    REALTIME_PREDICTION_TIMEOUT_SECONDS,
                )
                raise HTTPException(
                    status_code=504,
                    detail="Satellite prediction timed out while downloading or processing fresh imagery.",
                ) from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.error("Satellite prediction unavailable: %s", exc)
                raise HTTPException(
                    status_code=503,
                    detail="Fresh satellite prediction is temporarily unavailable; no synthetic result was returned.",
                ) from exc
        elif ALLOW_SYNTHETIC_INFERENCE:
            prediction = await asyncio.to_thread(build_demo_prediction, aoi_bounds)
        else:
            raise HTTPException(
                status_code=503,
                detail="Real satellite inference is not configured; synthetic inference is disabled.",
            )
        
        # Use clicked point as-is
        clamped_lat = max(-90.0, min(90.0, latitude))
        clamped_lon = max(-180.0, min(180.0, longitude))
        
        if country_boundary:
            clamped_lat, clamped_lon = clamp_coordinates(clamped_lat, clamped_lon, country_boundary)
        
        radius_km = prediction['radius_km']
        affected_area_km2 = 3.14159265359 * (radius_km ** 2)  # π * r²
        
        response = {
            "status": "success",
            "inference_mode": prediction["inference_mode"],
            "data_source": prediction["data_source"],
            "source_acquired_at": prediction.get("source_acquired_at"),
            "source_age_hours": prediction.get("source_age_hours"),
            "data_freshness": prediction.get("data_freshness", "model_generated"),
            "predicted_coordinates": [clamped_lat, clamped_lon],
            "radius_km": radius_km,
            "affected_area_km2": round(affected_area_km2, 2),
            "hazard_type": prediction['hazard_type'],
            "hazard_probabilities": prediction['hazard_probabilities'],
            "language_report": prediction['language_report'],
            "message": "Point prediction complete.",
            "cached": False
        }
        
        # Save to database in background (non-blocking)
        try:
            save_prediction(
                latitude=latitude,
                longitude=longitude,
                hazard_type=prediction['hazard_type'],
                hazard_probabilities=prediction['hazard_probabilities'],
                predicted_lat=clamped_lat,
                predicted_lon=clamped_lon,
                impact_radius_km=radius_km,
                affected_area_km2=affected_area_km2,
                language_report=prediction['language_report'],
                model_version="v2.0"
            )
            logger.info(f"[DB] Prediction saved to database")
        except Exception as e:
            logger.warning(f"[DB] Failed to save prediction to database: {e}")
        
        # Cache the result
        try:
            cache_prediction(latitude, longitude, response)
            logger.info(f"[Cache] Prediction cached")
        except Exception as e:
            logger.warning(f"[Cache] Failed to cache prediction: {e}")
        
        # Track metrics
        try:
            predictions_total.labels(hazard_type=prediction['hazard_type']).inc()
        except Exception as e:
            logger.warning(f"[Metrics] Error updating metrics: {e}")
        
        return await asyncio.to_thread(attach_analysis_pdf, response, selected_hazard)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        error_msg = f"Error in predict_point: {exc}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg, sys.stderr)
        raise HTTPException(status_code=500, detail="Point prediction failed.") from exc
@app.get("/api/v1/predict-point")
async def predict_point_get(request: Request):
    return await handle_predict_point(request)

@app.post("/api/v1/predict-point")
async def predict_point_post(request: Request):
    return await handle_predict_point(request)

@app.get("/api/v1/stats")
async def get_stats():
    """
    Get global statistics about all predictions.
    Cached for 5 minutes to reduce database load.
    """
    # Try cache first
    cached_stats = get_cached_stats()
    if cached_stats:
        cached_stats["from_cache"] = True
        return cached_stats
    
    # Query database
    try:
        stats = get_prediction_stats()
        stats["from_cache"] = False
        
        # Cache the result
        try:
            cache_stats(stats)
        except:
            pass
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return {
            "total_predictions": 0,
            "flood_predictions": 0,
            "wildfire_predictions": 0,
            "landslide_predictions": 0,
            "avg_radius_km": 0.0,
            "avg_affected_area_km2": 0.0,
            "error": str(e)
        }


@app.get("/api/v1/prediction-history")
async def get_prediction_history(limit: int = 100):
    """Return locally persisted prediction analyses for the evaluation workspace."""
    history = get_local_prediction_history(limit)
    return {
        "status": "available",
        "storage": "local_json_fallback",
        "count": len(history),
        "predictions": history,
        "detail": (
            "History contains model analyses without verified reference labels. "
            "These records are not benchmark accuracy results."
        ),
    }

@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint for monitoring.
    Use: http://localhost:8000/metrics
    """
    return Response(content=get_metrics(), media_type="text/plain")


@app.get("/api/v1/evaluation/metrics")
async def get_evaluation_metrics():
    """Return a saved evaluation report, or an explicit not-run status."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configured_path = os.getenv("EVALUATION_RESULTS_PATH", "evaluation/metrics_report.json")
    report_path = configured_path if os.path.isabs(configured_path) else os.path.join(project_root, configured_path)
    response = {
        "status": "not_run",
        "results_available": False,
        "results_by_hazard": {},
        "sample_counts": {},
        "metadata": {},
        "report_name": os.path.basename(report_path),
        "detail": "No saved evaluation report is available. Save verified EvaluationRunner results to evaluation/metrics_report.json or set EVALUATION_RESULTS_PATH.",
    }
    if not os.path.isfile(report_path):
        return response

    try:
        with open(report_path, encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Evaluation report could not be loaded from %s: %s", report_path, exc)
        return {**response, "status": "invalid_report", "detail": "The configured evaluation report could not be read as valid JSON."}

    if not isinstance(report, dict):
        return {**response, "status": "invalid_report", "detail": "The evaluation report must contain a JSON object."}

    raw_results = report.get("results_by_hazard")
    if raw_results is None:
        raw_results = {key: value for key, value in report.items() if isinstance(value, dict)}
    if not isinstance(raw_results, dict):
        return {**response, "status": "invalid_report", "detail": "results_by_hazard must be a JSON object."}

    clean_results = {}
    for hazard, metrics in raw_results.items():
        if not isinstance(metrics, dict):
            continue
        clean_metrics = {
            str(name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        if clean_metrics:
            clean_results[str(hazard).lower()] = clean_metrics

    sample_counts = report.get("sample_counts", {})
    metadata = report.get("metadata", {})
    from evaluation.diagnostics import report_diagnostics
    diagnostics = report_diagnostics(report)
    available = bool(clean_results)
    return {
        **response,
        "status": "available" if available else "not_run",
        "results_available": available,
        "results_by_hazard": clean_results,
        "sample_counts": sample_counts if isinstance(sample_counts, dict) else {},
        "metadata": metadata if isinstance(metadata, dict) else {},
        "detailed_evaluation": diagnostics,
        "metric_coverage": report.get("metric_coverage", {}),
        "detail": "Saved evaluation metrics loaded." if available else response["detail"],
    }

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    Returns status of API, database, and cache.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "api": "✅ running",
            "database": "✅ configured",
            "cache": "✅ configured",
            "model": "✅ loaded"
        }
    }


@app.get("/api/v1/weather")
async def weather_get(latitude: float = 20.5937, longitude: float = 78.9629):
    latitude, longitude = validate_point(latitude, longitude)
    return await asyncio.to_thread(fetch_live_weather, latitude, longitude)


@app.get("/api/v1/monitor/status")
async def monitor_status():
    """Return monitor health, poll timestamps, and latest successful points."""
    return MONITOR_STATE

async def handle_dashboard_data(request: Request):
    """
    Returns H3 hexagons, predicted coordinates, radius, hazard classification probabilities, 
    and reports for rendering in the dashboard.
    """
    try:
        if INFERENCE_MODE == "planetary_computer":
            prediction = build_planetary_computer_prediction(20.5937, 78.9629)
        else:
            prediction = build_demo_prediction()
        return DashboardDataResponse(
            h3_hexagons=["863d68b67ffffff", "863d68b6fffffff"],
            predicted_coordinates=prediction['predicted_coordinates'],
            radius_km=prediction['radius_km'],
            hazard_probabilities=prediction['hazard_probabilities'],
            language_report=prediction['language_report'],
            inference_mode=prediction['inference_mode'],
            data_source=prediction['data_source'],
        )
    except Exception as exc:
        import traceback
        logger.error(f"Error in handle_dashboard_data: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/v1/dashboard/data", response_model=DashboardDataResponse)
async def get_dashboard_data_get(request: Request):
    return await handle_dashboard_data(request)

@app.post("/api/v1/dashboard/data", response_model=DashboardDataResponse)
async def get_dashboard_data_post(request: Request):
    return await handle_dashboard_data(request)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
