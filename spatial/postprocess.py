"""Convert segmentation logits into conservative georeferenced footprints."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from pyproj import CRS, Transformer
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.validation import make_valid

SUPPORTED_HAZARDS = {"flood", "landslide", "wildfire"}
DEFAULT_THRESHOLD = 0.5
DEFAULT_NODATA = 255
DEFAULT_AREA_CRS = "EPSG:6933"


def _as_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _single_band(array: Any, name: str) -> np.ndarray:
    values = _as_numpy(array)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if values.shape[0] == 1:
            return values[0]
        if values.shape[-1] == 1:
            return values[..., 0]
    if values.ndim == 4 and values.shape[0] == 1 and values.shape[1] == 1:
        return values[0, 0]
    raise ValueError(f"{name} must be a single spatial band or a single-item single-band tensor.")


def _affine(transform: Any) -> Affine:
    if isinstance(transform, Affine):
        return transform
    return Affine(*tuple(transform)[:6])


def _valid_pixels(
    values: np.ndarray,
    validity_mask: Optional[Any] = None,
    nodata: Optional[float] = None,
) -> np.ndarray:
    valid = np.isfinite(values)
    if nodata is not None:
        if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
            valid &= ~np.isnan(values)
        else:
            valid &= values != nodata
    if validity_mask is not None:
        external = _single_band(validity_mask, "Validity mask").astype(bool)
        if external.shape != values.shape:
            raise ValueError("Validity mask dimensions do not match the prediction raster.")
        valid &= external
    return valid


def logits_to_probability(logits: Any) -> Any:
    """Apply sigmoid to raw segmentation logits, preserving tensor inputs."""
    if isinstance(logits, torch.Tensor):
        return torch.sigmoid(logits)
    values = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))).astype(np.float32)


def probability_to_mask(
    probability: Any,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    validity_mask: Optional[Any] = None,
    nodata: Optional[float] = None,
    nodata_value: int = DEFAULT_NODATA,
) -> np.ndarray:
    """Threshold one probability map into 0/1 labels, retaining invalid pixels as nodata."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Probability threshold must be between 0 and 1.")
    values = _single_band(probability, "Probability").astype(np.float32, copy=False)
    valid = _valid_pixels(values, validity_mask, nodata)
    if np.any(valid & ((values < 0.0) | (values > 1.0))):
        raise ValueError("Valid probability values must be within [0, 1].")
    output = np.full(values.shape, nodata_value if not valid.all() else 0, dtype=np.uint8)
    output[valid] = (values[valid] >= threshold).astype(np.uint8)
    return output


def _validate_binary_mask(
    mask: Any,
    validity_mask: Optional[Any],
    nodata: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    values = _single_band(mask, "Binary mask")
    valid = _valid_pixels(values, validity_mask, nodata)
    unexpected = np.unique(values[valid][(values[valid] != 0) & (values[valid] != 1)])
    if unexpected.size:
        raise ValueError(f"Mask contains non-binary valid values: {unexpected.tolist()}.")
    return values, valid


def _component_structure(connectivity: int) -> np.ndarray:
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8.")
    return ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)


def clean_binary_mask(
    mask: Any,
    *,
    validity_mask: Optional[Any] = None,
    nodata: Optional[float] = DEFAULT_NODATA,
    nodata_value: int = DEFAULT_NODATA,
    minimum_component_pixels: int = 0,
    minimum_component_area_m2: float = 0.0,
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    opening_radius: int = 0,
    closing_radius: int = 0,
    connectivity: int = 4,
) -> np.ndarray:
    """Apply optional morphology and small-component removal to a binary raster."""
    values, valid = _validate_binary_mask(mask, validity_mask, nodata)
    if minimum_component_pixels < 0 or minimum_component_area_m2 < 0:
        raise ValueError("Minimum component sizes cannot be negative.")
    if opening_radius < 0 or closing_radius < 0:
        raise ValueError("Morphology radii cannot be negative.")
    if minimum_component_area_m2 > 0 and (transform is None or crs is None):
        raise ValueError("Area-based component filtering requires an affine transform and CRS.")

    structure = _component_structure(connectivity)
    cleaned = (values == 1) & valid
    if opening_radius:
        cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=opening_radius)
    if closing_radius:
        cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=closing_radius)
    cleaned &= valid

    if minimum_component_pixels or minimum_component_area_m2:
        labels, component_count = ndimage.label(cleaned, structure=structure)
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            if minimum_component_pixels and int(component.sum()) < minimum_component_pixels:
                cleaned[component] = False
                continue
            if minimum_component_area_m2:
                component_geometry = mask_to_polygon(
                    component.astype(np.uint8),
                    transform,
                    crs,
                    connectivity=connectivity,
                )
                area = calculate_area(component_geometry, crs)["area_m2"]
                if area < minimum_component_area_m2:
                    cleaned[component] = False

    output = np.full(values.shape, nodata_value if not valid.all() else 0, dtype=np.uint8)
    output[valid] = cleaned[valid].astype(np.uint8)
    return output


def calculate_area(
    geometry: Any,
    crs: Any,
    area_crs: Any = DEFAULT_AREA_CRS,
) -> Dict[str, float]:
    """Measure polygon area in an equal-area CRS and return square metres/kilometres."""
    source_crs = CRS.from_user_input(crs)
    target_crs = CRS.from_user_input(area_crs)
    if geometry is None or geometry.is_empty:
        return {"area_m2": 0.0, "area_km2": 0.0}
    if not geometry.is_valid:
        raise ValueError("Cannot measure an invalid footprint geometry.")
    if source_crs == target_crs:
        projected = geometry
    else:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        projected = transform_geometry(transformer.transform, geometry)
    area_m2 = float(projected.area)
    if not math.isfinite(area_m2) or area_m2 < 0:
        raise ValueError("Projected footprint area is not finite and non-negative.")
    return {"area_m2": area_m2, "area_km2": area_m2 / 1_000_000.0}


def calculate_equivalent_radius(area_km2: float) -> float:
    """Return the radius of a circle having the supplied area; not a hazard boundary."""
    area = float(area_km2)
    if not math.isfinite(area) or area < 0:
        raise ValueError("Area in square kilometres must be finite and non-negative.")
    return math.sqrt(area / math.pi)


def mask_to_polygon(
    mask: Any,
    transform: Any,
    crs: Any,
    *,
    validity_mask: Optional[Any] = None,
    nodata: Optional[float] = DEFAULT_NODATA,
    connectivity: int = 4,
    minimum_component_area_m2: float = 0.0,
) -> Any:
    """Polygonize a binary raster using its affine transform, retaining components/holes."""
    values, valid = _validate_binary_mask(mask, validity_mask, nodata)
    _component_structure(connectivity)
    if minimum_component_area_m2 < 0:
        raise ValueError("Minimum component area cannot be negative.")
    transform = _affine(transform)
    polygons = []
    for geometry_mapping, value in shapes(
        (values == 1).astype(np.uint8),
        mask=valid,
        transform=transform,
        connectivity=connectivity,
    ):
        if int(value) != 1:
            continue
        polygon = shape(geometry_mapping)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            raise ValueError("Raster polygonization produced an invalid geometry.")
        if minimum_component_area_m2:
            if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("Polygonization returned a non-polygon geometry.")
            if calculate_area(polygon, crs)["area_m2"] < minimum_component_area_m2:
                continue
        if polygon.geom_type == "Polygon":
            polygons.append(polygon)
        elif polygon.geom_type == "MultiPolygon":
            polygons.extend(polygon.geoms)

    if not polygons:
        return Polygon()
    geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if not geometry.is_valid:
        raise ValueError("Polygonized footprint is invalid after validity repair.")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}.")
    return geometry


def summarize_spatial_prediction(
    hazard: str,
    threshold: float,
    processed_mask: Any,
    geometry: Any,
    crs: Any,
    *,
    nodata: Optional[float] = DEFAULT_NODATA,
    area_crs: Any = DEFAULT_AREA_CRS,
) -> Dict[str, Any]:
    """Create a measurement summary without estimating unavailable values."""
    if hazard not in SUPPORTED_HAZARDS:
        raise ValueError(f"Unsupported hazard {hazard!r}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Probability threshold must be between 0 and 1.")
    values, valid = _validate_binary_mask(processed_mask, None, nodata)
    area = calculate_area(geometry, crs, area_crs)
    polygon_count = 0
    if not geometry.is_empty:
        polygon_count = len(geometry.geoms) if geometry.geom_type == "MultiPolygon" else 1
    return {
        "hazard": hazard,
        "threshold": float(threshold),
        "affected_pixels": int(((values == 1) & valid).sum()),
        **area,
        "equivalent_radius_km": calculate_equivalent_radius(area["area_km2"]),
        "polygon_count": polygon_count,
        "geometry_type": geometry.geom_type,
        "crs": CRS.from_user_input(crs).to_string(),
        "bounds": None if geometry.is_empty else [float(value) for value in geometry.bounds],
    }


def process_segmentation_prediction(
    logits: Any,
    hazard: str,
    transform: Any,
    crs: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    validity_mask: Optional[Any] = None,
    source_nodata: Optional[float] = None,
    nodata_value: int = DEFAULT_NODATA,
    minimum_component_pixels: int = 0,
    minimum_component_area_m2: float = 0.0,
    opening_radius: int = 0,
    closing_radius: int = 0,
    connectivity: int = 4,
    area_crs: Any = DEFAULT_AREA_CRS,
) -> Dict[str, Any]:
    """Run logits-to-footprint post-processing while retaining logits and raw masks."""
    if hazard not in SUPPORTED_HAZARDS:
        raise ValueError(f"Unsupported hazard {hazard!r}.")
    probabilities = _single_band(logits_to_probability(logits), "Probability").astype(np.float32)
    valid = _valid_pixels(probabilities, validity_mask, source_nodata)
    probabilities = probabilities.copy()
    probabilities[~valid] = np.nan
    raw_mask = probability_to_mask(
        probabilities,
        threshold,
        validity_mask=valid,
        nodata_value=nodata_value,
    )
    processed_mask = clean_binary_mask(
        raw_mask,
        validity_mask=valid,
        nodata=nodata_value,
        nodata_value=nodata_value,
        minimum_component_pixels=minimum_component_pixels,
        minimum_component_area_m2=minimum_component_area_m2,
        transform=transform,
        crs=crs,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        connectivity=connectivity,
    )
    geometry = mask_to_polygon(
        processed_mask,
        transform,
        crs,
        validity_mask=valid,
        nodata=nodata_value,
        connectivity=connectivity,
    )
    summary = summarize_spatial_prediction(
        hazard,
        threshold,
        processed_mask,
        geometry,
        crs,
        nodata=nodata_value,
        area_crs=area_crs,
    )
    return {
        "logits": logits,
        "probability": probabilities,
        "raw_mask": raw_mask,
        "processed_mask": processed_mask,
        "geometry": geometry,
        "summary": summary,
    }


def _geojson_geometry(geometry: Any, source_crs: CRS) -> Any:
    if geometry is None or geometry.is_empty:
        return None
    geographic_crs = CRS.from_epsg(4326)
    if source_crs != geographic_crs:
        transformer = Transformer.from_crs(source_crs, geographic_crs, always_xy=True)
        geometry = transform_geometry(transformer.transform, geometry)
    if not geometry.is_valid:
        raise ValueError("Cannot write invalid GeoJSON geometry.")
    return mapping(geometry)


def save_spatial_outputs(
    output_dir: Any,
    prediction: Mapping[str, Any],
    transform: Any,
    crs: Any,
    *,
    nodata_value: int = DEFAULT_NODATA,
) -> Dict[str, str]:
    """Write georeferenced probability/mask rasters, GeoJSON, and a JSON summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    probability = _single_band(prediction["probability"], "Probability").astype(np.float32)
    raw_mask = _single_band(prediction["raw_mask"], "Raw mask").astype(np.uint8)
    processed_mask = _single_band(prediction["processed_mask"], "Processed mask").astype(np.uint8)
    if raw_mask.shape != probability.shape or processed_mask.shape != probability.shape:
        raise ValueError("Probability and mask raster dimensions must match.")
    transform = _affine(transform)
    raster_crs = CRS.from_user_input(crs)
    profile = {
        "driver": "GTiff",
        "height": probability.shape[0],
        "width": probability.shape[1],
        "count": 1,
        "transform": transform,
        "crs": raster_crs,
        "compress": "deflate",
    }
    probability_path = output_path / "prediction_probability.tif"
    with rasterio.open(probability_path, "w", **profile, dtype="float32", nodata=np.nan) as dataset:
        dataset.write(probability, 1)
    raw_mask_path = output_path / "prediction_mask.tif"
    with rasterio.open(raw_mask_path, "w", **profile, dtype="uint8", nodata=nodata_value) as dataset:
        dataset.write(raw_mask, 1)
    processed_mask_path = output_path / "prediction_mask_processed.tif"
    with rasterio.open(processed_mask_path, "w", **profile, dtype="uint8", nodata=nodata_value) as dataset:
        dataset.write(processed_mask, 1)

    geometry = prediction["geometry"]
    summary = dict(prediction["summary"])
    feature = {
        "type": "Feature",
        "geometry": _geojson_geometry(geometry, raster_crs),
        "properties": {
            "hazard": summary["hazard"],
            "threshold": summary["threshold"],
            "source_crs": raster_crs.to_string(),
            "area_km2": summary["area_km2"],
            "equivalent_radius_km": summary["equivalent_radius_km"],
        },
    }
    geojson_path = output_path / "prediction_polygon.geojson"
    with geojson_path.open("w", encoding="utf-8") as output_file:
        json.dump({"type": "FeatureCollection", "features": [feature]}, output_file, allow_nan=False)
    summary_path = output_path / "spatial_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=False)

    return {
        "probability_raster": str(probability_path),
        "raw_mask_raster": str(raw_mask_path),
        "processed_mask_raster": str(processed_mask_path),
        "polygon_geojson": str(geojson_path),
        "summary_json": str(summary_path),
    }
