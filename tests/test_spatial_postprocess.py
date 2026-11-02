import json
import math

import numpy as np
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from spatial.postprocess import (
    calculate_area,
    calculate_equivalent_radius,
    clean_binary_mask,
    logits_to_probability,
    mask_to_polygon,
    process_segmentation_prediction,
    probability_to_mask,
    save_spatial_outputs,
    summarize_spatial_prediction,
)
from spatial.visualization import save_footprint_overlay


def test_logits_probability_and_default_threshold_are_correct():
    logits = torch.tensor([[[-20.0, 0.0, 20.0]]])
    probabilities = logits_to_probability(logits)

    assert torch.allclose(probabilities, torch.tensor([[[0.0, 0.5, 1.0]]]), atol=1e-6)
    mask = probability_to_mask(probabilities)
    assert mask.tolist() == [[0, 1, 1]]
    assert set(np.unique(mask)).issubset({0, 1})


def test_custom_threshold_and_nodata_are_respected():
    probability = np.array([[0.49, 0.75], [np.nan, 0.9]], dtype=np.float32)
    mask = probability_to_mask(probability, threshold=0.8, nodata_value=255)

    assert mask.tolist() == [[0, 0], [255, 1]]
    assert set(np.unique(mask)).issubset({0, 1, 255})


def test_cleanup_can_remove_small_isolated_components_conservatively():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 1
    mask[4:7, 4:7] = 1

    raw = clean_binary_mask(mask)
    cleaned = clean_binary_mask(mask, minimum_component_pixels=2)

    assert raw.sum() == 10
    assert cleaned.sum() == 9
    assert cleaned[1, 1] == 0
    assert cleaned[4:7, 4:7].all()

    area_cleaned = clean_binary_mask(
        mask,
        minimum_component_area_m2=200.0,
        transform=from_origin(0, 80, 10, 10),
        crs="EPSG:32643",
    )
    assert area_cleaned.sum() == 9
    assert area_cleaned[1, 1] == 0


def test_polygonization_applies_affine_transform_and_preserves_components():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    mask[3:5, 5:6] = 1
    transform = from_origin(100.0, 200.0, 10.0, 10.0)

    geometry = mask_to_polygon(mask, transform, "EPSG:32643")

    assert geometry.geom_type == "MultiPolygon"
    assert len(geometry.geoms) == 2
    assert geometry.is_valid
    assert geometry.bounds == (120.0, 150.0, 160.0, 190.0)


def test_polygonization_retains_holes():
    mask = np.ones((3, 3), dtype=np.uint8)
    mask[1, 1] = 0

    geometry = mask_to_polygon(mask, from_origin(0, 30, 10, 10), "EPSG:32643")

    assert geometry.geom_type == "Polygon"
    assert len(geometry.interiors) == 1


def test_projected_area_and_geographic_area_are_measured_in_equal_area_crs():
    square = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    projected = calculate_area(square, "EPSG:6933")
    geographic_square = Polygon([(78.0, 20.0), (78.01, 20.0), (78.01, 20.01), (78.0, 20.01)])
    geographic = calculate_area(geographic_square, "EPSG:4326")

    assert projected["area_m2"] == pytest.approx(10_000.0, rel=1e-8)
    assert projected["area_km2"] == pytest.approx(0.01, rel=1e-8)
    assert geographic["area_m2"] > 0
    assert geographic["area_km2"] == pytest.approx(geographic["area_m2"] / 1_000_000)


def test_equivalent_radius_is_sqrt_area_over_pi():
    area_km2 = math.pi * 25.0
    assert calculate_equivalent_radius(area_km2) == pytest.approx(5.0)
    assert calculate_equivalent_radius(0.0) == 0.0


def test_empty_mask_has_zero_area_and_radius_without_error():
    mask = np.zeros((4, 4), dtype=np.uint8)
    result = process_segmentation_prediction(
        np.full((4, 4), -20.0),
        "flood",
        from_origin(0, 40, 10, 10),
        "EPSG:32643",
    )

    assert result["geometry"].is_empty
    assert result["summary"]["area_m2"] == 0.0
    assert result["summary"]["area_km2"] == 0.0
    assert result["summary"]["equivalent_radius_km"] == 0.0
    assert result["summary"]["affected_pixels"] == 0


def test_nodata_and_all_hazard_masks_are_supported():
    mask = np.array([[1, 1, 255], [1, 1, 255]], dtype=np.uint8)
    logits = np.where(mask == 1, 20.0, np.nan).astype(np.float32)
    result = process_segmentation_prediction(
        logits,
        "landslide",
        from_origin(0, 20, 10, 10),
        "EPSG:32643",
        validity_mask=mask != 255,
    )

    assert result["summary"]["affected_pixels"] == 4
    assert result["summary"]["polygon_count"] == 1
    assert not result["geometry"].is_empty
    assert (result["raw_mask"][:, 2] == 255).all()


def test_summary_reports_hazard_crs_bounds_and_components():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[3, 3] = 1
    geometry = mask_to_polygon(mask, from_origin(500000, 2500000, 10, 10), "EPSG:32643")

    summary = summarize_spatial_prediction(
        "wildfire", 0.5, mask, geometry, "EPSG:32643"
    )

    assert summary["hazard"] == "wildfire"
    assert summary["threshold"] == 0.5
    assert summary["affected_pixels"] == 2
    assert summary["polygon_count"] == 2
    assert summary["geometry_type"] == "MultiPolygon"
    assert summary["crs"] == "EPSG:32643"
    assert summary["bounds"] == list(geometry.bounds)
    assert summary["area_m2"] > 0


def test_saved_rasters_geojson_and_summary_preserve_spatial_metadata(tmp_path):
    logits = np.full((6, 6), -20.0, dtype=np.float32)
    logits[2:4, 1:3] = 20.0
    transform = from_origin(500000, 2500000, 10, 10)
    result = process_segmentation_prediction(
        logits, "wildfire", transform, "EPSG:32643"
    )

    paths = save_spatial_outputs(tmp_path, result, transform, "EPSG:32643")

    assert set(paths) == {
        "probability_raster",
        "raw_mask_raster",
        "processed_mask_raster",
        "polygon_geojson",
        "summary_json",
    }
    for key in ("probability_raster", "raw_mask_raster", "processed_mask_raster"):
        with rasterio.open(paths[key]) as raster:
            assert raster.crs == CRS.from_epsg(32643)
            assert raster.transform == transform
            assert raster.width == 6 and raster.height == 6
            assert raster.nodata is not None
    with open(paths["polygon_geojson"], encoding="utf-8") as geojson_file:
        geojson = json.load(geojson_file)
    assert geojson["features"][0]["properties"]["source_crs"] == "EPSG:32643"
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"
    with open(paths["summary_json"], encoding="utf-8") as summary_file:
        saved_summary = json.load(summary_file)
    assert saved_summary["hazard"] == "wildfire"
    assert saved_summary["crs"] == "EPSG:32643"


def test_overlay_saves_rgb_preview_with_mask(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    output_path = tmp_path / "overlay.png"

    saved_path = save_footprint_overlay(
        image, mask, output_path, bands=(0, 1, 2)
    )

    assert output_path.exists()
    assert saved_path == str(output_path)


def test_spatial_pipeline_is_hazard_specific():
    for hazard in ("flood", "landslide", "wildfire"):
        result = process_segmentation_prediction(
            np.zeros((2, 2), dtype=np.float32),
            hazard,
            from_origin(0, 20, 10, 10),
            "EPSG:32643",
        )
        assert result["summary"]["hazard"] == hazard
