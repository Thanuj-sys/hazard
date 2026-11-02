import numpy as np
import pytest
import torch
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds, from_bounds, from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window, transform as window_transform

from data.datasets import CASLandslideDataset, HLSBurnScarDataset, S1S2WaterDataset
from data.preprocessing import (
    AlignedRaster,
    RasterGrid,
    align_mask_to_reference,
    align_raster_to_reference,
    align_sample_to_reference,
    normalize_feature,
    tile_aligned_rasters,
    validate_target_mask,
)


def grid_metadata(crs, transform, width, height, nodata=None):
    grid = RasterGrid(CRS.from_user_input(crs), transform, width, height)
    return grid.to_metadata(nodata)


def test_continuous_raster_reprojects_crs_resolution_and_grid_metadata():
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    source_metadata = grid_metadata(
        "EPSG:4326", from_origin(0, 4, 1, 1), width=4, height=4
    )
    projected_bounds = transform_bounds(
        "EPSG:4326", "EPSG:3857", *source_metadata["bounds"], densify_pts=21
    )
    target_transform = from_bounds(*projected_bounds, width=8, height=8)
    target_metadata = grid_metadata("EPSG:3857", target_transform, 8, 8)

    aligned = align_raster_to_reference(source, source_metadata, target_metadata)

    assert aligned.data.shape == (8, 8)
    assert aligned.grid.crs == CRS.from_epsg(3857)
    assert aligned.metadata["resolution"] == target_metadata["resolution"]
    assert aligned.metadata["transform"] == target_transform
    assert np.allclose(aligned.metadata["bounds"], projected_bounds)
    assert aligned.valid_mask.any()
    assert aligned.data.dtype == np.float32


def test_mask_uses_nearest_neighbor_and_keeps_nodata_invalid():
    mask = np.array([[0, 1], [-1, 1]], dtype=np.int16)
    source_metadata = grid_metadata(
        "EPSG:4326", from_origin(0, 2, 1, 1), width=2, height=2, nodata=-1
    )
    projected_bounds = transform_bounds(
        "EPSG:4326", "EPSG:3857", *source_metadata["bounds"], densify_pts=21
    )
    target_transform = from_bounds(*projected_bounds, width=8, height=8)
    target_metadata = grid_metadata("EPSG:3857", target_transform, 8, 8)

    aligned = align_mask_to_reference(
        mask, source_metadata, target_metadata, expected_classes=(0, 1)
    )

    assert aligned.data.shape == (8, 8)
    assert set(np.unique(aligned.data[aligned.valid_mask])).issubset({0, 1})
    assert not aligned.valid_mask.all()
    assert aligned.nodata == -1
    assert validate_target_mask(
        aligned.data,
        aligned.metadata,
        reference_metadata=target_metadata,
        expected_classes=(0, 1),
    ).sum() == aligned.valid_mask.sum()


def test_mask_validation_rejects_fractional_labels_and_grid_mismatch():
    metadata = grid_metadata("EPSG:4326", from_origin(0, 2, 1, 1), 2, 2)
    fractional = np.array([[0.0, 0.25], [1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="fractional"):
        validate_target_mask(fractional, metadata)

    valid_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    other_grid = grid_metadata("EPSG:4326", from_origin(0, 2, 0.5, 0.5), 4, 4)
    with pytest.raises(ValueError, match="not aligned"):
        validate_target_mask(valid_mask, metadata, reference_metadata=other_grid)


def test_nodata_is_excluded_from_validity_and_feature_ranges_are_preserved():
    values = np.array([[2.0, -9999.0], [-1.5, 4.0]], dtype=np.float32)
    metadata = grid_metadata(
        "EPSG:4326", from_origin(0, 2, 1, 1), 2, 2, nodata=-9999.0
    )
    aligned = align_raster_to_reference(values, metadata, metadata)

    assert not aligned.valid_mask[0, 1]
    assert aligned.data[0, 1] == -9999.0
    assert np.array_equal(
        normalize_feature(values, "dNBR", scale=None, nodata=-9999.0), values
    )
    categories = np.array([[0, 5], [7, 5]], dtype=np.uint8)
    assert np.array_equal(normalize_feature(categories, "categorical", scale=100), categories)


def test_masked_source_pixels_become_explicit_nodata():
    source = np.ma.array(
        [[0, 1], [2, 3]],
        mask=[[False, True], [False, False]],
    )
    metadata = grid_metadata("EPSG:4326", from_origin(0, 2, 1, 1), 2, 2)

    aligned = align_raster_to_reference(source, metadata, metadata, feature_type="categorical")

    assert aligned.nodata is not None
    assert not aligned.valid_mask[0, 1]
    assert aligned.data[0, 1] == aligned.nodata


def test_sample_alignment_prefers_mask_grid_and_supports_inference_without_mask():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    image_metadata = grid_metadata("EPSG:4326", from_origin(0, 4, 1, 1), 4, 4)
    mask_metadata = grid_metadata("EPSG:4326", from_origin(1, 3, 0.5, 0.5), 4, 4, nodata=255)
    sample = {
        "image": image,
        "geospatial_features": None,
        "target_mask": torch.tensor([[0, 1, 0, 1]] * 4, dtype=torch.uint8),
        "severity_mask": None,
        "event_id": "EVT",
        "hazard": "flood",
        "metadata": {
            "raster": {"image": image_metadata, "target_mask": mask_metadata},
            "source_paths": {"target_mask": None},
        },
    }

    aligned = align_sample_to_reference(sample, expected_mask_classes=(0, 1))
    assert aligned["aligned_image"].shape[-2:] == (4, 4)
    assert aligned["aligned_target_mask"].shape == (4, 4)
    assert aligned["crs"] == "EPSG:4326"
    assert aligned["event_id"] == "EVT"
    assert aligned["metadata"]["aligned_grid"]["transform"] == mask_metadata["transform"]
    assert aligned["validity_masks"]["target_mask"].all()

    inference_sample = {**sample, "target_mask": None}
    image_metadata_with_nodata = {
        **image_metadata,
        "nodata": -9999.0,
    }
    inference_sample["metadata"] = {
        "raster": {"image": image_metadata_with_nodata, "target_mask": None},
        "source_paths": {"target_mask": None},
    }
    inference_aligned = align_sample_to_reference(
        inference_sample,
        normalization={"image": {"feature_type": "reflectance", "scale": 2.0}},
    )
    assert inference_aligned["aligned_target_mask"] is None
    assert inference_aligned["validity_masks"]["target_mask"] is None
    assert inference_aligned["metadata"]["aligned_grid"]["crs"] == "EPSG:4326"
    assert inference_aligned["nodata"] == -9999.0
    assert torch.allclose(inference_aligned["aligned_image"], image / 2.0)


def test_ancillary_categorical_feature_is_aligned_without_fractional_classes():
    image_metadata = grid_metadata("EPSG:4326", from_origin(0, 2, 1, 1), 2, 2)
    image = np.ones((1, 2, 2), dtype=np.float32)
    landcover = np.array([[2, 7], [7, 2]], dtype=np.uint8)
    ancillary = {
        "landcover": {
            "array": landcover,
            "metadata": image_metadata,
            "feature_type": "categorical",
        }
    }
    sample = {
        "image": image,
        "geospatial_features": None,
        "target_mask": None,
        "severity_mask": None,
        "event_id": None,
        "hazard": "wildfire",
        "metadata": {
            "raster": {"image": image_metadata, "target_mask": None},
            "source_paths": {"target_mask": None},
        },
    }

    aligned = align_sample_to_reference(sample, ancillary_features=ancillary)
    classes = torch.unique(aligned["aligned_geospatial_features"]["landcover"])
    assert set(classes.tolist()).issubset({2, 7})
    assert aligned["aligned_geospatial_features"]["landcover"].dtype == torch.uint8


def test_tile_windows_share_transform_and_bounds_for_image_and_mask():
    metadata = grid_metadata("EPSG:32643", from_origin(500000, 2500000, 10, 10), 5, 4)
    image = np.ones((3, 4, 5), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=np.uint8)

    tiles = tile_aligned_rasters(
        {"image": image, "mask": mask},
        metadata,
        tile_size=3,
        overlap=1,
        nodata={"mask": 255},
    )

    assert len(tiles) == 4
    for tile in tiles:
        assert tile["rasters"]["image"].shape[-2:] == tile["rasters"]["mask"].shape
        window_data = tile["window"]
        expected_transform = window_transform(
            Window(
                window_data["col_off"],
                window_data["row_off"],
                window_data["width"],
                window_data["height"],
            ),
            metadata["transform"],
        )
        assert tile["metadata"]["transform"] == expected_transform
        assert tile["raster_metadata"]["mask"]["nodata"] == 255
        height, width = tile["rasters"]["mask"].shape
        bounds = array_bounds(height, width, tile["metadata"]["transform"])
        assert np.allclose(bounds, tile["metadata"]["bounds"])
        assert tile["metadata"]["resolution"] == metadata["resolution"]


def test_staged_dataset_samples_align_to_their_mask_grids():
    for loader in (S1S2WaterDataset, HLSBurnScarDataset, CASLandslideDataset):
        sample = loader()[0]
        aligned = align_sample_to_reference(sample)
        grid = aligned["metadata"]["aligned_grid"]
        assert aligned["aligned_image"].shape[-2:] == (grid["height"], grid["width"])
        assert aligned["aligned_target_mask"].shape == (grid["height"], grid["width"])
        assert aligned["crs"] == grid["crs"]
        valid_labels = aligned["aligned_target_mask"][aligned["validity_masks"]["target_mask"]]
        assert torch.equal(valid_labels, valid_labels.round())
