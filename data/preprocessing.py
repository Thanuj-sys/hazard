"""Geospatial preprocessing, alignment, nodata handling, and raster tiling."""
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, transform as window_transform

_CATEGORICAL_FEATURE_TYPES = {
    "categorical", "class", "mask", "binary", "landcover", "land_cover",
    "land_cover_class", "geology", "geology_class", "soil", "soil_class",
}


def _is_categorical_feature(feature_type: str) -> bool:
    return feature_type.lower().replace("-", "_") in _CATEGORICAL_FEATURE_TYPES


@dataclass(frozen=True)
class RasterGrid:
    """A single affine target grid in a projected or geographic CRS."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "RasterGrid":
        required = ("crs", "transform", "width", "height", "resolution")
        missing = [key for key in required if metadata.get(key) is None]
        if missing:
            raise ValueError(f"Raster grid metadata is missing: {', '.join(missing)}.")
        crs = CRS.from_user_input(metadata["crs"])
        transform = metadata["transform"]
        if not isinstance(transform, Affine):
            coefficients = tuple(transform)
            transform = Affine(*coefficients[:6])
        width, height = int(metadata["width"]), int(metadata["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Raster grid width and height must be positive.")
        expected_resolution = _transform_resolution(transform)
        supplied_resolution = tuple(float(value) for value in metadata["resolution"])
        if len(supplied_resolution) != 2 or not np.allclose(
            supplied_resolution, expected_resolution, rtol=1e-6, atol=1e-8
        ):
            raise ValueError("Raster resolution does not match its affine transform.")
        return cls(crs, transform, width, height)

    @property
    def resolution(self) -> Tuple[float, float]:
        return _transform_resolution(self.transform)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        corners = [
            self.transform @ (0, 0),
            self.transform @ (self.width, 0),
            self.transform @ (0, self.height),
            self.transform @ (self.width, self.height),
        ]
        xs, ys = zip(*corners)
        return min(xs), min(ys), max(xs), max(ys)

    def to_metadata(self, nodata: Any = None) -> Dict[str, Any]:
        return {
            "crs": self.crs.to_string(),
            "transform": self.transform,
            "resolution": self.resolution,
            "bounds": self.bounds,
            "width": self.width,
            "height": self.height,
            "nodata": nodata,
        }


@dataclass
class AlignedRaster:
    """Raster data plus an explicit pixel-validity mask and its target grid."""

    data: np.ndarray
    valid_mask: np.ndarray
    grid: RasterGrid
    nodata: Any

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.grid.to_metadata(self.nodata)


def _transform_resolution(transform: Affine) -> Tuple[float, float]:
    return (
        math.hypot(transform.a, transform.d),
        math.hypot(transform.b, transform.e),
    )


def _as_bands(array: Any) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    masked_values = None
    if np.ma.isMaskedArray(array):
        masked_values = np.ma.getmaskarray(array)
        array = np.ma.getdata(array)
    values = np.asarray(array)
    if values.ndim == 2:
        return values[np.newaxis, ...], True, masked_values
    if values.ndim == 3:
        return values, False, masked_values
    raise ValueError("Raster data must have shape (height, width) or (bands, height, width).")


def _is_nodata(values: np.ndarray, nodata: Any) -> np.ndarray:
    if nodata is None:
        return np.zeros(values.shape, dtype=bool)
    try:
        if np.isnan(nodata):
            return np.isnan(values)
    except TypeError:
        pass
    return values == nodata


def _valid_source_pixels(
    bands: np.ndarray,
    nodata: Any,
    masked_values: Optional[np.ndarray],
) -> np.ndarray:
    valid = np.isfinite(bands) if np.issubdtype(bands.dtype, np.floating) else np.ones(bands.shape, bool)
    valid &= ~_is_nodata(bands, nodata)
    if masked_values is not None:
        if masked_values.ndim == 2:
            masked_values = masked_values[np.newaxis, ...]
        valid &= ~masked_values
    return np.all(valid, axis=0)


def _materialize_masked_nodata(
    bands: np.ndarray,
    masked_values: Optional[np.ndarray],
    nodata: Any,
) -> Tuple[np.ndarray, Any]:
    """Ensure masked source cells are represented by a nodata value during warping."""
    if masked_values is None or not np.any(masked_values):
        return bands, nodata
    if masked_values.ndim == 2:
        masked_values = masked_values[np.newaxis, ...]
    if nodata is None:
        if np.issubdtype(bands.dtype, np.floating):
            nodata = np.nan
        else:
            _, nodata = _destination_storage(bands, Resampling.nearest, None, None)
    if np.issubdtype(bands.dtype, np.floating):
        result = bands.astype(np.float32, copy=True)
    else:
        info = np.iinfo(bands.dtype)
        dtype = bands.dtype if info.min <= nodata <= info.max else np.dtype(np.int32)
        result = bands.astype(dtype, copy=True)
    result[masked_values] = nodata
    return result, nodata


def _destination_storage(
    source: np.ndarray,
    resampling: Resampling,
    src_nodata: Any,
    requested_nodata: Any,
) -> Tuple[np.dtype, Any]:
    continuous = resampling != Resampling.nearest or np.issubdtype(source.dtype, np.floating)
    if continuous:
        nodata = requested_nodata if requested_nodata is not None else src_nodata
        if nodata is None:
            nodata = np.nan
        return np.dtype(np.float32), nodata

    if requested_nodata is not None:
        nodata = requested_nodata
    elif src_nodata is not None:
        nodata = src_nodata
    else:
        nodata = None

    dtype = np.dtype(source.dtype)
    if nodata is not None:
        info = np.iinfo(dtype)
        if not info.min <= nodata <= info.max:
            dtype = np.dtype(np.int32)
    else:
        if np.issubdtype(dtype, np.bool_):
            dtype = np.dtype(np.uint8)
        info = np.iinfo(dtype)
        if not np.any(source == info.min):
            nodata = info.min
        elif not np.any(source == info.max):
            nodata = info.max
        else:
            occupied = np.unique(source)
            candidate_range = np.arange(info.min, info.min + occupied.size + 1, dtype=dtype)
            unused = np.setdiff1d(candidate_range, occupied, assume_unique=False)
            if unused.size:
                nodata = unused[0].item()
            else:
                dtype = np.dtype(np.int32 if dtype.itemsize < 4 else np.int64)
                nodata = np.iinfo(dtype).min
        if not np.issubdtype(dtype, np.integer):
            dtype = np.dtype(np.int32)
    return dtype, nodata


def reproject_to_grid(
    source: Any,
    source_metadata: Mapping[str, Any],
    target_grid: RasterGrid | Mapping[str, Any],
    *,
    resampling: Resampling = Resampling.bilinear,
    src_nodata: Any = None,
    dst_nodata: Any = None,
) -> AlignedRaster:
    """Warp a raster to an explicit grid and return validity separately from values."""
    bands, was_2d, masked_values = _as_bands(source)
    source_grid = RasterGrid.from_metadata(source_metadata)
    grid = target_grid if isinstance(target_grid, RasterGrid) else RasterGrid.from_metadata(target_grid)
    source_nodata = source_metadata.get("nodata") if src_nodata is None else src_nodata
    if source_nodata is None and np.issubdtype(bands.dtype, np.floating) and np.isnan(bands).any():
        source_nodata = np.nan
    warp_bands, source_nodata = _materialize_masked_nodata(
        bands, masked_values, source_nodata
    )
    destination_dtype, destination_nodata = _destination_storage(
        warp_bands, resampling, source_nodata, dst_nodata
    )

    destination = np.full(
        (bands.shape[0], grid.height, grid.width),
        destination_nodata,
        dtype=destination_dtype,
    )
    for band_index, source_band in enumerate(warp_bands):
        reproject(
            source=source_band,
            destination=destination[band_index],
            src_transform=source_grid.transform,
            src_crs=source_grid.crs,
            src_nodata=source_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=destination_nodata,
            resampling=resampling,
            init_dest_nodata=True,
        )

    source_valid = _valid_source_pixels(bands, source_nodata, masked_values).astype(np.uint8)
    destination_valid = np.zeros((grid.height, grid.width), dtype=np.uint8)
    reproject(
        source=source_valid,
        destination=destination_valid,
        src_transform=source_grid.transform,
        src_crs=source_grid.crs,
        src_nodata=0,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
        init_dest_nodata=True,
    )
    valid_mask = destination_valid.astype(bool)
    if np.issubdtype(destination.dtype, np.floating):
        valid_mask &= np.all(np.isfinite(destination), axis=0)
    if destination_nodata is not None:
        if isinstance(destination_nodata, (float, np.floating)) and np.isnan(destination_nodata):
            valid_mask &= ~np.any(np.isnan(destination), axis=0)
        else:
            valid_mask &= ~np.any(destination == destination_nodata, axis=0)
    destination[:, ~valid_mask] = destination_nodata
    output = destination[0] if was_2d else destination
    return AlignedRaster(output, valid_mask, grid, destination_nodata)


def align_raster_to_reference(
    source: Any,
    source_metadata: Mapping[str, Any],
    reference_metadata: RasterGrid | Mapping[str, Any],
    *,
    feature_type: str = "continuous",
    resampling: Optional[Resampling] = None,
    dst_nodata: Any = None,
) -> AlignedRaster:
    """Align one raster; categorical values always use nearest-neighbor resampling."""
    categorical = _is_categorical_feature(feature_type)
    method = resampling or (Resampling.nearest if categorical else Resampling.bilinear)
    if categorical and method != Resampling.nearest:
        raise ValueError("Categorical rasters must use nearest-neighbor resampling.")
    return reproject_to_grid(
        source,
        source_metadata,
        reference_metadata,
        resampling=method,
        dst_nodata=dst_nodata,
    )


def validate_target_mask(
    mask: Any,
    metadata: Mapping[str, Any],
    *,
    reference_metadata: Optional[Mapping[str, Any] | RasterGrid] = None,
    expected_classes: Optional[Sequence[int | float]] = None,
    required: bool = True,
) -> Optional[np.ndarray]:
    """Validate mask grid, dtype, discrete classes, and nodata; return pixel validity."""
    if mask is None:
        if required:
            raise ValueError("A target mask is required for this sample.")
        return None

    bands, was_2d, masked_values = _as_bands(mask)
    if bands.shape[0] != 1:
        raise ValueError("A target mask must contain exactly one band.")
    values = bands[0]
    if not (np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError("Target mask dtype must be numeric or boolean.")
    grid = RasterGrid.from_metadata(metadata)
    if values.shape != (grid.height, grid.width):
        raise ValueError(
            f"Target mask dimensions {values.shape[::-1]} do not match metadata "
            f"({grid.width}, {grid.height})."
        )

    nodata = metadata.get("nodata")
    valid = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else np.ones(values.shape, bool)
    valid &= ~_is_nodata(values, nodata)
    if masked_values is not None:
        if masked_values.ndim == 3:
            masked_values = masked_values[0]
        valid &= ~masked_values
    if not valid.any():
        raise ValueError("Target mask contains no valid pixels.")
    valid_values = values[valid]
    if np.issubdtype(valid_values.dtype, np.floating) and not np.allclose(
        valid_values, np.rint(valid_values), rtol=0.0, atol=1e-6
    ):
        raise ValueError("Target mask contains fractional labels; masks must remain categorical.")
    if expected_classes is not None:
        unexpected = np.setdiff1d(np.unique(valid_values), np.asarray(expected_classes))
        if unexpected.size:
            raise ValueError(f"Target mask contains unexpected class values: {unexpected.tolist()}.")

    if reference_metadata is not None:
        reference_grid = (
            reference_metadata
            if isinstance(reference_metadata, RasterGrid)
            else RasterGrid.from_metadata(reference_metadata)
        )
        if (
            grid.crs != reference_grid.crs
            or grid.width != reference_grid.width
            or grid.height != reference_grid.height
            or not grid.transform.almost_equals(reference_grid.transform)
        ):
            raise ValueError("Target mask is not aligned to the reference grid.")
    return valid


def align_mask_to_reference(
    mask: Any,
    mask_metadata: Mapping[str, Any],
    reference_metadata: RasterGrid | Mapping[str, Any],
    *,
    expected_classes: Optional[Sequence[int | float]] = None,
) -> AlignedRaster:
    """Validate and align a ground-truth mask with nearest-neighbor only."""
    validate_target_mask(mask, mask_metadata)
    aligned = reproject_to_grid(
        mask,
        mask_metadata,
        reference_metadata,
        resampling=Resampling.nearest,
    )
    validate_target_mask(
        aligned.data,
        aligned.metadata,
        reference_metadata=aligned.grid,
        expected_classes=expected_classes,
    )
    return aligned


def normalize_feature(
    array: Any,
    feature_type: str,
    *,
    valid_mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    nodata: Any = None,
) -> np.ndarray:
    """Apply configurable feature-aware scaling without treating classes as magnitudes."""
    values = np.asarray(array)
    kind = feature_type.lower()
    categorical = _is_categorical_feature(kind)
    if categorical:
        if np.issubdtype(values.dtype, np.floating):
            finite = np.isfinite(values)
            if not np.allclose(values[finite], np.rint(values[finite]), rtol=0.0, atol=1e-6):
                raise ValueError("Categorical feature contains fractional class values.")
        return values.copy()

    if scale is not None and scale <= 0:
        raise ValueError("Feature scale must be positive.")
    if (mean is None) != (std is None):
        raise ValueError("Feature standardization requires both mean and std.")
    if std is not None and std <= 0:
        raise ValueError("Feature std must be positive.")

    result = values.astype(np.float32, copy=True)
    if scale is not None:
        result /= float(scale)
    if mean is not None and std is not None:
        result = (result - float(mean)) / float(std)
    if valid_mask is not None:
        if valid_mask.shape != result.shape[-2:]:
            raise ValueError("Feature validity mask dimensions do not match the raster.")
        fill = nodata if nodata is not None else np.nan
        result[..., ~valid_mask] = fill
    return result


def _metadata_from_dataset(dataset: rasterio.io.DatasetReader) -> Dict[str, Any]:
    if dataset.crs is None:
        raise ValueError(f"Raster {dataset.name!r} has no CRS.")
    return {
        "crs": dataset.crs.to_string(),
        "transform": dataset.transform,
        "resolution": tuple(float(value) for value in dataset.res),
        "width": dataset.width,
        "height": dataset.height,
        "bounds": tuple(dataset.bounds),
        "nodata": dataset.nodata,
    }


def align_sample_to_reference(
    sample: Mapping[str, Any],
    *,
    ancillary_features: Optional[Mapping[str, Mapping[str, Any]]] = None,
    normalization: Optional[Mapping[str, Mapping[str, Any]]] = None,
    expected_mask_classes: Optional[Sequence[int | float]] = None,
) -> Dict[str, Any]:
    """Create aligned tensor fields while preserving the Phase 2 sample contract."""
    metadata = sample.get("metadata") or {}
    raster_metadata = metadata.get("raster") or {}
    target_mask = sample.get("target_mask")
    image_metadata = raster_metadata.get("image") or metadata
    mask_metadata = raster_metadata.get("target_mask")
    if target_mask is not None:
        if mask_metadata is None:
            raise ValueError("A target mask was supplied without raster metadata.")
        validate_target_mask(target_mask, mask_metadata, expected_classes=expected_mask_classes)
        reference_grid = RasterGrid.from_metadata(mask_metadata)
    else:
        reference_grid = RasterGrid.from_metadata(image_metadata)

    source_paths = metadata.get("source_paths") or {}
    input_paths = [(name, path) for name, path in source_paths.items() if name != "target_mask" and path]
    input_scales = dict(metadata.get("input_normalization") or {})
    if normalization:
        input_scales.update(normalization)
    aligned_inputs: List[np.ndarray] = []
    input_validity: List[np.ndarray] = []
    input_grids: Dict[str, Dict[str, Any]] = {}

    if input_paths:
        for name, path in input_paths:
            with rasterio.open(path) as source_dataset:
                source_array = source_dataset.read()
                source_grid_metadata = _metadata_from_dataset(source_dataset)
            aligned = align_raster_to_reference(
                source_array,
                source_grid_metadata,
                reference_grid,
                feature_type="continuous",
            )
            settings = input_scales.get(name, {})
            normalized = normalize_feature(
                aligned.data,
                settings.get("feature_type", "continuous"),
                valid_mask=aligned.valid_mask,
                scale=settings.get("scale"),
                mean=settings.get("mean"),
                std=settings.get("std"),
                nodata=aligned.nodata,
            )
            if normalized.ndim == 2:
                normalized = normalized[np.newaxis, ...]
            aligned_inputs.append(normalized)
            input_validity.append(aligned.valid_mask)
            input_grids[name] = aligned.metadata
        aligned_image = np.concatenate(aligned_inputs, axis=0)
        image_validity = np.logical_and.reduce(input_validity)
    else:
        aligned = align_raster_to_reference(
            sample["image"],
            image_metadata,
            reference_grid,
            feature_type="continuous",
        )
        settings = input_scales.get("image", {})
        aligned_image = normalize_feature(
            aligned.data,
            settings.get("feature_type", "continuous"),
            valid_mask=aligned.valid_mask,
            scale=settings.get("scale"),
            mean=settings.get("mean"),
            std=settings.get("std"),
            nodata=aligned.nodata,
        )
        if aligned_image.ndim == 2:
            aligned_image = aligned_image[np.newaxis, ...]
        image_validity = aligned.valid_mask
        input_grids["image"] = aligned.metadata

    aligned_mask = None
    mask_validity = None
    if target_mask is not None:
        mask_result = align_mask_to_reference(
            target_mask,
            mask_metadata,
            reference_grid,
            expected_classes=expected_mask_classes,
        )
        aligned_mask = mask_result.data
        mask_validity = mask_result.valid_mask

    features = dict(sample.get("geospatial_features") or {})
    if ancillary_features:
        features.update(ancillary_features)
    aligned_features: Dict[str, torch.Tensor] = {}
    feature_validity: Dict[str, torch.Tensor] = {}
    feature_grids: Dict[str, Dict[str, Any]] = {}
    for name, feature in features.items():
        if not isinstance(feature, Mapping) or "array" not in feature or "metadata" not in feature:
            raise ValueError(
                f"Ancillary feature {name!r} must provide array and metadata entries."
            )
        kind = feature.get("feature_type", "continuous")
        feature_result = align_raster_to_reference(
            feature["array"],
            feature["metadata"],
            reference_grid,
            feature_type=kind,
        )
        feature_normalization = dict(feature.get("normalization") or {})
        feature_normalization.pop("nodata", None)
        normalized = normalize_feature(
            feature_result.data,
            kind,
            valid_mask=feature_result.valid_mask,
            **feature_normalization,
            nodata=feature_result.nodata,
        )
        aligned_features[name] = torch.from_numpy(np.ascontiguousarray(normalized))
        feature_validity[name] = torch.from_numpy(feature_result.valid_mask.copy())
        feature_grids[name] = feature_result.metadata

    result = dict(sample)
    result["aligned_image"] = torch.from_numpy(np.ascontiguousarray(aligned_image))
    result["aligned_target_mask"] = (
        torch.from_numpy(np.ascontiguousarray(aligned_mask)) if aligned_mask is not None else None
    )
    result["aligned_geospatial_features"] = aligned_features
    result["validity_masks"] = {
        "image": torch.from_numpy(image_validity.copy()),
        "target_mask": torch.from_numpy(mask_validity.copy()) if mask_validity is not None else None,
        "geospatial_features": feature_validity,
    }
    result["crs"] = reference_grid.crs.to_string()
    result["transform"] = reference_grid.transform
    result["resolution"] = reference_grid.resolution
    result["bounds"] = reference_grid.bounds
    result["width"] = reference_grid.width
    result["height"] = reference_grid.height
    reference_metadata = mask_metadata if target_mask is not None else image_metadata
    reference_nodata = reference_metadata.get("nodata")
    result["nodata"] = reference_nodata
    result["metadata"] = {
        **metadata,
        "aligned_grid": reference_grid.to_metadata(reference_nodata),
        "aligned_rasters": {
            "image_inputs": input_grids,
            "target_mask": mask_metadata if target_mask is None else mask_result.metadata,
            "geospatial_features": feature_grids,
        },
    }
    return result


def align_sentinel_pairs(
    s1_data: np.ndarray,
    s2_data: np.ndarray,
    target_crs: Any = None,
    target_resolution: float = 10.0,
    *,
    s1_metadata: Optional[Mapping[str, Any]] = None,
    s2_metadata: Optional[Mapping[str, Any]] = None,
    reference_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[AlignedRaster, AlignedRaster]:
    """Align Sentinel arrays using explicit source metadata, never array shape alone."""
    if s1_metadata is None or s2_metadata is None:
        raise ValueError("Sentinel alignment requires CRS, transform, resolution, width, and height metadata for both rasters.")
    if reference_metadata is not None:
        grid = RasterGrid.from_metadata(reference_metadata)
    else:
        primary = RasterGrid.from_metadata(s1_metadata)
        crs = CRS.from_user_input(target_crs) if target_crs is not None else primary.crs
        if target_resolution <= 0:
            raise ValueError("Target resolution must be positive.")
        bounds = transform_bounds(primary.crs, crs, *primary.bounds, densify_pts=21)
        left, bottom, right, top = bounds
        width = max(1, math.ceil((right - left) / target_resolution))
        height = max(1, math.ceil((top - bottom) / target_resolution))
        grid = RasterGrid(crs, from_origin(left, top, target_resolution, target_resolution), width, height)
    return (
        align_raster_to_reference(s1_data, s1_metadata, grid),
        align_raster_to_reference(s2_data, s2_metadata, grid),
    )


def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Standard NDVI computation."""
    denominator = nir_band + red_band
    return np.divide(
        nir_band - red_band,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float32),
        where=denominator != 0,
    )


def compute_slope(dem: np.ndarray, resolution: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Compute gradient magnitude using pixel spacing in the DEM's CRS units."""
    values = np.asarray(dem, dtype=np.float32)
    x_resolution, y_resolution = resolution
    if x_resolution <= 0 or y_resolution <= 0:
        raise ValueError("DEM resolution values must be positive.")
    gy, gx = np.gradient(values, y_resolution, x_resolution)
    return np.hypot(gx, gy)


def create_quality_mask(data_dict: Dict[str, np.ndarray], nodata: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Return pixels valid in every supplied raster, with nodata kept invalid."""
    if not data_dict:
        raise ValueError("At least one raster is required to create a quality mask.")
    validity = []
    for name, array in data_dict.items():
        bands, _, masked = _as_bands(array)
        validity.append(_valid_source_pixels(bands, (nodata or {}).get(name), masked))
    if any(mask.shape != validity[0].shape for mask in validity):
        raise ValueError("Quality-mask inputs must already share the same pixel grid.")
    return np.logical_and.reduce(validity).astype(np.uint8)


def normalize_bands(
    bands: np.ndarray,
    stats: Optional[Dict[str, Any]] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize each continuous band independently using valid pixels only."""
    values, was_2d, _ = _as_bands(bands)
    output = values.astype(np.float32, copy=True)
    if valid_mask is None:
        valid_mask = np.ones(values.shape[-2:], dtype=bool)
    if valid_mask.shape != values.shape[-2:]:
        raise ValueError("Validity mask dimensions do not match the bands.")
    means = None if stats is None else np.asarray(stats.get("mean"))
    stds = None if stats is None else np.asarray(stats.get("std"))
    for index, band in enumerate(output):
        valid = valid_mask & np.isfinite(band)
        if not valid.any():
            raise ValueError(f"Band {index} has no valid pixels to normalize.")
        mean = float(means[index] if means is not None and means.ndim else means) if means is not None else float(band[valid].mean())
        std = float(stds[index] if stds is not None and stds.ndim else stds) if stds is not None else float(band[valid].std())
        if std <= 1e-8:
            std = 1.0
        band[valid] = (band[valid] - mean) / std
        band[~valid] = np.nan
    return output[0] if was_2d else output


def _tile_offsets(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    offsets = list(range(0, length - tile_size + 1, stride))
    final_offset = length - tile_size
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


def tile_image(image: np.ndarray, tile_size: int = 512, overlap: int = 64) -> List[np.ndarray]:
    """Pixel-only tiling for legacy callers; use tile_aligned_rasters for spatial data."""
    values = np.asarray(image)
    if values.ndim not in (2, 3) or tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Expected a 2D/3D image and 0 <= overlap < tile_size.")
    height, width = values.shape[-2:]
    stride = tile_size - overlap
    return [
        values[..., row:row + tile_size, col:col + tile_size]
        for row in _tile_offsets(height, tile_size, stride)
        for col in _tile_offsets(width, tile_size, stride)
    ]


def tile_aligned_rasters(
    rasters: Mapping[str, Any],
    grid_metadata: RasterGrid | Mapping[str, Any],
    *,
    tile_size: int = 512,
    overlap: int = 64,
    stride: Optional[int] = None,
    nodata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Tile aligned rasters with identical windows and per-tile affine metadata."""
    grid = grid_metadata if isinstance(grid_metadata, RasterGrid) else RasterGrid.from_metadata(grid_metadata)
    if not rasters:
        raise ValueError("At least one aligned raster is required for tiling.")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Expected 0 <= overlap < tile_size.")
    step = stride if stride is not None else tile_size - overlap
    if step <= 0 or step > tile_size:
        raise ValueError("Stride must be positive and no larger than tile_size.")

    arrays: Dict[str, np.ndarray] = {}
    raster_nodata: Dict[str, Any] = {}
    for name, raster in rasters.items():
        array = raster.data if isinstance(raster, AlignedRaster) else raster
        if isinstance(raster, AlignedRaster) and (
            raster.grid.crs != grid.crs
            or raster.grid.width != grid.width
            or raster.grid.height != grid.height
            or not raster.grid.transform.almost_equals(grid.transform)
        ):
            raise ValueError(f"Raster {name!r} is not aligned to the supplied grid.")
        bands, was_2d, _ = _as_bands(array)
        if bands.shape[-2:] != (grid.height, grid.width):
            raise ValueError(f"Raster {name!r} does not match the supplied grid dimensions.")
        arrays[name] = bands[0] if was_2d else bands
        raster_nodata[name] = (
            raster.nodata if isinstance(raster, AlignedRaster) else (nodata or {}).get(name)
        )

    tiles = []
    for row in _tile_offsets(grid.height, tile_size, step):
        for col in _tile_offsets(grid.width, tile_size, step):
            height = min(tile_size, grid.height - row)
            width = min(tile_size, grid.width - col)
            window = Window(col, row, width, height)
            tile_grid = RasterGrid(
                grid.crs,
                window_transform(window, grid.transform),
                width,
                height,
            )
            tiles.append({
                "rasters": {
                    name: array[..., row:row + height, col:col + width]
                    for name, array in arrays.items()
                },
                "window": {"row_off": row, "col_off": col, "height": height, "width": width},
                "metadata": tile_grid.to_metadata(),
                "raster_metadata": {
                    name: tile_grid.to_metadata(value)
                    for name, value in raster_nodata.items()
                },
            })
    return tiles


def ensure_channel_dim(array: np.ndarray, target_channels: int, fill_value: float = 0.0) -> np.ndarray:
    """Pad or trim a raster stack to a target channel count."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.shape[0] < target_channels:
        pad = np.full((target_channels - arr.shape[0], *arr.shape[1:]), fill_value, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > target_channels:
        arr = arr[:target_channels, ...]
    return arr


def normalize_raster(array: np.ndarray, scale: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """Legacy uniform scale helper; use normalize_feature for typed spatial features."""
    if scale <= 0:
        raise ValueError("Raster scale must be positive.")
    return np.asarray(array, dtype=np.float32) / max(scale, eps)


def build_quality_mask(mask: np.ndarray, nodata: Any = None) -> np.ndarray:
    """Return one for valid mask pixels and zero for nodata/non-finite pixels."""
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[0]
    valid = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else np.ones(values.shape, bool)
    valid &= ~_is_nodata(values, nodata)
    return valid.astype(np.uint8)


def cross_modality_align(
    s1_tiles: np.ndarray,
    s2_tiles: np.ndarray,
    dem_tile: np.ndarray,
    slope_tile: np.ndarray,
    ndvi_tile: np.ndarray,
    quality_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack already aligned modalities, rejecting shape mismatches instead of hiding them."""
    arrays = [np.asarray(layer) for layer in (s1_tiles, s2_tiles, dem_tile, slope_tile, ndvi_tile)]
    expected_shape = arrays[0].shape[-2:]
    if any(array.shape[-2:] != expected_shape for array in arrays[1:]):
        raise ValueError("All modalities must be reprojected to the same grid before stacking.")
    if np.asarray(quality_mask).shape != expected_shape:
        raise ValueError("Quality mask dimensions must match the aligned modalities.")
    stacked = np.concatenate([array if array.ndim == 3 else array[np.newaxis, ...] for array in arrays], axis=0)
    return stacked, quality_mask


def prepare_hazard_tensor(raw_array: np.ndarray, target_channels: int, scale: float, fill_value: float = 0.0) -> np.ndarray:
    """Legacy tensor preparation; new aligned paths should use normalize_feature."""
    arr = ensure_channel_dim(raw_array, target_channels, fill_value=fill_value)
    arr = normalize_raster(arr, scale=scale)
    return arr.astype(np.float32)
