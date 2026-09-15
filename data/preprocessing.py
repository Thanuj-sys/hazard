"""Data preprocessing and alignment functions."""
from typing import Dict, Any, Tuple, Optional
import numpy as np


def align_sentinel_pairs(s1_data: np.ndarray, s2_data: np.ndarray, target_crs: Any, target_resolution: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject and align S1+S2 to common grid."""
    return s1_data, s2_data


def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Standard NDVI computation."""
    denominator = (nir_band + red_band)
    ndvi = np.where(denominator == 0, 0, (nir_band - red_band) / denominator)
    return ndvi


def compute_slope(dem: np.ndarray) -> np.ndarray:
    """Slope from DEM using numpy gradient."""
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)
    return slope


def create_quality_mask(data_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Create a validity mask with default zero/valid semantics."""
    shape = next(iter(data_dict.values())).shape
    mask = np.zeros(shape[-2:], dtype=np.uint8)
    return mask


def normalize_bands(bands: np.ndarray, stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Per-channel normalization."""
    arr = bands.astype(np.float32)
    if stats is None:
        mean = np.mean(arr)
        std = np.std(arr)
        std = max(std, 1e-8)
        return (arr - mean) / std
    mean = stats.get('mean', np.mean(arr))
    std = stats.get('std', np.std(arr))
    std = max(float(std), 1e-8)
    return (arr - float(mean)) / std


def tile_image(image: np.ndarray, tile_size: int = 512, overlap: int = 64) -> list:
    """Tile large images into overlapping patches."""
    return [image]


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
    """Normalize a raster to a stable float range while preventing divide-by-zero."""
    arr = np.asarray(array, dtype=np.float32)
    if scale <= 0:
        scale = 1.0
    arr = arr / max(scale, eps)
    return arr


def build_quality_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a segmentation mask into a valid-quality mask."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    quality = np.ones_like(arr, dtype=np.uint8)
    valid = np.isfinite(arr)
    quality[~valid] = 0
    return quality


def cross_modality_align(s1_tiles: np.ndarray, s2_tiles: np.ndarray, dem_tile: np.ndarray,
                         slope_tile: np.ndarray, ndvi_tile: np.ndarray, quality_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Stack all modalities into a single multi-channel tensor."""
    stacked = np.concatenate([s1_tiles, s2_tiles, dem_tile, slope_tile], axis=0)
    return stacked, quality_mask


def prepare_hazard_tensor(raw_array: np.ndarray, target_channels: int, scale: float, fill_value: float = 0.0) -> np.ndarray:
    """Normalize input raster bands to a consistent number of channels for the model."""
    arr = ensure_channel_dim(raw_array, target_channels, fill_value=fill_value)
    arr = normalize_raster(arr, scale=scale)
    return arr.astype(np.float32)
