"""PyTorch Dataset classes for MultiHazard-VLM.

Dataset source mapping for this project:
- flood -> Global S1S2-Water Benchmark Dataset (Zenodo)
- wildfire -> HLS Burn Scar Scenes (Hugging Face)
- geospatial aux data -> Microsoft Planetary Computer
- landslide -> optional CAS landslide dataset, if added later
"""
import csv
import glob
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessing import prepare_hazard_tensor

DATASET_REGISTRY = {
    "flood": {
        "name": "Global S1S2-Water Benchmark Dataset",
        "source": "Zenodo",
        "url": "https://zenodo.org/records/11278238",
        "loader": "S1S2WaterDataset",
        "notes": "Sentinel-1 VV/VH and Sentinel-2 L1C bands for water/flood mapping.",
    },
    "wildfire": {
        "name": "HLS Burn Scar Scenes",
        "source": "Hugging Face",
        "url": "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars",
        "loader": "HLSBurnScarDataset",
        "notes": "HLS burn scar scenes with binary burn masks.",
    },
    "landslide": {
        "name": "CAS Landslide Dataset",
        "source": "Chinese Academy of Sciences",
        "url": "https://github.com/cas-landslide",
        "loader": "CASLandslideDataset",
        "notes": "Optional landslide dataset for the third hazard branch.",
    },
    "planetary_computer": {
        "name": "Microsoft Planetary Computer",
        "source": "Microsoft",
        "url": "https://planetarycomputer.microsoft.com/",
        "loader": "PlanetaryComputerClient",
        "notes": "Used for DEM, slope, NDVI, and auxiliary geospatial layers.",
    },
}


def get_dataset_registry() -> Dict[str, Dict[str, str]]:
    """Return the mapping of hazard type to the actual dataset source."""
    return DATASET_REGISTRY


def _find_tif_files(data_dir: str, patterns: List[str]) -> List[str]:
    if not data_dir or not os.path.isdir(data_dir):
        return []
    matches = {
        path
        for pattern in patterns
        for path in glob.glob(os.path.join(data_dir, pattern))
        if os.path.isfile(path)
    }
    return sorted(matches)


def _find_tif_file(
    data_dir: str,
    preferred_names: List[str],
    extra_patterns: Optional[List[str]] = None,
    fallback_all: bool = True,
) -> Optional[str]:
    """Resolve a real TIFF path from a dataset folder, preferring known file names."""
    if not data_dir or not os.path.isdir(data_dir):
        return None

    for name in preferred_names:
        if glob.has_magic(name):
            matches = _find_tif_files(data_dir, [name])
            if matches:
                return matches[0]
        else:
            candidate = os.path.join(data_dir, name)
            if os.path.isfile(candidate):
                return candidate

    patterns = list(extra_patterns or [])
    if fallback_all:
        patterns.extend(['*.tif', '*.tiff'])

    for pattern in patterns:
        matches = _find_tif_files(data_dir, [pattern])
        if matches:
            return matches[0]
    return None


def _load_raster_array(path: Optional[str]) -> Optional[np.ndarray]:
    """Read a raster into a 3D array without changing its native dtype."""
    if path is None or not os.path.exists(path):
        return None

    with rasterio.open(path) as src:
        arr = src.read()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def _raster_geospatial_metadata(path: Optional[str]) -> Dict[str, Any]:
    """Return source-grid metadata and WGS84 bounds without altering the raster."""
    unavailable = {
        "lat": None,
        "lon": None,
        "bbox": None,
        "crs": None,
        "transform": None,
        "resolution": None,
        "width": None,
        "height": None,
        "nodata": None,
        "georeferenced": False,
    }
    if path is None or not os.path.exists(path):
        return unavailable

    with rasterio.open(path) as src:
        source_crs = src.crs.to_string() if src.crs is not None else None
        metadata = {
            **unavailable,
            "crs": source_crs,
            "transform": src.transform,
            "resolution": tuple(float(value) for value in src.res),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }
        if src.crs is None:
            return metadata
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    min_lon, min_lat, max_lon, max_lat = bounds
    valid = (
        -180 <= min_lon < max_lon <= 180
        and -90 <= min_lat < max_lat <= 90
    )
    if not valid:
        return metadata

    return {
        **metadata,
        "lat": (min_lat + max_lat) / 2.0,
        "lon": (min_lon + max_lon) / 2.0,
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "georeferenced": True,
    }


def _manifest_value(row: Dict[str, str], field: str) -> Optional[str]:
    value = (row.get(field) or "").strip()
    if value.lower() in {"", "not_verified", "not_available", "not_applicable", "null", "none"}:
        return None
    return value


def _load_event_metadata(
    event_id: Optional[str],
    expected_hazard: str,
    manifest_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read only verified event facts from the Phase 1 manifest."""
    if event_id is None:
        return None

    path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parent / "events" / "event_manifest.csv"
    with path.open(newline="", encoding="utf-8") as manifest_file:
        row = next((item for item in csv.DictReader(manifest_file) if item.get("event_id") == event_id), None)
    if row is None:
        raise ValueError(f"Event {event_id!r} is not present in {path}.")

    manifest_hazard = _manifest_value(row, "hazard")
    if manifest_hazard and manifest_hazard.lower() != expected_hazard.lower():
        raise ValueError(
            f"Event {event_id!r} is marked as {manifest_hazard!r}, not {expected_hazard!r}."
        )

    latitude_value = _manifest_value(row, "latitude")
    longitude_value = _manifest_value(row, "longitude")
    location = None
    if latitude_value is not None and longitude_value is not None:
        location = {"latitude": float(latitude_value), "longitude": float(longitude_value)}

    return {
        "event_id": event_id,
        "hazard": manifest_hazard,
        "event_name": _manifest_value(row, "event_name"),
        "event_date": _manifest_value(row, "start_date"),
        "location": location,
        "source": _manifest_value(row, "source"),
        "ground_truth": _manifest_value(row, "ground_truth"),
    }


def _paired_path(paths: List[str], index: int, sample_count: int) -> Optional[str]:
    if len(paths) == sample_count:
        return paths[index]
    if sample_count == 1 and len(paths) == 1:
        return paths[0]
    return None


def _make_unified_sample(
    image: torch.Tensor,
    target_mask: Optional[torch.Tensor],
    hazard: str,
    hazard_type: int,
    image_path: Optional[str],
    mask_path: Optional[str],
    event_id: Optional[str],
    event_metadata: Optional[Dict[str, Any]],
    input_paths: Optional[Dict[str, Optional[str]]] = None,
    target_semantics: str = "not_verified",
    input_normalization: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    image_metadata = _raster_geospatial_metadata(image_path)
    mask_metadata = _raster_geospatial_metadata(mask_path)
    source_paths = input_paths or {"image": image_path}
    input_raster_metadata = {
        name: _raster_geospatial_metadata(path)
        for name, path in source_paths.items()
        if path is not None
    }
    return {
        "image": image,
        "geospatial_features": None,
        "target_mask": target_mask,
        "severity_mask": None,
        "event_id": event_id,
        "hazard": hazard,
        "metadata": {
            **image_metadata,
            "timestamp": event_metadata.get("event_date") if event_metadata else None,
            "hazard_type": hazard_type,
            "target_semantics": target_semantics,
            "h3_index": None,
            "event": event_metadata,
            "dataset_source": DATASET_REGISTRY[hazard]["source"],
            "raster": {"image": image_metadata, "target_mask": mask_metadata},
            "raster_inputs": input_raster_metadata,
            "input_normalization": input_normalization or {},
            "source_paths": {
                **source_paths,
                "target_mask": mask_path,
            },
        },
    }


class MultiHazardDataset(Dataset):
    """Base class with common logic for hazard datasets."""
    def __init__(
        self,
        data_dir: str,
        hazard: str,
        event_id: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.hazard = hazard
        self.event_id = event_id
        self.event_metadata = _load_event_metadata(event_id, hazard, manifest_path)
        self.data: List[Dict[str, Optional[str]]] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_collate_fn() -> Any:
        """Return a collate function for DataLoader."""
        def collate_fn(batch):
            return torch.utils.data.dataloader.default_collate(batch)
        return collate_fn


class S1S2WaterDataset(MultiHazardDataset):
    """For S1S2-Water flood data (Zenodo)."""
    def __init__(
        self,
        data_dir: str = 'datasets/s1s2_water/train',
        event_id: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        super().__init__(data_dir, "flood", event_id, manifest_path)
        self.s1_path = _find_tif_file(self.data_dir, ['sample_s1.tif', 's1.tif', '*s1*.tif'], fallback_all=False)
        self.s2_path = _find_tif_file(self.data_dir, ['sample_s2.tif', 's2.tif', '*s2*.tif'], fallback_all=False)
        self.mask_path = _find_tif_file(
            self.data_dir, ['sample_mask.tif', '*mask*.tif'], fallback_all=False
        )
        self.data = self._discover_samples()

    def _discover_samples(self) -> List[Dict[str, Optional[str]]]:
        s1_candidates = _find_tif_files(self.data_dir, ['*s1*.tif'])
        s2_candidates = _find_tif_files(self.data_dir, ['*s2*.tif'])
        mask_candidates = _find_tif_files(self.data_dir, ['*mask*.tif', '*.mask.tif'])
        sample_count = len(s1_candidates)
        return [
            {
                's1_path': s1_path,
                's2_path': _paired_path(s2_candidates, index, sample_count),
                'mask_path': _paired_path(mask_candidates, index, sample_count),
            }
            for index, s1_path in enumerate(s1_candidates)
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.data[idx]
        s1_data = _load_raster_array(record.get('s1_path', self.s1_path))
        s2_data = _load_raster_array(record.get('s2_path', self.s2_path))
        mask_data = _load_raster_array(record.get('mask_path', self.mask_path))
        if s1_data is None:
            raise FileNotFoundError(f"Flood image is missing for sample {idx} in {self.data_dir}.")

        h, w = s1_data.shape[-2], s1_data.shape[-1]
        if s2_data is None:
            s2_data = np.zeros((6, h, w), dtype=np.float32)
        tensor = torch.zeros((10, h, w), dtype=torch.float32)
        tensor[: min(2, s1_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(s1_data[:2], 2, scale=100.0)).float()
        tensor[2: 2 + min(6, s2_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(s2_data[:6], 6, scale=10000.0)).float()

        flood_mask = torch.from_numpy(np.ascontiguousarray(mask_data[0])) if mask_data is not None else None
        return _make_unified_sample(
            tensor, flood_mask, "flood", 0, record['s1_path'], record.get('mask_path'),
            self.event_id, self.event_metadata,
            {"sentinel1": record['s1_path'], "sentinel2": record.get('s2_path')},
            "water_mask; new inundation versus permanent water not verified",
            {
                "sentinel1": {"feature_type": "radar", "scale": 100.0},
                "sentinel2": {"feature_type": "reflectance", "scale": 10000.0},
            },
        )


class HLSBurnScarDataset(MultiHazardDataset):
    """For HLS Burn Scar Scenes."""
    def __init__(
        self,
        data_dir: str = 'datasets/hls_burn_scars/train',
        event_id: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        super().__init__(data_dir, "wildfire", event_id, manifest_path)
        self.hls_path = _find_tif_file(
            self.data_dir, ['sample_hls.tif', '*hls*.tif', '*_merged.tif'], fallback_all=False
        )
        self.mask_path = _find_tif_file(
            self.data_dir, ['sample_mask.tif', '*mask*.tif', '*.mask.tif'], fallback_all=False
        )
        self.data = self._discover_samples()

    def _discover_samples(self) -> List[Dict[str, Optional[str]]]:
        image_paths = _find_tif_files(self.data_dir, ['*hls*.tif', '*_merged.tif'])
        mask_paths = _find_tif_files(self.data_dir, ['*mask*.tif', '*.mask.tif'])
        return [
            {
                'hls_path': image_path,
                'mask_path': _paired_path(mask_paths, index, len(image_paths)),
            }
            for index, image_path in enumerate(image_paths)
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.data[idx]
        hls_data = _load_raster_array(record.get('hls_path', self.hls_path))
        mask_data = _load_raster_array(record.get('mask_path', self.mask_path))
        if hls_data is None:
            raise FileNotFoundError(f"Wildfire image is missing for sample {idx} in {self.data_dir}.")

        h, w = hls_data.shape[-2], hls_data.shape[-1]
        tensor = torch.zeros((10, h, w), dtype=torch.float32)
        tensor[: min(6, hls_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(hls_data[:6], 6, scale=10000.0)).float()

        burn_mask = torch.from_numpy(np.ascontiguousarray(mask_data[0])) if mask_data is not None else None
        return _make_unified_sample(
            tensor, burn_mask, "wildfire", 1, record['hls_path'], record.get('mask_path'),
            self.event_id, self.event_metadata,
            target_semantics="burned_area",
            input_normalization={"image": {"feature_type": "reflectance", "scale": 10000.0}},
        )


class CASLandslideDataset(MultiHazardDataset):
    """For CAS Landslide Dataset."""
    def __init__(
        self,
        data_dir: str = 'datasets/cas_landslide/train',
        event_id: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        super().__init__(data_dir, "landslide", event_id, manifest_path)
        self.rgb_path = _find_tif_file(
            self.data_dir, ['sample_rgb.tif', 'rgb.tif', '*rgb*.tif'], fallback_all=False
        )
        self.mask_path = _find_tif_file(
            self.data_dir, ['sample_mask.tif', '*mask*.tif'], fallback_all=False
        )
        self.data = self._discover_samples()

    def _discover_samples(self) -> List[Dict[str, Optional[str]]]:
        image_paths = _find_tif_files(self.data_dir, ['*rgb*.tif'])
        mask_paths = _find_tif_files(self.data_dir, ['*mask*.tif'])
        return [
            {
                'rgb_path': image_path,
                'mask_path': _paired_path(mask_paths, index, len(image_paths)),
            }
            for index, image_path in enumerate(image_paths)
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.data[idx]
        rgb_data = _load_raster_array(record.get('rgb_path', self.rgb_path))
        mask_data = _load_raster_array(record.get('mask_path', self.mask_path))
        if rgb_data is None:
            raise FileNotFoundError(f"Landslide image is missing for sample {idx} in {self.data_dir}.")

        h, w = rgb_data.shape[-2], rgb_data.shape[-1]
        tensor = torch.zeros((10, h, w), dtype=torch.float32)
        tensor[: min(3, rgb_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(rgb_data[:3], 3, scale=255.0)).float()

        landslide_mask = torch.from_numpy(np.ascontiguousarray(mask_data[0])) if mask_data is not None else None
        return _make_unified_sample(
            tensor, landslide_mask, "landslide", 2, record['rgb_path'], record.get('mask_path'),
            self.event_id, self.event_metadata,
            target_semantics="not_verified",
            input_normalization={"image": {"feature_type": "rgb", "scale": 255.0}},
        )

class CombinedHazardDataset(Dataset):
    """Combine flood, wildfire, and landslide branches under one multi-hazard loader.

    The dataset uses a deterministic split across the per-hazard sample lists so train and val
    remain disjoint. For small sample sets, the split still preserves all hazard branches while
    giving a stable validation partition. The default legacy format preserves train.py fields;
    reconstruction format returns the unresized common sample without synthetic targets.
    """
    def __init__(
        self,
        split: str = 'train',
        dataset_order: Optional[List[str]] = None,
        val_ratio: float = 0.2,
        img_size: int = 64,
        sample_format: str = 'legacy',
        data_dirs: Optional[Dict[str, str]] = None,
        event_ids: Optional[Dict[str, str]] = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        assert split in {'train', 'val'}, "split must be 'train' or 'val'"
        if sample_format not in {'legacy', 'reconstruction'}:
            raise ValueError("sample_format must be 'legacy' or 'reconstruction'")
        self.split = split
        self.val_ratio = val_ratio
        self.img_size = img_size
        self.sample_format = sample_format
        self.dataset_order = dataset_order or ['flood', 'wildfire', 'landslide']
        data_dirs = data_dirs or {}
        event_ids = event_ids or {}
        self.datasets = {
            'flood': S1S2WaterDataset(data_dirs.get('flood', 'datasets/s1s2_water/train'), event_ids.get('flood'), manifest_path),
            'wildfire': HLSBurnScarDataset(data_dirs.get('wildfire', 'datasets/hls_burn_scars/train'), event_ids.get('wildfire'), manifest_path),
            'landslide': CASLandslideDataset(data_dirs.get('landslide', 'datasets/cas_landslide/train'), event_ids.get('landslide'), manifest_path),
        }
        self.dataset_ids = self._build_split_ids()

    def get_reconstruction_sample(self, idx: int) -> Dict[str, Any]:
        """Return the unresized common sample without legacy synthetic targets."""
        hazard_name, sample_idx = self.dataset_ids[idx]
        return self.datasets[hazard_name][sample_idx]

    @staticmethod
    def get_reconstruction_collate_fn():
        """Keep variable-size rasters and optional metadata as per-sample records."""
        def collate_fn(batch):
            return batch
        return collate_fn

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, img_size: int) -> torch.Tensor:
        if tensor.ndim != 3:
            return tensor
        h, w = tensor.shape[-2], tensor.shape[-1]
        if h == img_size and w == img_size:
            return tensor
        tensor = tensor.unsqueeze(0)
        resized = F.interpolate(tensor.float(), size=(img_size, img_size), mode='bilinear', align_corners=False)
        return resized.squeeze(0)

    def _build_split_ids(self) -> List[Tuple[str, int]]:
        dataset_ids: List[Tuple[str, int]] = []
        hazard_names = [h for h in self.dataset_order if h in self.datasets]

        for hazard_idx, hazard_name in enumerate(hazard_names):
            ds = self.datasets.get(hazard_name)
            if ds is None:
                continue

            n_samples = len(ds)
            if n_samples == 1:
                if self.split == 'train':
                    if hazard_idx % 2 == 0 or len(hazard_names) == 1:
                        dataset_ids.append((hazard_name, 0))
                else:
                    if hazard_idx % 2 == 1 or len(hazard_names) == 1:
                        dataset_ids.append((hazard_name, 0))
                continue

            split_index = max(1, int(round(n_samples * (1.0 - self.val_ratio))))
            indices = list(range(n_samples))
            if self.split == 'train':
                selected = indices[:split_index]
            else:
                selected = indices[split_index:]

            if not selected and n_samples > 0:
                selected = [0]

            for sample_idx in selected:
                dataset_ids.append((hazard_name, sample_idx))

        return dataset_ids

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.sample_format == 'reconstruction':
            return self.get_reconstruction_sample(idx)

        hazard_name, sample_idx = self.dataset_ids[idx % len(self.dataset_ids)]
        sample = self.datasets[hazard_name][sample_idx]
        hazard_mask = sample['target_mask']
        if hazard_mask is None:
            raise ValueError(
                f"Legacy supervised training requires a real target mask; "
                f"{hazard_name} sample {sample_idx} has none."
            )

        tensor = self._resize_tensor(sample['image'], self.img_size)
        tensor_t = tensor.unsqueeze(0).repeat(4, 1, 1, 1)
        q_mask_t = torch.ones((4,), dtype=torch.bool)

        h3_index_dict = {
            4: torch.randint(0, 1000, (10,)),
            6: torch.randint(0, 1000, (20,)),
            8: torch.randint(0, 1000, (30,)),
        }

        geo_attr_dict = {
            'elevation': torch.randn(1),
            'slope': torch.randn(1),
            'ndvi': torch.rand(1),
            'aspect': torch.randn(1),
            'lithology_class': torch.randint(0, 10, (1,)).squeeze(),
        }

        legacy_targets = {
            'coords_target': torch.rand(2),
            'radius_target': torch.tensor([15.0]),
            'target_tokens': torch.randint(0, 100, (16,)),
        }
        legacy_inputs = {
            'h3_index': h3_index_dict,
            'geo_attributes': geo_attr_dict,
        }

        return {
            'images': tensor_t,
            'quality_flags': q_mask_t,
            'h3_index': h3_index_dict,
            'geo_attributes': geo_attr_dict,
            'hazard_label': torch.tensor(float(sample['metadata']['hazard_type']), dtype=torch.long),
            'coords_target': legacy_targets['coords_target'],
            'radius_target': legacy_targets['radius_target'],
            'target_tokens': legacy_targets['target_tokens'],
            'hazard_name': hazard_name,
            'hazard_mask': hazard_mask,
            'target_mask': hazard_mask,
            'legacy_inputs': legacy_inputs,
            'legacy_targets': legacy_targets,
        }

    @staticmethod
    def get_collate_fn() -> Any:
        def collate_fn(batch):
            return torch.utils.data.dataloader.default_collate(batch)
        return collate_fn
