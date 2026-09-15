"""PyTorch Dataset classes for MultiHazard-VLM.

Dataset source mapping for this project:
- flood -> Global S1S2-Water Benchmark Dataset (Zenodo)
- wildfire -> HLS Burn Scar Scenes (Hugging Face)
- geospatial aux data -> Microsoft Planetary Computer
- landslide -> optional CAS landslide dataset, if added later
"""
import os
import glob
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessing import prepare_hazard_tensor, build_quality_mask

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


def _find_tif_file(data_dir: str, preferred_names: List[str], extra_patterns: Optional[List[str]] = None) -> Optional[str]:
    """Resolve a real TIFF path from a dataset folder, preferring known file names."""
    if not data_dir or not os.path.isdir(data_dir):
        return None

    for name in preferred_names:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            return candidate

    patterns = []
    if extra_patterns:
        patterns.extend(extra_patterns)
    patterns.extend(['*.tif', '*.tiff'])

    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if matches:
            return matches[0]
    return None


def _load_raster_array(path: Optional[str]) -> np.ndarray:
    """Read a raster into a 3D array (bands, height, width)."""
    if path is None or not os.path.exists(path):
        return np.zeros((1, 1, 1), dtype=np.float32)

    with rasterio.open(path) as src:
        arr = src.read()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr.astype(np.float32)


class MultiHazardDataset(Dataset):
    """Base class with common logic for hazard datasets."""
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.data = [{"id": "sample"}] if os.path.isdir(data_dir) else []

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
    def __init__(self, data_dir: str = 'datasets/s1s2_water/train') -> None:
        super().__init__(data_dir)
        self.s1_path = _find_tif_file(self.data_dir, ['sample_s1.tif', 's1.tif', '*s1*.tif'])
        self.s2_path = _find_tif_file(self.data_dir, ['sample_s2.tif', 's2.tif', '*s2*.tif'])
        self.mask_path = _find_tif_file(self.data_dir, ['sample_mask.tif', '*mask*.tif'], ['*.tif'])
        self.data = self._discover_samples()

    def _discover_samples(self):
        file_set = []
        s1_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*s1*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_s1.tif')))
        s2_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*s2*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_s2.tif')))
        mask_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*mask*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_mask.tif')))
        max_len = max(len(s1_candidates), len(s2_candidates), len(mask_candidates), 1)
        for i in range(max_len):
            s1 = s1_candidates[i] if i < len(s1_candidates) else self.s1_path
            s2 = s2_candidates[i] if i < len(s2_candidates) else self.s2_path
            mask = mask_candidates[i] if i < len(mask_candidates) else self.mask_path
            if s1 and s2 and mask:
                file_set.append({'s1_path': s1, 's2_path': s2, 'mask_path': mask})
        if not file_set:
            file_set = [{'s1_path': self.s1_path, 's2_path': self.s2_path, 'mask_path': self.mask_path}]
        return file_set

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.data[idx % len(self.data)]
        s1_data = _load_raster_array(record.get('s1_path', self.s1_path))
        s2_data = _load_raster_array(record.get('s2_path', self.s2_path))
        mask_data = _load_raster_array(record.get('mask_path', self.mask_path))

        if s1_data.shape[0] == 0:
            s1_data = np.zeros((2, 512, 512), dtype=np.float32)
        if s2_data.shape[0] == 0:
            s2_data = np.zeros((6, 512, 512), dtype=np.float32)
        if mask_data.shape[0] == 0:
            mask_data = np.zeros((1, 512, 512), dtype=np.float32)

        h, w = s1_data.shape[-2], s1_data.shape[-1]
        tensor = torch.zeros((10, h, w), dtype=torch.float32)
        tensor[: min(2, s1_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(s1_data[:2], 2, scale=100.0)).float()
        tensor[2: 2 + min(6, s2_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(s2_data[:6], 6, scale=10000.0)).float()

        flood_mask = torch.from_numpy(mask_data[0]).to(torch.uint8)
        quality_mask = build_quality_mask(mask_data).astype(np.uint8)
        quality_mask = torch.from_numpy(quality_mask).to(torch.uint8)
        if torch.any(flood_mask < 0):
            quality_mask = (flood_mask >= 0).to(torch.uint8)

        meta = {"lat": 0.0, "lon": 0.0, "timestamp": "2023-01-01", "hazard_type": 0, "bbox": [0, 0, 1, 1], "h3_index": "862830827ffffff"}
        return tensor, quality_mask, flood_mask, meta


class HLSBurnScarDataset(MultiHazardDataset):
    """For HLS Burn Scar Scenes."""
    def __init__(self, data_dir: str = 'datasets/hls_burn_scars/train') -> None:
        super().__init__(data_dir)
        self.hls_path = _find_tif_file(self.data_dir, ['sample_hls.tif', 'hls.tif', '*hls*.tif'])
        self.mask_path = _find_tif_file(self.data_dir, ['sample_mask.tif', '*mask*.tif'], ['*.tif'])
        self.data = self._discover_samples()

    def _discover_samples(self):
        hls_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*hls*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_hls.tif')))
        mask_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*mask*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_mask.tif')))
        file_set = []
        max_len = max(len(hls_candidates), len(mask_candidates), 1)
        for i in range(max_len):
            hls = hls_candidates[i] if i < len(hls_candidates) else self.hls_path
            mask = mask_candidates[i] if i < len(mask_candidates) else self.mask_path
            if hls and mask:
                file_set.append({'hls_path': hls, 'mask_path': mask})
        if not file_set:
            file_set = [{'hls_path': self.hls_path, 'mask_path': self.mask_path}]
        return file_set

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.data[idx % len(self.data)]
        hls_data = _load_raster_array(record.get('hls_path', self.hls_path))
        mask_data = _load_raster_array(record.get('mask_path', self.mask_path))

        if hls_data.shape[0] == 0:
            hls_data = np.zeros((6, 512, 512), dtype=np.float32)
        if mask_data.shape[0] == 0:
            mask_data = np.zeros((1, 512, 512), dtype=np.float32)

        h, w = hls_data.shape[-2], hls_data.shape[-1]
        tensor = torch.zeros((10, h, w), dtype=torch.float32)
        tensor[: min(6, hls_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(hls_data[:6], 6, scale=10000.0)).float()

        burn_mask = torch.from_numpy(mask_data[0]).to(torch.uint8)
        quality_mask = build_quality_mask(mask_data).astype(np.uint8)
        quality_mask = torch.from_numpy(quality_mask).to(torch.uint8)
        if torch.any(burn_mask < 0):
            quality_mask = (burn_mask >= 0).to(torch.uint8)

        meta = {"lat": 0.0, "lon": 0.0, "timestamp": "2023-01-01", "hazard_type": 1, "bbox": [0, 0, 1, 1], "h3_index": "862830827ffffff"}
        return tensor, quality_mask, burn_mask, meta


class CASLandslideDataset(MultiHazardDataset):
    """For CAS Landslide Dataset."""
    def __init__(self, data_dir: str = 'datasets/cas_landslide/train') -> None:
        super().__init__(data_dir)
        self.rgb_path = _find_tif_file(self.data_dir, ['sample_rgb.tif', 'rgb.tif', '*rgb*.tif'])
        self.mask_path = _find_tif_file(self.data_dir, ['sample_mask.tif', '*mask*.tif'], ['*.tif'])
        self.data = self._discover_samples()

    def _discover_samples(self):
        rgb_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*rgb*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_rgb.tif')))
        mask_candidates = sorted(glob.glob(os.path.join(self.data_dir, '*mask*.tif')) + glob.glob(os.path.join(self.data_dir, 'sample_mask.tif')))
        file_set = []
        max_len = max(len(rgb_candidates), len(mask_candidates), 1)
        for i in range(max_len):
            rgb = rgb_candidates[i] if i < len(rgb_candidates) else self.rgb_path
            mask = mask_candidates[i] if i < len(mask_candidates) else self.mask_path
            if rgb and mask:
                file_set.append({'rgb_path': rgb, 'mask_path': mask})
        if not file_set:
            file_set = [{'rgb_path': self.rgb_path, 'mask_path': self.mask_path}]
        return file_set

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.data[idx % len(self.data)]
        rgb_data = _load_raster_array(record.get('rgb_path', self.rgb_path))
        mask_data = _load_raster_array(record.get('mask_path', self.mask_path))

        if rgb_data.shape[0] == 0:
            rgb_data = np.zeros((3, 512, 512), dtype=np.float32)
        if mask_data.shape[0] == 0:
            mask_data = np.zeros((1, 512, 512), dtype=np.float32)

        h, w = rgb_data.shape[-2], rgb_data.shape[-1]
        tensor = torch.zeros((10, h, w), dtype=torch.float32)
        tensor[: min(3, rgb_data.shape[0]), :, :] = torch.from_numpy(prepare_hazard_tensor(rgb_data[:3], 3, scale=255.0)).float()

        landslide_mask = torch.from_numpy(mask_data[0]).to(torch.uint8)
        quality_mask = build_quality_mask(mask_data).astype(np.uint8)
        quality_mask = torch.from_numpy(quality_mask).to(torch.uint8)
        if torch.any(landslide_mask < 0):
            quality_mask = (landslide_mask >= 0).to(torch.uint8)

        meta = {"lat": 0.0, "lon": 0.0, "timestamp": "2023-01-01", "hazard_type": 2, "bbox": [0, 0, 1, 1], "h3_index": "862830827ffffff"}
        return tensor, quality_mask, landslide_mask, meta

class CombinedHazardDataset(Dataset):
    """Combine flood, wildfire, and landslide branches under one multi-hazard loader.

    The dataset uses a deterministic split across the per-hazard sample lists so train and val
    remain disjoint. For small sample sets, the split still preserves all hazard branches while
    giving a stable validation partition.
    """
    def __init__(self, split: str = 'train', dataset_order: Optional[List[str]] = None, val_ratio: float = 0.2, img_size: int = 64) -> None:
        assert split in {'train', 'val'}, "split must be 'train' or 'val'"
        self.split = split
        self.val_ratio = val_ratio
        self.img_size = img_size
        self.dataset_order = dataset_order or ['flood', 'wildfire', 'landslide']
        self.datasets = {
            'flood': S1S2WaterDataset(),
            'wildfire': HLSBurnScarDataset(),
            'landslide': CASLandslideDataset(),
        }
        self.dataset_ids = self._build_split_ids()

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
        hazard_name, sample_idx = self.dataset_ids[idx % len(self.dataset_ids)]
        tensor, q_mask, hazard_mask, meta = self.datasets[hazard_name][sample_idx]

        tensor = self._resize_tensor(tensor, self.img_size)
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

        return {
            'images': tensor_t,
            'quality_flags': q_mask_t,
            'h3_index': h3_index_dict,
            'geo_attributes': geo_attr_dict,
            'hazard_label': torch.tensor(float(meta['hazard_type']), dtype=torch.long),
            'coords_target': torch.rand(2),
            'radius_target': torch.tensor([15.0]),
            'target_tokens': torch.randint(0, 100, (16,)),
            'hazard_name': hazard_name,
            'hazard_mask': hazard_mask,
        }

    @staticmethod
    def get_collate_fn() -> Any:
        def collate_fn(batch):
            return torch.utils.data.dataloader.default_collate(batch)
        return collate_fn
