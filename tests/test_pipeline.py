"""Smoke test for the MultiHazardVLM pipeline.

Verifies that:
1. All modules import correctly
2. Forward pass runs with dummy data of the correct shapes
3. Output keys and shapes match expectations
"""
import torch
import sys
import os
import asyncio
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multihazard_vlm import MultiHazardVLM
from data.datasets import (
    CASLandslideDataset,
    CombinedHazardDataset,
    HLSBurnScarDataset,
    S1S2WaterDataset,
)

UNIFIED_SAMPLE_KEYS = {
    'image',
    'geospatial_features',
    'target_mask',
    'severity_mask',
    'event_id',
    'hazard',
    'metadata',
}


def test_dataset_loaders_read_real_sample_tiffs():
    loaders = [
        ('flood', S1S2WaterDataset),
        ('wildfire', HLSBurnScarDataset),
        ('landslide', CASLandslideDataset),
    ]
    for hazard, loader_type in loaders:
        dataset = loader_type()
        sample = dataset[0]
        assert set(sample) == UNIFIED_SAMPLE_KEYS
        assert sample['hazard'] == hazard
        assert sample['image'].shape == (10, 512, 512)
        assert sample['image'].abs().max().item() > 1e-6
        assert sample['target_mask'] is not None
        assert sample['target_mask'].shape == (512, 512)
        assert sample['severity_mask'] is None
        assert sample['event_id'] is None
        if hazard == 'flood':
            assert sample['metadata']['target_semantics'].startswith('water_mask;')
        elif hazard == 'wildfire':
            assert sample['metadata']['target_semantics'] == 'burned_area'
        else:
            assert sample['metadata']['target_semantics'] == 'not_verified'
            assert 'runout_mask' not in sample

        mask_path = dataset.data[0]['mask_path']
        with rasterio.open(mask_path) as source:
            expected_mask = torch.from_numpy(source.read(1))
            assert torch.equal(sample['target_mask'], expected_mask)
            target_metadata = sample['metadata']['raster']['target_mask']
            assert target_metadata['crs'] == (source.crs.to_string() if source.crs else None)
            assert target_metadata['transform'] == source.transform
            assert target_metadata['resolution'] == tuple(float(value) for value in source.res)
            assert target_metadata['width'] == source.width
            assert target_metadata['height'] == source.height

        assert sample['metadata']['event'] is None
        assert sample['metadata']['timestamp'] is None


def test_missing_target_mask_remains_unavailable(tmp_path):
    image_path = tmp_path / 'sample_s1.tif'
    image = np.ones((2, 4, 4), dtype=np.uint16)
    with rasterio.open(
        image_path,
        'w',
        driver='GTiff',
        height=4,
        width=4,
        count=2,
        dtype=image.dtype,
        crs='EPSG:32643',
        transform=from_origin(68.0, 10.0, 10.0, 10.0),
    ) as destination:
        destination.write(image)

    sample = S1S2WaterDataset(data_dir=str(tmp_path))[0]

    assert sample['target_mask'] is None
    assert sample['metadata']['source_paths']['target_mask'] is None
    assert sample['metadata']['raster']['target_mask']['transform'] is None


def test_event_manifest_only_populates_verified_metadata():
    from data.datasets import _load_event_metadata

    event = _load_event_metadata('F1', 'flood')

    assert event['event_id'] == 'F1'
    assert event['hazard'] == 'Flood'
    assert event['event_name'] == 'Upper Assam flood'
    assert event['event_date'] is None
    assert event['location'] is None
    assert event['source'] is None
    assert event['ground_truth'] is None


def test_reconstruction_sample_keeps_mask_out_of_legacy_targets():
    dataset = CombinedHazardDataset(split='train', dataset_order=['flood'])
    legacy_sample = dataset[0]
    reconstruction_sample = dataset.get_reconstruction_sample(0)

    assert torch.equal(legacy_sample['target_mask'], legacy_sample['hazard_mask'])
    assert legacy_sample['legacy_targets']['radius_target'].item() == 15.0
    assert set(reconstruction_sample) == UNIFIED_SAMPLE_KEYS
    assert reconstruction_sample['target_mask'] is not None
    assert 'radius_target' not in reconstruction_sample
    assert 'coords_target' not in reconstruction_sample
    assert 'target_tokens' not in reconstruction_sample
    assert reconstruction_sample['geospatial_features'] is None

    reconstruction_dataset = CombinedHazardDataset(
        split='train', dataset_order=['flood'], sample_format='reconstruction'
    )
    assert set(reconstruction_dataset[0]) == UNIFIED_SAMPLE_KEYS


def test_combined_dataset_has_train_val_split():
    from data.datasets import CombinedHazardDataset

    train_ds = CombinedHazardDataset(split='train')
    val_ds = CombinedHazardDataset(split='val')

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert train_ds.dataset_ids != val_ds.dataset_ids

    train_hazards = {item['hazard_name'] for item in [train_ds[i] for i in range(min(3, len(train_ds)))]}
    val_hazards = {item['hazard_name'] for item in [val_ds[i] for i in range(min(3, len(val_ds)))]}

    assert {'flood', 'wildfire', 'landslide'}.issubset(train_hazards | val_hazards)


def test_pipeline_forward():
    print("=" * 60)
    print("  MultiHazard-VLM Pipeline Smoke Test")
    print("=" * 60)

    print("\n[1/4] Initializing MultiHazardVLM...")
    # Write a temporary config for testing with small image size
    import tempfile, yaml
    test_config = {
        'model': {
            'embed_dim': 256,
            'num_heads': 8,
            'num_input_channels': 10,
            'num_geo_attributes': 5,
            'patch_size': 16,
            'max_seq_len': 30,
            'img_size': 64,  # Small for test speed
        }
    }
    config_file = os.path.join(os.path.dirname(__file__), '_test_config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    model = MultiHazardVLM(config_path=config_file)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Model created. Total parameters: {total_params:,}")

    print("\n[2/4] Creating dummy inputs...")
    batch_size = 2
    max_seq_len = 4    # Use small T for speed (full = 30)
    num_channels = 10
    img_size = 64       # Use small image for speed (full = 512)

    # (B, T, C, H, W) spatiotemporal satellite imagery
    images = torch.randn(batch_size, max_seq_len, num_channels, img_size, img_size)
    print(f"       images: {images.shape}")

    # (B, T) quality flags, True = valid
    quality_flags = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
    quality_flags[0, 2] = False  # simulate one missing frame
    print(f"       quality_flags: {quality_flags.shape} (1 missing frame)")

    # H3 indices: dict mapping resolution -> (B, num_cells) tensors
    h3_indices = {
        4: torch.randint(0, 1000, (batch_size, 10)),
        6: torch.randint(0, 1000, (batch_size, 20)),
        8: torch.randint(0, 1000, (batch_size, 30)),
    }
    print(f"       h3_indices: res4={h3_indices[4].shape}, res6={h3_indices[6].shape}, res8={h3_indices[8].shape}")

    # Geo-attributes dict
    geo_attributes = {
        'elevation': torch.randn(batch_size, 1),
        'slope': torch.randn(batch_size, 1),
        'ndvi': torch.rand(batch_size, 1),
        'aspect': torch.randn(batch_size, 1),
        'lithology_class': torch.randint(0, 10, (batch_size,)),
    }
    print(f"       geo_attributes: {list(geo_attributes.keys())}")

    # AOI bounds [min_lat, min_lon, max_lat, max_lon]
    aoi_bounds = torch.tensor([
        [10.0, 70.0, 15.0, 75.0],
        [20.0, 80.0, 25.0, 85.0],
    ])
    print(f"       aoi_bounds: {aoi_bounds.shape}")

    # Target tokens for teacher forcing (language head)
    target_tokens = torch.randint(0, 32000, (batch_size, 16))
    print(f"       target_tokens: {target_tokens.shape}")

    print("\n[3/4] Running forward pass...")
    try:
        outputs = model(
            images=images,
            quality_flags=quality_flags,
            h3_indices=h3_indices,
            geo_attributes=geo_attributes,
            target_tokens=target_tokens,
            aoi_bounds=aoi_bounds,
        )

        print("\n[4/4] PASS - Forward pass successful!")
        print("\n       Output keys and shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"         {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"         {key}: dict with {len(value)} keys")
            elif value is None:
                print(f"         {key}: None")
            else:
                print(f"         {key}: {type(value).__name__}")

        # Validate key outputs exist
        assert 'hazard_logits' in outputs, "Missing hazard_logits"
        assert 'coords_normalized' in outputs, "Missing coords_normalized"
        assert 'radius_km' in outputs, "Missing radius_km"
        assert 'language_logits' in outputs, "Missing language_logits"

        print("\n       All assertions passed!")
        print("=" * 60)
        print("  RESULT: ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n[4/4] FAIL - Forward pass failed with error:")
        print(f"       {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        print("  RESULT: TEST FAILED")
        print("=" * 60)


def test_training_uses_real_evaluation_module():
    train_path = Path(__file__).resolve().parents[1] / 'train.py'
    source = train_path.read_text(encoding='utf-8')

    assert 'from evaluation.metrics import MultiHazardEvaluator' in source
    assert 'from evaluator import MultiHazardEvaluator' not in source

    from evaluation.metrics import MultiHazardEvaluator
    evaluator = MultiHazardEvaluator()
    metrics = evaluator.evaluate(torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 2, 1]))

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_api_dashboard_route_returns_successful_payload():
    from fastapi.testclient import TestClient
    from api.app import app

    client = TestClient(app)
    response = client.get('/api/v1/dashboard/data')

    assert response.status_code == 200
    payload = response.json()
    assert 'hazard_probabilities' in payload
    assert set(payload['hazard_probabilities'].keys()) == {'flood', 'wildfire', 'landslide'}
    assert payload['radius_km'] > 0
    expected_mode = os.getenv('INFERENCE_MODE', 'planetary_computer')
    assert payload['inference_mode'] == expected_mode
    if expected_mode == 'demo_synthetic':
        assert payload['data_source'] == 'Synthetic demo tensors'
    else:
        assert payload['data_source'].startswith('Microsoft Planetary Computer')


def test_evaluation_metrics_endpoint_reports_not_run_or_saved_results(tmp_path, monkeypatch):
    from api.app import get_evaluation_metrics

    report_path = tmp_path / 'metrics_report.json'
    monkeypatch.setenv('EVALUATION_RESULTS_PATH', str(report_path))

    missing_report = asyncio.run(get_evaluation_metrics())
    assert missing_report['status'] == 'not_run'
    assert missing_report['results_available'] is False
    assert missing_report['results_by_hazard'] == {}

    report_path.write_text(json.dumps({
        'results_by_hazard': {
            'overall': {'accuracy': 0.75, 'haversine_dist_km': 1.25},
            'wildfire': {'accuracy': 0.8},
        },
        'sample_counts': {'overall': 12, 'wildfire': 4},
        'metadata': {'run_id': 'test-run'},
    }), encoding='utf-8')

    saved_report = asyncio.run(get_evaluation_metrics())
    assert saved_report['status'] == 'available'
    assert saved_report['results_available'] is True
    assert saved_report['results_by_hazard']['overall']['accuracy'] == 0.75
    assert saved_report['sample_counts']['wildfire'] == 4
    assert saved_report['metadata']['run_id'] == 'test-run'


def test_checkpoint_positional_embedding_is_adapted():
    from api.app import adapt_checkpoint_state

    config_file = Path(__file__).resolve().parent / '_test_config.yaml'
    model = MultiHazardVLM(config_path=str(config_file))
    current_state = model.state_dict()
    checkpoint_state = dict(current_state)
    checkpoint_state['backbone.patch_embed.spatial_pos_embed'] = torch.randn(1, 1, 1024, 256)

    adapted = adapt_checkpoint_state(checkpoint_state, model)

    assert adapted['backbone.patch_embed.spatial_pos_embed'].shape == current_state[
        'backbone.patch_embed.spatial_pos_embed'
    ].shape


if __name__ == "__main__":
    test_pipeline_forward()
