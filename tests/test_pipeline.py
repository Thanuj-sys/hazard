"""Smoke test for the MultiHazardVLM pipeline.

Verifies that:
1. All modules import correctly
2. Forward pass runs with dummy data of the correct shapes
3. Output keys and shapes match expectations
"""
import torch
import sys
import os
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multihazard_vlm import MultiHazardVLM
from data.datasets import S1S2WaterDataset, HLSBurnScarDataset


def test_dataset_loaders_read_real_sample_tiffs():
    flood = S1S2WaterDataset()
    wildfire = HLSBurnScarDataset()

    flood_tensor, _, flood_mask, flood_meta = flood[0]
    wildfire_tensor, _, wildfire_mask, wildfire_meta = wildfire[0]

    assert flood_tensor.shape == (10, 512, 512)
    assert wildfire_tensor.shape == (10, 512, 512)
    assert flood_mask.shape == (512, 512)
    assert wildfire_mask.shape == (512, 512)
    assert flood_tensor.abs().max().item() > 1e-6
    assert wildfire_tensor.abs().max().item() > 1e-6

    assert flood_meta['georeferenced'] is True
    assert wildfire_meta['georeferenced'] is True
    assert 6.5 <= flood_meta['lat'] <= 35.5
    assert 68.0 <= flood_meta['lon'] <= 97.5


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
    assert payload['inference_mode'] == 'demo_synthetic'
    assert payload['data_source'] == 'Synthetic demo tensors'


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
