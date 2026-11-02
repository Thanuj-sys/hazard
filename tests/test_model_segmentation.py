import torch
import yaml

from models.multihazard_vlm import MultiHazardVLM


def make_model_config(tmp_path):
    config = {
        "model": {
            "embed_dim": 64,
            "num_heads": 8,
            "num_input_channels": 10,
            "num_geo_attributes": 5,
            "patch_size": 16,
            "max_seq_len": 3,
            "img_size": 64,
            "segmentation_enabled": True,
            "segmentation_num_classes": 1,
            "segmentation_decoder_channels": 16,
            "segmentation_output_size": [64, 64],
        }
    }
    config_path = tmp_path / "segmentation_model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def make_inputs(batch_size=2):
    return {
        "images": torch.randn(batch_size, 3, 10, 64, 64),
        "quality_flags": torch.tensor(
            [[True, True, False], [True, True, True]][:batch_size], dtype=torch.bool
        ),
        "h3_indices": {
            4: torch.randint(0, 1000, (batch_size, 1)),
            6: torch.randint(0, 1000, (batch_size, 1)),
            8: torch.randint(0, 1000, (batch_size, 1)),
        },
        "geo_attributes": {
            "elevation": torch.randn(batch_size, 1),
            "slope": torch.rand(batch_size, 1),
            "ndvi": torch.rand(batch_size, 1),
            "aspect": torch.randn(batch_size, 1),
            "lithology_class": torch.randint(0, 10, (batch_size,)),
        },
        "target_tokens": torch.randint(0, 100, (batch_size, 4)),
        "aoi_bounds": torch.tensor(
            [[10.0, 70.0, 15.0, 75.0], [20.0, 80.0, 25.0, 85.0]][:batch_size]
        ),
    }


def test_segmentation_heads_return_binary_logits_and_keep_legacy_outputs(tmp_path):
    model = MultiHazardVLM(str(make_model_config(tmp_path)))
    outputs = model(**make_inputs())

    assert model.segmentation_enabled
    assert set(outputs["segmentation"]) == {"flood", "landslide", "wildfire"}
    for logits in outputs["segmentation"].values():
        assert logits.shape == (2, 1, 64, 64)

    flood = outputs["segmentation"]["flood"]
    landslide = outputs["segmentation"]["landslide"]
    wildfire = outputs["segmentation"]["wildfire"]
    assert not torch.equal(flood, landslide)
    assert not torch.equal(flood, wildfire)
    assert not torch.equal(landslide, wildfire)
    assert len({id(head) for head in model.segmentation_head.heads.values()}) == 3

    assert outputs["hazard_logits"].shape == (2, 3)
    assert outputs["hazard_probs"].shape == (2, 3)
    assert outputs["coords_normalized"].shape == (2, 2)
    assert outputs["coords_geo"].shape == (2, 2)
    assert outputs["radius_km"].shape == (2, 1)
    assert outputs["language_logits"].shape[:2] == (2, 4)
    assert "hazard_type" in outputs["report_sections"]


def test_segmentation_loss_reaches_decoder_and_all_hazard_heads(tmp_path):
    model = MultiHazardVLM(str(make_model_config(tmp_path)))
    outputs = model(**make_inputs())
    segmentation = outputs["segmentation"]
    targets = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32)
    loss = sum(
        torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        for logits in segmentation.values()
    )

    loss.backward()

    assert model.segmentation_head.decoder[0].weight.grad is not None
    for hazard in ("flood", "landslide", "wildfire"):
        assert model.segmentation_head.heads[hazard].weight.grad is not None


def test_old_checkpoint_loads_with_new_segmentation_head_initialized(tmp_path):
    model = MultiHazardVLM(str(make_model_config(tmp_path)))
    legacy_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("segmentation_head.")
    }
    expected_weight = legacy_state["classification_head.mlp.0.weight"].clone()

    incompatible = model.load_state_dict(legacy_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(model.classification_head.mlp[0].weight, expected_weight)
