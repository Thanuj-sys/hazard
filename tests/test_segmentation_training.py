import torch
import yaml

from data.datasets import CASLandslideDataset, HLSBurnScarDataset, S1S2WaterDataset
from models.multihazard_vlm import MultiHazardVLM
from models.segmentation_loss import BinarySegmentationLoss
from train import (
    compute_segmentation_batch_loss,
    load_checkpoint,
    prepare_reconstruction_batch,
    save_checkpoint,
    segmentation_training_step,
)


def write_tiny_config(tmp_path):
    config = {
        "model": {
            "embed_dim": 64,
            "num_heads": 8,
            "num_input_channels": 10,
            "num_geo_attributes": 5,
            "patch_size": 16,
            "max_seq_len": 2,
            "img_size": 64,
            "segmentation_enabled": True,
            "segmentation_num_classes": 1,
            "segmentation_decoder_channels": 16,
            "segmentation_output_size": [64, 64],
        },
        "training": {
            "mode": "segmentation",
            "segmentation_bce_weight": 1.0,
            "segmentation_dice_weight": 1.0,
        },
    }
    path = tmp_path / "tiny_training.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def make_model_inputs(batch_size=2):
    return {
        "images": torch.randn(batch_size, 1, 10, 64, 64),
        "quality_flags": torch.ones(batch_size, 1, dtype=torch.bool),
        "h3_indices": {
            4: torch.randint(0, 10000, (batch_size, 1)),
            6: torch.randint(0, 10000, (batch_size, 1)),
            8: torch.randint(0, 10000, (batch_size, 1)),
        },
        "geo_attributes": {
            "elevation": torch.zeros(batch_size, 1),
            "slope": torch.zeros(batch_size, 1),
            "ndvi": torch.zeros(batch_size, 1),
            "aspect": torch.zeros(batch_size, 1),
            "lithology_class": torch.zeros(batch_size, dtype=torch.long),
        },
    }


def test_bce_dice_loss_is_finite_and_backpropagates():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 8, 8), dtype=torch.float32)
    criterion = BinarySegmentationLoss(bce_weight=1.0, dice_weight=1.0)

    losses = criterion(logits, targets)

    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["bce"])
    assert torch.isfinite(losses["dice"])
    losses["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_validity_mask_excludes_nodata_pixels_from_loss_and_gradient():
    criterion = BinarySegmentationLoss(bce_weight=1.0, dice_weight=1.0)
    target = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    valid = torch.tensor([[[True, True], [True, False]]])
    first_logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    first = criterion(first_logits, target, valid)
    first["total"].backward()

    changed_logits = torch.tensor([[[[0.0, 0.0], [0.0, 10000.0]]]], requires_grad=True)
    changed_target = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])
    second = criterion(changed_logits, changed_target, valid)

    assert torch.allclose(first["total"], second["total"])
    assert first_logits.grad[0, 0, 1, 1].item() == 0.0


def test_mixed_batch_routes_each_target_only_to_its_hazard_head():
    hazards = ["flood", "landslide", "wildfire"]
    outputs = {
        hazard: torch.zeros(3, 1, 4, 4, requires_grad=True)
        for hazard in ("flood", "landslide", "wildfire")
    }
    targets = torch.randint(0, 2, (3, 1, 4, 4), dtype=torch.float32)
    valid = torch.ones_like(targets, dtype=torch.bool)

    losses = compute_segmentation_batch_loss(
        outputs,
        hazards,
        targets,
        valid,
        BinarySegmentationLoss(),
    )
    losses["total"].backward()

    assert losses["hazard_sample_counts"] == {"flood": 1, "landslide": 1, "wildfire": 1}
    for sample_index, hazard in enumerate(hazards):
        gradient = outputs[hazard].grad
        assert gradient[sample_index].abs().sum() > 0
        other_indices = [index for index in range(3) if index != sample_index]
        assert gradient[other_indices].count_nonzero() == 0


def test_synthetic_segmentation_step_updates_hazard_head(tmp_path):
    model = MultiHazardVLM(str(write_tiny_config(tmp_path)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.segmentation_head.heads["flood"].weight.detach().clone()
    batch = {
        "model_inputs": make_model_inputs(),
        "hazards": ["flood", "flood"],
        "target_masks": torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32),
        "validity_masks": torch.ones(2, 1, 64, 64, dtype=torch.bool),
    }

    result = segmentation_training_step(
        model, batch, BinarySegmentationLoss(), optimizer
    )

    assert torch.isfinite(result["losses"]["total"])
    assert not torch.equal(before, model.segmentation_head.heads["flood"].weight)


def test_staged_local_masks_complete_one_training_update(tmp_path):
    model = MultiHazardVLM(str(write_tiny_config(tmp_path)))
    samples = [
        S1S2WaterDataset()[0],
        CASLandslideDataset()[0],
        HLSBurnScarDataset()[0],
    ]
    batch = prepare_reconstruction_batch(samples, model, torch.device("cpu"))

    assert batch["hazards"] == ["flood", "landslide", "wildfire"]
    assert batch["event_ids"] == [None, None, None]
    assert batch["model_inputs"]["images"].shape == (3, 1, 10, 64, 64)
    assert batch["target_masks"].shape == (3, 1, 64, 64)
    assert batch["validity_masks"].shape == (3, 1, 64, 64)
    assert batch["valid_pixels"] > 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {
        hazard: model.segmentation_head.heads[hazard].weight.detach().clone()
        for hazard in ("flood", "landslide", "wildfire")
    }
    result = segmentation_training_step(
        model, batch, BinarySegmentationLoss(), optimizer
    )

    assert torch.isfinite(result["losses"]["total"])
    assert result["losses"]["hazard_sample_counts"] == {
        "flood": 1, "landslide": 1, "wildfire": 1
    }
    for hazard in before:
        assert not torch.equal(before[hazard], model.segmentation_head.heads[hazard].weight)


def test_legacy_checkpoint_without_segmentation_weights_loads(tmp_path):
    model = MultiHazardVLM(str(write_tiny_config(tmp_path)))
    legacy_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("segmentation_head.")
    ]
    legacy_optimizer = torch.optim.AdamW(legacy_parameters, lr=1e-3)
    legacy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(legacy_optimizer, T_max=2)
    checkpoint = {
        "epoch": 3,
        "model_state_dict": {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("segmentation_head.")
        },
        "optimizer_state_dict": legacy_optimizer.state_dict(),
        "scheduler_state_dict": legacy_scheduler.state_dict(),
    }
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    resumed_epoch = load_checkpoint(model, optimizer, scheduler, str(path))

    assert resumed_epoch == 3
    assert model.segmentation_head is not None


def test_checkpoint_saves_segmentation_model_losses_and_config(tmp_path):
    model = MultiHazardVLM(str(write_tiny_config(tmp_path)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    path = tmp_path / "segmentation.pt"
    config = {"training": {"mode": "segmentation"}}

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch=1,
        checkpoint_path=str(path),
        train_loss=1.25,
        validation_loss=0.75,
        config=config,
    )
    checkpoint = torch.load(path, map_location="cpu")

    assert checkpoint["epoch"] == 1
    assert checkpoint["train_loss"] == 1.25
    assert checkpoint["validation_loss"] == 0.75
    assert checkpoint["config"] == config
    assert "segmentation_head.heads.flood.weight" in checkpoint["model_state_dict"]
