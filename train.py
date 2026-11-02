import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    checkpoint_path,
    metrics=None,
    train_loss=None,
    validation_loss=None,
    config=None,
):
    """Persist model + optimizer state for resume later."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'config': config or {},
        'metrics': metrics or {},
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Restore training state if a checkpoint exists."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state = checkpoint.get('optimizer_state_dict')
    scheduler_state = checkpoint.get('scheduler_state_dict')
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
        except ValueError as exc:
            optimizer.state.clear()
            print(f"Checkpoint optimizer state is incompatible; starting optimizer fresh: {exc}")
    return int(checkpoint.get('epoch', 0))

from data.datasets import CombinedHazardDataset
from data.preprocessing import align_sample_to_reference
from evaluation.metrics import MultiHazardEvaluator
from models.multihazard_vlm import MultiHazardVLM
from models.segmentation_loss import BinarySegmentationLoss


SEGMENTATION_HAZARDS = ('flood', 'landslide', 'wildfire')


def compute_segmentation_batch_loss(
    segmentation_outputs: Dict[str, torch.Tensor],
    hazards: Sequence[str],
    target_masks: torch.Tensor,
    validity_masks: Optional[torch.Tensor],
    criterion: BinarySegmentationLoss,
) -> Dict[str, Any]:
    """Route each sample to its own hazard head and combine sample-weighted losses."""
    batch_size = len(hazards)
    if batch_size == 0 or target_masks.shape[0] != batch_size:
        raise ValueError("Hazards and target masks must contain the same non-empty batch.")
    if validity_masks is not None and validity_masks.shape[0] != batch_size:
        raise ValueError("Validity masks must have the same batch dimension as targets.")

    reference = next(iter(segmentation_outputs.values()))
    total = reference.sum() * 0.0
    bce_total = total
    dice_total = total
    counts = Counter(hazards)
    for hazard in counts:
        if hazard not in SEGMENTATION_HAZARDS or hazard not in segmentation_outputs:
            raise ValueError(f"Unsupported or missing segmentation head for hazard {hazard!r}.")
        indices = torch.tensor(
            [index for index, value in enumerate(hazards) if value == hazard],
            device=reference.device,
            dtype=torch.long,
        )
        logits = segmentation_outputs[hazard].index_select(0, indices)
        targets = target_masks.to(reference.device).index_select(0, indices)
        valid = None
        if validity_masks is not None:
            valid = validity_masks.to(reference.device).index_select(0, indices)
        group_losses = criterion(logits, targets, valid)
        group_weight = len(indices) / batch_size
        total = total + group_losses['total'] * group_weight
        bce_total = bce_total + group_losses['bce'] * group_weight
        dice_total = dice_total + group_losses['dice'] * group_weight

    return {
        'total': total,
        'segmentation': total,
        'bce': bce_total,
        'dice': dice_total,
        'hazard_sample_counts': {hazard: counts.get(hazard, 0) for hazard in SEGMENTATION_HAZARDS},
    }


def _sample_location(sample: Dict[str, Any]) -> tuple[float, float]:
    metadata = sample.get('metadata') or {}
    event = metadata.get('event') or {}
    location = event.get('location') or {}
    latitude = location.get('latitude', metadata.get('lat'))
    longitude = location.get('longitude', metadata.get('lon'))
    if latitude is None or longitude is None:
        raise ValueError(
            f"Sample {sample.get('event_id')!r} has no verified event or raster coordinates for H3 encoding."
        )
    return float(latitude), float(longitude)


def _h3_embedding_indices(latitude: float, longitude: float) -> Dict[int, int]:
    import h3

    result = {}
    for resolution in (4, 6, 8):
        if hasattr(h3, 'geo_to_h3'):
            cell = h3.geo_to_h3(latitude, longitude, resolution)
        else:
            cell = h3.latlng_to_cell(latitude, longitude, resolution)
        result[resolution] = int(cell, 16) % 10000
    return result


def _feature_scalar(
    name: str,
    aligned_sample: Dict[str, Any],
) -> Optional[float]:
    features = aligned_sample.get('aligned_geospatial_features') or {}
    if name not in features:
        return None
    values = features[name].detach().cpu().float()
    valid = (aligned_sample.get('validity_masks', {}).get('geospatial_features') or {}).get(name)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if valid is not None:
        valid = valid.detach().cpu().bool()
        valid_values = values[:, valid]
    else:
        valid_values = values.reshape(-1)
    valid_values = valid_values[torch.isfinite(valid_values)]
    if valid_values.numel() == 0:
        return None
    if name == 'lithology_class':
        values_int = valid_values.long()
        return float(torch.mode(values_int).values.item())
    return float(valid_values.mean().item())


def _derive_ndvi(hazard: str, image: torch.Tensor, image_validity: torch.Tensor) -> Optional[float]:
    if hazard == 'flood' and image.shape[0] >= 6:
        red_index, nir_index = 4, 5
    elif hazard == 'wildfire' and image.shape[0] >= 4:
        red_index, nir_index = 2, 3
    else:
        return None
    red, nir = image[red_index], image[nir_index]
    denominator = nir + red
    valid = image_validity & torch.isfinite(denominator) & (denominator.abs() > 1e-6)
    if not valid.any():
        return None
    return float(((nir[valid] - red[valid]) / denominator[valid]).mean().item())


def prepare_reconstruction_batch(
    samples: Sequence[Dict[str, Any]],
    model: MultiHazardVLM,
    device: torch.device,
) -> Dict[str, Any]:
    """Align reconstruction samples and adapt them to the existing model input contract."""
    if not samples:
        raise ValueError("Cannot prepare an empty reconstruction batch.")
    model_size = int(model.backbone.img_size)
    output_size = model.segmentation_output_size
    input_channels = int(model.backbone.patch_embed.num_channels)
    image_tensors = []
    target_tensors = []
    validity_tensors = []
    hazards = []
    event_ids = []
    h3_rows = {resolution: [] for resolution in (4, 6, 8)}
    geo_rows = {name: [] for name in ('elevation', 'slope', 'ndvi', 'aspect', 'lithology_class')}
    missing_features = set()

    for sample in samples:
        hazard = str(sample.get('hazard', '')).lower()
        if hazard not in SEGMENTATION_HAZARDS:
            raise ValueError(f"Sample has unsupported hazard {hazard!r}.")
        if sample.get('target_mask') is None:
            raise ValueError(f"{hazard} sample has no real target_mask; segmentation training requires one.")

        aligned = align_sample_to_reference(sample)
        image = aligned['aligned_image'].detach().cpu().float()
        if image.ndim != 3:
            raise ValueError(f"Aligned image must be [C,H,W], got {tuple(image.shape)}.")
        image_validity = aligned['validity_masks']['image'].detach().cpu().bool()
        if image.shape[-2:] != image_validity.shape:
            raise ValueError("Aligned image and its validity mask have different spatial dimensions.")
        image = image.masked_fill(~image_validity.unsqueeze(0), 0.0)
        if image.shape[0] > input_channels:
            raise ValueError(
                f"Aligned image has {image.shape[0]} channels but model expects {input_channels}."
            )
        if image.shape[0] < input_channels:
            padding = torch.zeros(
                input_channels - image.shape[0], *image.shape[-2:], dtype=image.dtype
            )
            image = torch.cat((image, padding), dim=0)
        image_coverage = F.interpolate(
            image_validity.float().view(1, 1, *image_validity.shape),
            size=(model_size, model_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        resized_image_values = F.interpolate(
            (image * image_validity.unsqueeze(0)).unsqueeze(0),
            size=(model_size, model_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        resized_image = torch.where(
            image_coverage > 1e-6,
            resized_image_values / image_coverage.clamp_min(1e-6),
            torch.zeros_like(resized_image_values),
        )
        image_tensors.append(resized_image)

        target = aligned['aligned_target_mask'].detach().cpu().float()
        target_validity = aligned['validity_masks']['target_mask'].detach().cpu().bool()
        target_validity &= image_validity
        if hazard == 'wildfire':
            target_validity &= target.ne(-1)
        target_validity &= torch.isfinite(target)
        unknown = target_validity & target.ne(0) & target.ne(1)
        if unknown.any():
            values = torch.unique(target[unknown]).tolist()
            raise ValueError(f"{hazard} target has unsupported valid labels {values}; expected binary 0/1.")
        target = torch.where(target_validity, target, torch.zeros_like(target))
        target_t = F.interpolate(
            target.view(1, 1, *target.shape), size=output_size, mode='nearest'
        ).squeeze(0).squeeze(0)
        valid_t = F.interpolate(
            target_validity.float().view(1, 1, *target_validity.shape),
            size=output_size,
            mode='nearest',
        ).squeeze(0).squeeze(0).bool()
        if not valid_t.any():
            raise ValueError(f"{hazard} target has no valid pixels at model output resolution.")
        target_t = torch.where(valid_t, target_t, torch.zeros_like(target_t))
        target_t = target_t.round()
        target_t = target_t.unsqueeze(0)
        valid_t = valid_t.unsqueeze(0)
        target_tensors.append(target_t)
        validity_tensors.append(valid_t)

        latitude, longitude = _sample_location(sample)
        encoded = _h3_embedding_indices(latitude, longitude)
        for resolution, token in encoded.items():
            h3_rows[resolution].append([token])

        for name in geo_rows:
            scalar = _feature_scalar(name, aligned)
            if scalar is None and name == 'ndvi':
                spectral_channels = aligned['aligned_image'].shape[0]
                scalar = _derive_ndvi(hazard, image[:spectral_channels], image_validity)
            if scalar is None:
                missing_features.add(name)
                scalar = 0.0
            geo_rows[name].append(scalar)

        hazards.append(hazard)
        event_ids.append(sample.get('event_id'))

    images = torch.stack(image_tensors).unsqueeze(1).to(device)
    model_inputs = {
        'images': images,
        'quality_flags': torch.ones((len(samples), 1), dtype=torch.bool, device=device),
        'h3_indices': {
            resolution: torch.tensor(rows, dtype=torch.long, device=device)
            for resolution, rows in h3_rows.items()
        },
        'geo_attributes': {
            name: torch.tensor(values, dtype=torch.long if name == 'lithology_class' else torch.float32, device=device)
            if name == 'lithology_class'
            else torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1)
            for name, values in geo_rows.items()
        },
    }
    return {
        'model_inputs': model_inputs,
        'hazards': hazards,
        'target_masks': torch.stack(target_tensors).to(device),
        'validity_masks': torch.stack(validity_tensors).to(device),
        'event_ids': event_ids,
        'missing_geospatial_features': sorted(missing_features),
        'positive_pixels': int(torch.stack(target_tensors).sum().item()),
        'valid_pixels': int(torch.stack(validity_tensors).sum().item()),
    }


def segmentation_training_step(
    model: MultiHazardVLM,
    batch: Dict[str, Any],
    criterion: BinarySegmentationLoss,
    optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Run one segmentation forward/loss step, optionally applying an optimizer update."""
    if optimizer is not None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch['model_inputs'])
    losses = compute_segmentation_batch_loss(
        outputs['segmentation'],
        batch['hazards'],
        batch['target_masks'],
        batch.get('validity_masks'),
        criterion,
    )
    if not torch.isfinite(losses['total']):
        raise FloatingPointError("Segmentation objective is non-finite.")
    if optimizer is not None:
        losses['total'].backward()
        optimizer.step()
    return {'outputs': outputs, 'losses': losses}


def run_segmentation_training(args, config: Dict[str, Any], device: torch.device) -> None:
    model = MultiHazardVLM(
        config_path=args.config if os.path.exists(args.config) else None
    ).to(device)
    train_dataset = CombinedHazardDataset(split='train', sample_format='reconstruction')
    validation_dataset = CombinedHazardDataset(split='val', sample_format='reconstruction')
    if len(train_dataset) == 0:
        raise ValueError("Segmentation training has no reconstruction samples.")

    collate_fn = CombinedHazardDataset.get_reconstruction_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    training_config = config.get('training') or {}
    criterion = BinarySegmentationLoss(
        bce_weight=float(training_config.get('segmentation_bce_weight', 1.0)),
        dice_weight=float(training_config.get('segmentation_dice_weight', 1.0)),
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume) if args.resume else 0
    missing_features_reported = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_totals = Counter()
        train_batches = 0
        train_sample_counts = Counter()
        train_positive_pixels = 0
        train_valid_pixels = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Segmentation]")
        for samples in progress:
            batch = prepare_reconstruction_batch(samples, model, device)
            if batch['missing_geospatial_features'] and not missing_features_reported:
                print(
                    "Unavailable ancillary inputs use deterministic zero placeholders "
                    f"(not targets): {', '.join(batch['missing_geospatial_features'])}."
                )
                missing_features_reported = True

            result = segmentation_training_step(model, batch, criterion, optimizer)
            losses = result['losses']
            train_totals['total'] += float(losses['total'].detach().item())
            train_totals['bce'] += float(losses['bce'].detach().item())
            train_totals['dice'] += float(losses['dice'].detach().item())
            train_batches += 1
            train_sample_counts.update(losses['hazard_sample_counts'])
            train_positive_pixels += batch['positive_pixels']
            train_valid_pixels += batch['valid_pixels']
            progress.set_postfix(
                total=f"{losses['total'].item():.4f}",
                segmentation=f"{losses['segmentation'].item():.4f}",
                bce=f"{losses['bce'].item():.4f}",
                dice=f"{losses['dice'].item():.4f}",
            )

        train_loss = train_totals['total'] / max(train_batches, 1)
        validation_totals = Counter()
        validation_batches = 0
        validation_sample_counts = Counter()
        model.eval()
        with torch.no_grad():
            for samples in tqdm(
                validation_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} [Segmentation Val]",
            ):
                batch = prepare_reconstruction_batch(samples, model, device)
                result = segmentation_training_step(model, batch, criterion)
                losses = result['losses']
                validation_totals['total'] += float(losses['total'].item())
                validation_totals['bce'] += float(losses['bce'].item())
                validation_totals['dice'] += float(losses['dice'].item())
                validation_batches += 1
                validation_sample_counts.update(losses['hazard_sample_counts'])

        validation_loss = (
            validation_totals['total'] / validation_batches if validation_batches else None
        )
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.epochs} | mode=segmentation "
            f"train_total={train_loss:.4f} "
            f"train_segmentation={train_loss:.4f} "
            f"train_bce={train_totals['bce'] / max(train_batches, 1):.4f} "
            f"train_dice={train_totals['dice'] / max(train_batches, 1):.4f} "
            f"validation_loss={validation_loss if validation_loss is not None else 'unavailable'} "
            f"train_samples={dict(train_sample_counts)} "
            f"validation_samples={dict(validation_sample_counts)} "
            f"positive_pixels={train_positive_pixels}/{train_valid_pixels}"
        )
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f'model_epoch_{epoch + 1}.pt'
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            checkpoint_path,
            metrics={
                'training_mode': 'segmentation',
                'train_sample_counts': dict(train_sample_counts),
                'validation_sample_counts': dict(validation_sample_counts),
                'positive_pixels': train_positive_pixels,
                'valid_pixels': train_valid_pixels,
            },
            train_loss=train_loss,
            validation_loss=validation_loss,
            config=config,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiHazard-VLM")
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for the prototype training loop')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory where checkpoints are saved')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument(
        '--training-mode',
        choices=('legacy', 'segmentation'),
        default=None,
        help='Use legacy multitask targets or real-mask segmentation training.',
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load config
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    config.setdefault('model', {})
    config['model']['img_size'] = args.img_size
    training_config = config.setdefault('training', {})
    training_mode = args.training_mode or training_config.get('mode', 'legacy')
    if training_mode not in {'legacy', 'segmentation'}:
        raise ValueError(f"Unsupported training mode {training_mode!r}.")
    training_config['mode'] = training_mode
    if training_mode == 'segmentation':
        run_segmentation_training(args, config, device)
        return

    # 2. Datasets & Loaders
    train_dataset = CombinedHazardDataset(split='train', img_size=args.img_size)
    val_dataset = CombinedHazardDataset(split='val', img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model
    model = MultiHazardVLM(config_path=config_path if os.path.exists(config_path) else None).to(device)

    # 4. Losses
    cls_criterion = nn.CrossEntropyLoss()
    coord_criterion = nn.SmoothL1Loss()
    radius_criterion = nn.HuberLoss()
    lang_criterion = nn.CrossEntropyLoss()

    # Loss Weights
    w_cls, w_coord, w_rad, w_lang = 1.0, 10.0, 0.1, 1.0

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    evaluator = MultiHazardEvaluator()

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            images = batch['images'].to(device)
            quality_flags = batch['quality_flags'].to(device)
            h3_indices = {k: v.to(device) for k, v in batch['h3_index'].items()}
            geo_attributes = {k: v.to(device) for k, v in batch['geo_attributes'].items()}

            hazard_target = batch['hazard_label'].to(device)
            coords_target = batch['coords_target'].to(device)
            radius_target = batch['radius_target'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            optimizer.zero_grad()

            outputs = model(
                images=images,
                quality_flags=quality_flags,
                h3_indices=h3_indices,
                geo_attributes=geo_attributes,
                target_tokens=target_tokens
            )

            loss_cls = cls_criterion(outputs['hazard_logits'], hazard_target)
            loss_coord = coord_criterion(outputs['coords_normalized'], coords_target)
            loss_rad = radius_criterion(outputs['radius_km'].squeeze(), radius_target.squeeze())

            B, L, V = outputs['language_logits'].shape
            loss_lang = lang_criterion(
                outputs['language_logits'].view(-1, V),
                target_tokens.view(-1)
            )

            total_loss = w_cls * loss_cls + w_coord * loss_coord + w_rad * loss_rad + w_lang * loss_lang
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = batch['images'].to(device)
                quality_flags = batch['quality_flags'].to(device)
                h3_indices = {k: v.to(device) for k, v in batch['h3_index'].items()}
                geo_attributes = {k: v.to(device) for k, v in batch['geo_attributes'].items()}

                outputs = model(
                    images=images,
                    quality_flags=quality_flags,
                    h3_indices=h3_indices,
                    geo_attributes=geo_attributes
                )

                val_loss += nn.functional.cross_entropy(outputs['hazard_logits'], batch['hazard_label'].to(device)).item()
                all_preds.append(torch.argmax(outputs['hazard_logits'], dim=-1).cpu())
                all_targets.append(batch['hazard_label'].cpu())

        metrics = evaluator.evaluate(torch.cat(all_preds), torch.cat(all_targets)) if all_preds else {}
        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | metrics={metrics}")

        checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            checkpoint_path,
            metrics,
            train_loss=epoch_loss,
            validation_loss=val_loss,
            config=config,
        )

if __name__ == "__main__":
    main()
