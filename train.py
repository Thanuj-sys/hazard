import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, metrics=None):
    """Persist model + optimizer state for resume later."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return int(checkpoint.get('epoch', 0))

from data.datasets import CombinedHazardDataset
from evaluation.metrics import MultiHazardEvaluator
from models.multihazard_vlm import MultiHazardVLM

def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiHazard-VLM")
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for the prototype training loop')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory where checkpoints are saved')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from')
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
        save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path, metrics)

if __name__ == "__main__":
    main()
