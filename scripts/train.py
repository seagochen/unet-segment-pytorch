#!/usr/bin/env python3
"""
UNet Training Script for Lung Tumor Segmentation.

Usage:
    # Train with default config
    python scripts/train.py --config configs/lung_tumor.yaml

    # Override specific parameters
    python scripts/train.py --config configs/lung_tumor.yaml --epochs 50 --batch-size 16

    # Resume training from checkpoint
    python scripts/train.py --config configs/lung_tumor.yaml --resume runs/lung_tumor/weights/last.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unet.models import UNet
from unet.data.dataset import LungTumorDataset
from unet.data.augmentations import get_train_transforms, get_val_transforms
from unet.utils.loss import create_loss_function
from unet.utils.metrics import SegmentationMetrics
from unet.utils.general import set_seed, get_device, load_config, increment_path
from unet.utils.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from unet.utils.plots import plot_training_curves, plot_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train UNet for lung tumor segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config
    parser.add_argument('--config', type=str, default='configs/lung_tumor.yaml',
                        help='Path to config file')

    # Data
    parser.add_argument('--data', type=str, default=None,
                        help='Override data root path')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Override image size')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override number of workers')

    # Training
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Output
    parser.add_argument('--name', type=str, default=None,
                        help='Override experiment name')
    parser.add_argument('--project', type=str, default=None,
                        help='Override project directory')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu, mps)')

    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    metrics: SegmentationMetrics,
    device: torch.device
) -> dict:
    """
    Validate the model.

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Validating', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        metrics.update(outputs, masks)

    # Compute final metrics
    results = metrics.compute()
    results['loss'] = total_loss / len(dataloader)

    return results


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data:
        config['data']['root'] = args.data
    if args.img_size:
        config['data']['img_size'] = args.img_size
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.workers:
        config['data']['num_workers'] = args.workers
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.lr:
        config['train']['lr'] = args.lr
    if args.name:
        config['output']['experiment_name'] = args.name
    if args.project:
        config['output']['save_dir'] = args.project
    if args.device:
        config['device'] = args.device

    # Setup
    set_seed(config.get('seed', 42))
    device = get_device(config.get('device', ''))
    print(f"Using device: {device}")

    # Create output directory
    save_dir = Path(config['output']['save_dir']) / config['output']['experiment_name']
    save_dir = increment_path(save_dir)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Data
    print("\nLoading data...")
    data_config = config['data']
    aug_config = config.get('augmentation', {})

    train_transform = get_train_transforms(
        img_size=data_config['img_size'],
        p_flip=aug_config.get('horizontal_flip', 0.5),
        rotation_limit=aug_config.get('rotation_limit', 15),
        p_elastic=aug_config.get('elastic', 0.3),
        p_brightness=aug_config.get('brightness_contrast', 0.3),
    ) if aug_config.get('enabled', True) else get_val_transforms(data_config['img_size'])

    val_transform = get_val_transforms(img_size=data_config['img_size'])

    train_dataset = LungTumorDataset(
        root=data_config['root'],
        split='train',
        transform=train_transform,
        val_ratio=data_config.get('val_ratio', 0.2),
        seed=config.get('seed', 42),
        img_size=data_config['img_size'],
    )

    val_dataset = LungTumorDataset(
        root=data_config['root'],
        split='val',
        transform=val_transform,
        val_ratio=data_config.get('val_ratio', 0.2),
        seed=config.get('seed', 42),
        img_size=data_config['img_size'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    print("\nCreating model...")
    model_config = config['model']
    model = UNet(
        n_channels=model_config['n_channels'],
        n_classes=model_config['n_classes'],
        bilinear=model_config.get('bilinear', True),
        base_features=model_config.get('base_features', 64),
    ).to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Loss function
    loss_config = config['loss']
    criterion = create_loss_function(
        loss_type=loss_config['type'],
        ce_weight=loss_config.get('ce_weight', 1.0),
        dice_weight=loss_config.get('dice_weight', 1.0),
        class_weights=loss_config.get('class_weights'),
    )

    # Optimizer
    train_config = config['train']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config.get('weight_decay', 0.0001),
    )

    # Scheduler
    scheduler_config = config.get('scheduler', {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=scheduler_config.get('factor', 0.5),
        patience=scheduler_config.get('patience', 10),
        min_lr=scheduler_config.get('min_lr', 1e-6),
    )

    # Callbacks
    early_stop_config = config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stop_config.get('patience', 20),
        mode=early_stop_config.get('mode', 'max'),
    ) if early_stop_config.get('enabled', True) else None

    checkpoint = ModelCheckpoint(
        save_dir=weights_dir,
        monitor=early_stop_config.get('monitor', 'mean_dice'),
        mode=early_stop_config.get('mode', 'max'),
        save_last=config['output'].get('save_last', True),
    )

    # Metrics
    metrics = SegmentationMetrics(
        num_classes=model_config['n_classes'],
        class_names=['background', 'tumor'],
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_accuracy': [],
        'lr': [],
    }

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    num_epochs = train_config['epochs']
    grad_clip = train_config.get('grad_clip', 0.0)

    for epoch in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{num_epochs} (lr={current_lr:.2e})")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )

        # Validate
        val_results = validate(model, val_loader, criterion, metrics, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_dice'].append(val_results['mean_dice'])
        history['val_iou'].append(val_results['mean_iou'])
        history['val_accuracy'].append(val_results['pixel_accuracy'])
        history['lr'].append(current_lr)

        # Print results
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f} | "
              f"Dice: {val_results['mean_dice']:.4f} | "
              f"IoU: {val_results['mean_iou']:.4f} | "
              f"Acc: {val_results['pixel_accuracy']:.4f}")
        print(f"  Tumor Dice: {val_results['class_dice'].get('tumor', 0):.4f} | "
              f"Tumor IoU: {val_results['class_iou'].get('tumor', 0):.4f}")

        # Save checkpoint
        checkpoint.save(
            model, optimizer, epoch, val_results,
            scheduler=scheduler, config=config
        )

        # Update scheduler
        scheduler.step(val_results['mean_dice'])

        # Early stopping
        if early_stopping and early_stopping(val_results['mean_dice']):
            print("\nEarly stopping triggered!")
            break

    print("\n" + "=" * 60)
    print("Training complete!")

    # Save training curves
    plot_training_curves(history, save_path=save_dir / 'training_curves.png')

    # Save sample predictions
    print("\nSaving sample predictions...")
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)

        plot_predictions(
            images, masks, predictions,
            num_samples=4,
            save_path=save_dir / 'val_predictions.png',
            class_names=['background', 'tumor']
        )

    # Final summary
    print(f"\nResults saved to: {save_dir}")
    print(f"Best model: {weights_dir / 'best.pt'}")

    best_dice = max(history['val_dice'])
    best_epoch = history['val_dice'].index(best_dice) + 1
    print(f"Best validation Dice: {best_dice:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()
