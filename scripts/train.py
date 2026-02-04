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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unet.models import UNet, AttentionUNet
from unet.data.dataset import LungTumorDataset
from unet.data.augmentations import get_train_transforms, get_val_transforms
from unet.utils.loss import create_loss_function, DeepSupervisionLoss
from unet.utils.metrics import SegmentationMetrics
from unet.utils.general import set_seed, get_device, load_config, increment_path, ModelEMA
from unet.utils.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from unet.utils.plots import plot_training_curves, plot_predictions


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs, warmup_lr, base_lr):
    """
    Create a scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: The optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        warmup_lr: Starting learning rate for warmup
        base_lr: Target learning rate after warmup
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return warmup_lr / base_lr + (1 - warmup_lr / base_lr) * (epoch / warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    grad_clip: float = 0.0,
    ema: ModelEMA = None
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

        # Update EMA after each optimizer step
        if ema is not None:
            ema.update(model)

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
    model_type = model_config.get('type', 'unet').lower()

    deep_supervision = model_config.get('deep_supervision', False)

    model_kwargs = {
        'n_channels': model_config['n_channels'],
        'n_classes': model_config['n_classes'],
        'bilinear': model_config.get('bilinear', True),
        'base_features': model_config.get('base_features', 64),
    }

    if model_type == 'attention_unet' or model_type == 'attention':
        model = AttentionUNet(**model_kwargs, deep_supervision=deep_supervision).to(device)
        print(f"Using Attention U-Net" + (" with Deep Supervision" if deep_supervision else ""))
    else:
        model = UNet(**model_kwargs).to(device)
        print(f"Using standard U-Net")

    print(f"Model parameters: {model.get_num_params():,}")

    # Create EMA for model stability
    ema_config = config.get('ema', {})
    use_ema = ema_config.get('enabled', True)  # Enable by default
    if use_ema:
        ema_decay = ema_config.get('decay', 0.99)
        ema_warmup = ema_config.get('warmup_epochs', 5)
        ema = ModelEMA(model, decay=ema_decay)
        print(f"Using EMA with decay={ema_decay}, warmup={ema_warmup} epochs")
    else:
        ema = None

    # Loss function
    loss_config = config['loss']
    base_criterion = create_loss_function(
        loss_type=loss_config['type'],
        ce_weight=loss_config.get('ce_weight', 1.0),
        dice_weight=loss_config.get('dice_weight', 1.0),
        class_weights=loss_config.get('class_weights'),
        focal_gamma=loss_config.get('focal_gamma', 0.75),
        tversky_alpha=loss_config.get('tversky_alpha', 0.7),
        tversky_beta=loss_config.get('tversky_beta', 0.3),
        balanced_class_weight=loss_config.get('balanced_class_weight', 0.5),
    )

    # Wrap with deep supervision loss if enabled
    if deep_supervision:
        ds_weights = loss_config.get('ds_weights', [1.0, 0.4, 0.2, 0.1])
        criterion = DeepSupervisionLoss(base_criterion, weights=ds_weights)
        print(f"Loss function: {loss_config['type']} + Deep Supervision (weights={ds_weights})")
    else:
        criterion = base_criterion
        print(f"Loss function: {loss_config['type']}")

    # Optimizer
    train_config = config['train']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config.get('weight_decay', 0.0001),
    )

    # Scheduler
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

    if scheduler_type == 'warmup_cosine':
        # Warmup + Cosine Annealing
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        warmup_lr = scheduler_config.get('warmup_lr', 1e-6)
        scheduler = get_warmup_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=train_config['epochs'],
            warmup_lr=warmup_lr,
            base_lr=train_config['lr'],
        )
        use_warmup_scheduler = True
        print(f"Using warmup+cosine scheduler (warmup: {warmup_epochs} epochs)")
    else:
        # ReduceLROnPlateau (default)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 1e-6),
        )
        use_warmup_scheduler = False

    # Callbacks
    early_stop_config = config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stop_config.get('patience', 20),
        mode=early_stop_config.get('mode', 'max'),
    ) if early_stop_config.get('enabled', True) else None

    # Use tumor dice for monitoring (not mean dice which rewards all-background)
    monitor_metric = early_stop_config.get('monitor', 'class_dice.tumor')
    checkpoint = ModelCheckpoint(
        save_dir=weights_dir,
        monitor=monitor_metric,
        mode=early_stop_config.get('mode', 'max'),
        save_last=config['output'].get('save_last', True),
    )
    print(f"Monitoring metric: {monitor_metric}")

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
        'tumor_dice': [],  # Track tumor dice specifically
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
            model, train_loader, criterion, optimizer, device, grad_clip, ema=ema
        )

        # EMA warmup: use training model for first N epochs, then switch to EMA
        # This gives EMA time to accumulate meaningful weights
        ema_warmup_epochs = ema_config.get('warmup_epochs', 5) if ema is not None else 0
        use_ema_for_val = ema is not None and epoch >= ema_warmup_epochs
        val_model = ema.ema_model if use_ema_for_val else model
        val_results = validate(val_model, val_loader, criterion, metrics, device)

        # Print which model is being used for validation
        if ema is not None and epoch < ema_warmup_epochs:
            val_model_name = "training model (EMA warmup)"
        elif ema is not None:
            val_model_name = "EMA model"
        else:
            val_model_name = "training model"

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_dice'].append(val_results['mean_dice'])
        history['val_iou'].append(val_results['mean_iou'])
        history['val_accuracy'].append(val_results['pixel_accuracy'])
        history['tumor_dice'].append(val_results['class_dice'].get('tumor', 0.0))
        history['lr'].append(current_lr)

        # Print results
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val [{val_model_name}]: Loss={val_results['loss']:.4f} | "
              f"Dice={val_results['mean_dice']:.4f} | "
              f"IoU={val_results['mean_iou']:.4f} | "
              f"Acc={val_results['pixel_accuracy']:.4f}")
        print(f"  Tumor Dice: {val_results['class_dice'].get('tumor', 0):.4f} | "
              f"Tumor IoU: {val_results['class_iou'].get('tumor', 0):.4f}")

        # Save checkpoint (use same model as validation)
        save_model = ema.ema_model if use_ema_for_val else model
        checkpoint.save(
            save_model, optimizer, epoch, val_results,
            scheduler=scheduler, config=config
        )

        # Get monitored metric value (supports nested keys like 'class_dice.tumor')
        def get_metric(results, key):
            if '.' in key:
                parts = key.split('.')
                val = results
                for p in parts:
                    val = val.get(p, {}) if isinstance(val, dict) else 0.0
                return float(val) if not isinstance(val, dict) else 0.0
            return results.get(key, 0.0)

        monitored_value = get_metric(val_results, monitor_metric)

        # Update scheduler
        if use_warmup_scheduler:
            scheduler.step()
        else:
            scheduler.step(monitored_value)

        # Early stopping
        if early_stopping and early_stopping(monitored_value):
            print("\nEarly stopping triggered!")
            break

    print("\n" + "=" * 60)
    print("Training complete!")

    # Save training curves
    plot_training_curves(history, save_path=save_dir / 'training_curves.png')

    # Load BEST model for final predictions (not the degraded final model)
    print("\nLoading best model for predictions...")
    best_path = weights_dir / 'best.pt'
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        # Create a fresh model for loading best weights (with DS heads to match state_dict)
        if model_type == 'attention_unet' or model_type == 'attention':
            best_model = AttentionUNet(**model_kwargs, deep_supervision=deep_supervision).to(device)
        else:
            best_model = UNet(**model_kwargs).to(device)
        best_model.load_state_dict(best_ckpt['model_state_dict'])
        best_model.eval()  # eval mode returns single output even with DS
        print(f"Loaded best model from epoch {best_ckpt.get('epoch', '?') + 1}")
    else:
        best_model = ema.ema_model if ema is not None else model
        best_model.eval()

    # Find samples WITH tumors for visualization
    print("Saving sample predictions...")
    tumor_images, tumor_masks = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            for i in range(images.size(0)):
                if masks[i].sum() > 0:  # Has tumor pixels
                    tumor_images.append(images[i])
                    tumor_masks.append(masks[i])
                if len(tumor_images) >= 8:
                    break
            if len(tumor_images) >= 8:
                break

    if len(tumor_images) > 0:
        tumor_images = torch.stack(tumor_images).to(device)
        tumor_masks = torch.stack(tumor_masks).to(device)
        with torch.no_grad():
            predictions = best_model(tumor_images)
        plot_predictions(
            tumor_images, tumor_masks, predictions,
            num_samples=min(4, len(tumor_images)),
            save_path=save_dir / 'val_predictions.png',
            class_names=['background', 'tumor']
        )
    else:
        print("Warning: No tumor samples found in validation set for visualization")

    # Final summary
    print(f"\nResults saved to: {save_dir}")
    print(f"Best model: {best_path}")

    best_tumor_dice = max(history['tumor_dice'])
    best_epoch = history['tumor_dice'].index(best_tumor_dice) + 1
    print(f"Best Tumor Dice: {best_tumor_dice:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()
