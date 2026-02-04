"""
Visualization utilities for training and evaluation.

Includes:
- Training curve plotting
- Prediction visualization
- Confusion matrix plotting
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'val_dice', etc.
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics plot
    ax = axes[1]
    if 'val_dice' in history:
        ax.plot(history['val_dice'], label='Val Dice', linewidth=2)
    if 'val_iou' in history:
        ax.plot(history['val_iou'], label='Val IoU', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Visualize predictions alongside ground truth.

    Args:
        images: Input images (N, C, H, W)
        masks: Ground truth masks (N, H, W)
        predictions: Predicted masks (N, C, H, W) logits or (N, H, W) class indices
        num_samples: Number of samples to show
        save_path: Path to save the plot
        show: Whether to display the plot
        class_names: Optional list of class names
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return

    # Convert predictions to class indices if needed
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)

    # Move to CPU and numpy
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()

    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Image
        img = images[i]
        if img.shape[0] == 1:
            img = img[0]  # Grayscale
        else:
            img = img.transpose(1, 2, 0)  # RGB

        # Denormalize if needed (assuming mean=0.5, std=0.5)
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(masks[i], cmap='tab10', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Predicted mask
        axes[i, 2].imshow(predictions[i], cmap='tab10', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    # Add legend if class names provided
    if class_names:
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        patches = [mpatches.Patch(color=colors[i], label=name)
                   for i, name in enumerate(class_names)]
        fig.legend(handles=patches, loc='lower center', ncol=len(class_names))

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save the plot
        show: Whether to display the plot
        normalize: Whether to normalize the matrix
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return

    if normalize:
        # Normalize by row (true labels)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = confusion_matrix.astype(float) / row_sums
    else:
        cm = confusion_matrix

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_sample_with_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> None:
    """
    Plot image with mask overlay.

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Ground truth mask (H, W)
        prediction: Optional predicted mask (H, W)
        alpha: Overlay transparency
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return

    num_cols = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # Original image
    axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Image with ground truth overlay
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image.copy()

    # Create colored overlay for tumor
    overlay = img_rgb.copy()
    overlay[mask > 0] = [1, 0, 0]  # Red for tumor
    blended = (1 - alpha) * img_rgb + alpha * overlay

    axes[1].imshow(blended)
    axes[1].set_title('Ground Truth Overlay')
    axes[1].axis('off')

    # Image with prediction overlay
    if prediction is not None:
        overlay = img_rgb.copy()
        overlay[prediction > 0] = [0, 1, 0]  # Green for predicted tumor
        blended = (1 - alpha) * img_rgb + alpha * overlay

        axes[2].imshow(blended)
        axes[2].set_title('Prediction Overlay')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()
