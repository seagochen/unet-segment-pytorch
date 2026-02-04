"""
Evaluation metrics for image segmentation.

Implements:
- Pixel Accuracy: Fraction of correctly classified pixels
- IoU (Intersection over Union): Per-class and mean IoU
- Dice Coefficient: Per-class and mean Dice score
"""

from typing import Dict, List, Optional

import torch
import numpy as np


class SegmentationMetrics:
    """
    Computes segmentation evaluation metrics.

    Accumulates predictions over batches and computes final metrics.

    Args:
        num_classes: Number of segmentation classes
        class_names: Optional list of class names for reporting
        ignore_index: Optional index to ignore in computation

    Example:
        >>> metrics = SegmentationMetrics(num_classes=2)
        >>> for pred, target in dataloader:
        ...     metrics.update(pred, target)
        >>> results = metrics.compute()
        >>> print(f"Mean IoU: {results['mean_iou']:.4f}")
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        ignore_index: Optional[int] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index

        # Confusion matrix: (num_classes, num_classes)
        # confusion[i, j] = pixels with true class i predicted as class j
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: Model output logits (N, C, H, W) or class indices (N, H, W)
            targets: Ground truth labels (N, H, W)
        """
        # Convert logits to class predictions if needed
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        # Move to numpy
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Create mask for valid pixels
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]

        # Update confusion matrix
        for t, p in zip(targets, predictions):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated statistics.

        Returns:
            Dictionary with:
            - pixel_accuracy: Overall pixel accuracy
            - mean_iou: Mean IoU across classes
            - mean_dice: Mean Dice coefficient across classes
            - class_iou: Dict of per-class IoU
            - class_dice: Dict of per-class Dice
        """
        # Pixel accuracy = sum(diagonal) / sum(all)
        total = self.confusion_matrix.sum()
        if total == 0:
            return self._empty_results()

        correct = np.diag(self.confusion_matrix).sum()
        pixel_accuracy = correct / total

        # Per-class IoU and Dice
        class_iou = {}
        class_dice = {}

        for i in range(self.num_classes):
            # True positives
            tp = self.confusion_matrix[i, i]
            # False positives (predicted as i but actually other)
            fp = self.confusion_matrix[:, i].sum() - tp
            # False negatives (actually i but predicted as other)
            fn = self.confusion_matrix[i, :].sum() - tp

            # IoU = TP / (TP + FP + FN)
            iou_denom = tp + fp + fn
            iou = tp / iou_denom if iou_denom > 0 else 0.0

            # Dice = 2*TP / (2*TP + FP + FN)
            dice_denom = 2 * tp + fp + fn
            dice = 2 * tp / dice_denom if dice_denom > 0 else 0.0

            class_name = self.class_names[i]
            class_iou[class_name] = iou
            class_dice[class_name] = dice

        # Mean metrics (excluding classes with no samples)
        valid_ious = [v for v in class_iou.values() if v > 0]
        valid_dices = [v for v in class_dice.values() if v > 0]

        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        mean_dice = np.mean(valid_dices) if valid_dices else 0.0

        return {
            'pixel_accuracy': float(pixel_accuracy),
            'mean_iou': float(mean_iou),
            'mean_dice': float(mean_dice),
            'class_iou': class_iou,
            'class_dice': class_dice,
        }

    def _empty_results(self) -> Dict[str, float]:
        """Return empty results when no samples have been processed."""
        return {
            'pixel_accuracy': 0.0,
            'mean_iou': 0.0,
            'mean_dice': 0.0,
            'class_iou': {name: 0.0 for name in self.class_names},
            'class_dice': {name: 0.0 for name in self.class_names},
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix."""
        return self.confusion_matrix.copy()


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute IoU for each class.

    Args:
        predictions: Model output logits (N, C, H, W) or class indices (N, H, W)
        targets: Ground truth labels (N, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor

    Returns:
        Tensor of shape (num_classes,) with IoU for each class
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)

    ious = []
    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()

        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return torch.stack(ious)


def compute_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient for each class.

    Args:
        predictions: Model output logits (N, C, H, W) or class indices (N, H, W)
        targets: Ground truth labels (N, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor

    Returns:
        Tensor of shape (num_classes,) with Dice for each class
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)

    dices = []
    for cls in range(num_classes):
        pred_cls = (predictions == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dices.append(dice)

    return torch.stack(dices)
