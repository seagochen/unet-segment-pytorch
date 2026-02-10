"""
Loss functions for image segmentation.

Includes:
- DiceLoss: Based on Dice coefficient, handles class imbalance well
- BalancedCELoss: Per-image dynamic weighting for class balance
- DiceBCELoss: Combined BalancedCE + Dice (stable, recommended)
- DeepSupervisionLoss: Wrapper for multi-scale auxiliary outputs
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice coefficient

    This loss naturally handles class imbalance as it normalizes by the
    sum of predictions and targets.

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
        ignore_background: If True, only compute Dice for foreground classes
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
        ignore_background: bool = True
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_background = ignore_background

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            predictions: Model output logits (N, C, H, W)
            targets: Ground truth labels (N, H, W) with values in [0, C-1]

        Returns:
            Dice loss value
        """
        num_classes = predictions.shape[1]

        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Compute Dice coefficient per class
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Ignore background class (class 0) if specified
        if self.ignore_background and num_classes > 1:
            dice = dice[:, 1:]  # Only foreground classes

        # Average over classes and batch
        if self.reduction == 'mean':
            return 1.0 - dice.mean()
        elif self.reduction == 'sum':
            return (1.0 - dice).sum()
        else:  # 'none'
            return 1.0 - dice


class BalancedCELoss(nn.Module):
    """
    Balanced Cross-Entropy Loss with per-image dynamic weighting.

    For each image:
    - Total weight = 1.0
    - All tumor pixels share weight = class_weight (default 0.5)
    - All background pixels share weight = 1 - class_weight (default 0.5)

    This ensures tumor pixels get much higher individual weights when they are few,
    preventing the model from ignoring them.

    Args:
        class_weight: Weight for tumor class (0-1), background gets 1-class_weight
        smooth: Small value to prevent division by zero
    """

    def __init__(self, class_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.class_weight = class_weight
        self.smooth = smooth

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute balanced cross-entropy loss.

        Args:
            predictions: Model output logits (N, C, H, W)
            targets: Ground truth labels (N, H, W)

        Returns:
            Balanced CE loss value
        """
        N, C, H, W = predictions.shape
        total_pixels = H * W

        # Compute per-pixel cross entropy (no reduction)
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')  # (N, H, W)

        # Create per-pixel weights
        weights = torch.zeros_like(ce_loss)

        for i in range(N):
            target_i = targets[i]
            tumor_mask = (target_i == 1)
            bg_mask = (target_i == 0)

            num_tumor = tumor_mask.sum().float() + self.smooth
            num_bg = bg_mask.sum().float() + self.smooth

            # Weight per tumor pixel = class_weight / num_tumor_pixels
            # Weight per bg pixel = (1 - class_weight) / num_bg_pixels
            weights[i][tumor_mask] = self.class_weight / num_tumor
            weights[i][bg_mask] = (1 - self.class_weight) / num_bg

        # Weighted loss
        weighted_loss = (ce_loss * weights).sum() / N

        return weighted_loss


class DiceBCELoss(nn.Module):
    """
    Combined Balanced CE + Dice Loss.

    Stable loss for medical image segmentation:
    1. Balanced CE: Per-image dynamic weighting ensures equal attention to tumor vs background
    2. Dice: Region-level overlap metric focusing on foreground classes

    This combination provides smooth, stable gradients (from BCE) with
    region-level awareness (from Dice), avoiding the instability of
    focal-based losses.

    Args:
        ce_weight: Weight for balanced CE component
        dice_weight: Weight for Dice component
        class_weight: Tumor class weight for balanced CE (default 0.5)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weight: float = 0.5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.balanced_ce = BalancedCELoss(class_weight=class_weight)
        self.dice_loss = DiceLoss(ignore_background=True)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce = self.balanced_ce(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss wrapper.

    Computes weighted loss across multiple output scales from the decoder.
    During training, the model returns [main_output, ds1, ds2, ds3] where
    ds outputs are at intermediate decoder resolutions (upsampled to full size).

    Args:
        base_criterion: The base loss function to apply at each scale
        weights: Weights for [main, ds1, ds2, ds3] outputs
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        weights: list = None,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.weights = weights or [1.0, 0.4, 0.2, 0.1]

    def forward(
        self,
        predictions,
        targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(predictions, (list, tuple)):
            total_loss = 0.0
            for pred, w in zip(predictions, self.weights):
                # Auxiliary outputs are already upsampled to full resolution by the model
                total_loss += w * self.base_criterion(pred, targets)
            return total_loss
        else:
            # Single output (eval mode or no deep supervision)
            return self.base_criterion(predictions, targets)


def create_loss_function(
    loss_type: str = 'dice_bce',
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weights: Optional[list] = None,
    balanced_class_weight: float = 0.5,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: One of 'dice', 'ce', 'balanced_ce', 'dice_bce'
        ce_weight: Weight for CE component
        dice_weight: Weight for Dice component
        class_weights: Optional class weights for CE
        balanced_class_weight: Tumor class weight for balanced losses (default 0.5)

    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()

    if loss_type == 'dice':
        return DiceLoss(ignore_background=True)
    elif loss_type == 'ce' or loss_type == 'crossentropy':
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()
    elif loss_type == 'balanced_ce':
        return BalancedCELoss(class_weight=balanced_class_weight)
    elif loss_type == 'dice_bce':
        return DiceBCELoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weight=balanced_class_weight,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
