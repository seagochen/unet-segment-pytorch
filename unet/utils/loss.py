"""
Loss functions for image segmentation.

Includes:
- DiceLoss: Based on Dice coefficient, handles class imbalance well
- FocalLoss: Focuses on hard examples, good for imbalanced data
- CombinedLoss: Weighted combination of CrossEntropy and Dice
"""

from typing import Optional, List

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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses learning on hard, misclassified examples by down-weighting
    easy examples.

    Args:
        alpha: Class weights (can be a list or tensor)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.

        Args:
            predictions: Model output logits (N, C, H, W)
            targets: Ground truth labels (N, H, W)

        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')

        # Compute probabilities
        probs = F.softmax(predictions, dim=1)

        # Get probability of correct class
        targets_flat = targets.view(-1)
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, probs.shape[1])
        p_t = probs_flat[torch.arange(targets_flat.size(0)), targets_flat]
        p_t = p_t.view(targets.shape)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=predictions.device)[targets]
            focal_weight = alpha_t * focal_weight

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy and Dice Loss.

    Loss = ce_weight * CrossEntropy + dice_weight * Dice

    This combination leverages both pixel-wise classification (CE) and
    region-based overlap (Dice).

    Args:
        ce_weight: Weight for CrossEntropy loss
        dice_weight: Weight for Dice loss
        class_weights: Optional class weights for CrossEntropy
        smooth: Smoothing factor for Dice loss
        ignore_background: If True, Dice loss ignores background class
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
        smooth: float = 1.0,
        ignore_background: bool = True
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        # CrossEntropy with optional class weights
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(smooth=smooth, ignore_background=ignore_background)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predictions: Model output logits (N, C, H, W)
            targets: Ground truth labels (N, H, W)

        Returns:
            Combined loss value
        """
        # Move class weights to correct device
        if hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(predictions.device)

        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        return self.ce_weight * ce + self.dice_weight * dice


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced segmentation.

    Tversky index is a generalization of Dice that allows controlling
    the trade-off between false positives and false negatives.

    TI = TP / (TP + alpha*FN + beta*FP)
    Focal Tversky Loss = (1 - TI)^gamma

    When alpha > beta, false negatives are penalized more heavily,
    which helps detect small objects.

    Args:
        alpha: Weight for false negatives (default: 0.7)
        beta: Weight for false positives (default: 0.3)
        gamma: Focal parameter to focus on hard examples (default: 0.75)
        smooth: Smoothing factor

    Reference:
        Abraham & Khan, "A Novel Focal Tversky loss function with improved
        Attention U-Net for lesion segmentation"
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha  # Weight for FN
        self.beta = beta    # Weight for FP
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Tversky Loss for foreground class only.

        Args:
            predictions: Model output logits (N, C, H, W)
            targets: Ground truth labels (N, H, W)

        Returns:
            Focal Tversky loss value
        """
        # Get probabilities for tumor class (class 1)
        probs = F.softmax(predictions, dim=1)
        tumor_probs = probs[:, 1]  # (N, H, W)

        # Get tumor targets
        tumor_targets = (targets == 1).float()  # (N, H, W)

        # Flatten (use reshape instead of view for non-contiguous tensors)
        tumor_probs_flat = tumor_probs.reshape(-1)
        tumor_targets_flat = tumor_targets.reshape(-1)

        # True positives, false negatives, false positives
        tp = (tumor_probs_flat * tumor_targets_flat).sum()
        fn = ((1 - tumor_probs_flat) * tumor_targets_flat).sum()
        fp = (tumor_probs_flat * (1 - tumor_targets_flat)).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        # Focal Tversky loss
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky


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


class BalancedFocalTverskyLoss(nn.Module):
    """
    Combined Balanced CE + Focal Tversky Loss.

    Combines:
    1. Balanced CE: Ensures equal attention to tumor vs background
    2. Focal Tversky: Focuses on hard examples and penalizes false negatives

    Args:
        ce_weight: Weight for balanced CE component
        tversky_weight: Weight for Focal Tversky component
        class_weight: Tumor class weight for balanced CE (default 0.5)
        alpha: Tversky alpha (weight for false negatives)
        beta: Tversky beta (weight for false positives)
        gamma: Focal parameter
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        tversky_weight: float = 1.0,
        class_weight: float = 0.5,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight

        self.balanced_ce = BalancedCELoss(class_weight=class_weight)
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce = self.balanced_ce(predictions, targets)
        ft = self.focal_tversky(predictions, targets)

        return self.ce_weight * ce + self.tversky_weight * ft


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
    loss_type: str = 'combined',
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weights: Optional[List[float]] = None,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    balanced_class_weight: float = 0.5,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: One of 'dice', 'ce', 'combined', 'focal', 'focal_tversky',
                   'balanced_ce', 'balanced_focal_tversky'
        ce_weight: Weight for CE in combined loss
        dice_weight: Weight for Dice in combined loss
        class_weights: Optional class weights
        focal_gamma: Gamma for focal loss / focal tversky loss
        tversky_alpha: Alpha for Focal Tversky (weight for false negatives)
        tversky_beta: Beta for Focal Tversky (weight for false positives)
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
    elif loss_type == 'combined':
        return CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights
        )
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    elif loss_type == 'focal_tversky':
        return FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_gamma
        )
    elif loss_type == 'balanced_ce':
        return BalancedCELoss(class_weight=balanced_class_weight)
    elif loss_type == 'balanced_focal_tversky':
        return BalancedFocalTverskyLoss(
            ce_weight=ce_weight,
            tversky_weight=dice_weight,  # Reuse dice_weight for tversky_weight
            class_weight=balanced_class_weight,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_gamma,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
