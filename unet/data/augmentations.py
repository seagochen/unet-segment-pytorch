"""
Data augmentation pipelines for medical image segmentation.

Uses albumentations library for efficient augmentations that are applied
consistently to both image and mask.

Medical imaging augmentations are conservative to preserve anatomical structure:
- Horizontal flip (anatomically valid for CT)
- Small rotations (up to 15 degrees)
- Elastic deformation (mimics tissue variability)
- Brightness/contrast adjustments (mimics scanner variability)
"""

from typing import Tuple, Optional

import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_train_transforms(
    img_size: int = 256,
    mean: float = 0.5,
    std: float = 0.5,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    rotation_limit: int = 15,
    p_elastic: float = 0.3,
    p_brightness: float = 0.3,
) -> Optional["A.Compose"]:
    """
    Get training augmentation pipeline.

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        p_flip: Probability of horizontal flip
        p_rotate: Probability of rotation
        rotation_limit: Maximum rotation angle in degrees
        p_elastic: Probability of elastic deformation
        p_brightness: Probability of brightness/contrast adjustment

    Returns:
        Albumentations Compose object or None if library not available
    """
    if not ALBUMENTATIONS_AVAILABLE:
        print("Warning: albumentations not installed. Using basic transforms.")
        return None

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=0.3),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.85, 1.15),
            rotate=(-rotation_limit, rotation_limit),
            p=0.5,
            border_mode=0,
        ),
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=p_elastic,
            border_mode=0,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3, border_mode=0),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=p_brightness,
        ),
        A.GaussNoise(std_range=(0.01, 0.02), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.03, 0.06),
            hole_width_range=(0.03, 0.06),
            fill=0,
            p=0.1,
        ),
        A.Normalize(mean=[mean], std=[std]),
        ToTensorV2(),
    ])


def get_val_transforms(
    img_size: int = 256,
    mean: float = 0.5,
    std: float = 0.5,
) -> Optional["A.Compose"]:
    """
    Get validation/test augmentation pipeline (no augmentation, only normalization).

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations Compose object or None if library not available
    """
    if not ALBUMENTATIONS_AVAILABLE:
        print("Warning: albumentations not installed. Using basic transforms.")
        return None

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[mean], std=[std]),
        ToTensorV2(),
    ])


def apply_basic_transforms(
    image: np.ndarray,
    mask: np.ndarray,
    img_size: int = 256,
    mean: float = 0.5,
    std: float = 0.5,
    is_train: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply basic transforms without albumentations (fallback).

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Input mask (H, W)
        img_size: Target size
        mean: Normalization mean
        std: Normalization std
        is_train: Whether this is for training (enables augmentation)

    Returns:
        Tuple of (transformed_image, transformed_mask)
    """
    import torch
    from PIL import Image

    # Convert to PIL for resizing
    if image.ndim == 3:
        image = image[:, :, 0]  # Take first channel if multi-channel

    img_pil = Image.fromarray((image * 255).astype(np.uint8))
    mask_pil = Image.fromarray(mask.astype(np.uint8))

    # Resize
    img_pil = img_pil.resize((img_size, img_size), Image.BILINEAR)
    mask_pil = mask_pil.resize((img_size, img_size), Image.NEAREST)

    # Convert back to numpy
    image = np.array(img_pil, dtype=np.float32) / 255.0
    mask = np.array(mask_pil, dtype=np.int64)

    # Apply random horizontal flip for training
    if is_train and np.random.rand() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    # Normalize
    image = (image - mean) / std

    # Convert to tensor format (C, H, W)
    image = torch.from_numpy(image).unsqueeze(0).float()
    mask = torch.from_numpy(mask).long()

    return image, mask
