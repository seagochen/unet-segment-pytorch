"""
Lung Tumor Segmentation Dataset.

Supports loading PNG images and masks from the converted lung tumor dataset.
Implements volume-based train/val split to prevent data leakage.
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .augmentations import apply_basic_transforms


class LungTumorDataset(Dataset):
    """
    Dataset for Lung Tumor Segmentation.

    Directory structure expected:
        root/
        ├── images/
        │   ├── 0_slice_0001.png
        │   ├── 0_slice_0002.png
        │   └── ...
        └── labels/
            ├── 0_slice_0001.png
            ├── 0_slice_0002.png
            └── ...

    The dataset splits by volume ID (e.g., "0", "1", "2") to prevent data leakage,
    ensuring all slices from the same volume stay in the same split.

    Args:
        root: Path to dataset directory containing 'images' and 'labels' subdirs
        split: One of 'train', 'val', 'test', or 'all'
        transform: Optional albumentations transform pipeline
        val_ratio: Fraction of volumes for validation (default: 0.2)
        test_ratio: Fraction of volumes for testing (default: 0.0)
        seed: Random seed for reproducible splits
        img_size: Target image size (used if transform is None)

    Example:
        >>> dataset = LungTumorDataset('./lung_tumor_segmentation', split='train')
        >>> image, mask = dataset[0]
        >>> print(image.shape, mask.shape)  # (1, 256, 256), (256, 256)
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        img_size: int = 256,
    ):
        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.img_size = img_size

        # Validate paths
        self.images_dir = self.root / 'images'
        self.labels_dir = self.root / 'labels'

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Get all image files
        all_files = sorted([f.name for f in self.images_dir.glob('*.png')])
        if len(all_files) == 0:
            raise ValueError(f"No PNG files found in {self.images_dir}")

        # Create volume-based split
        self.files = self._create_split(all_files)

        print(f"LungTumorDataset [{split}]: {len(self.files)} samples")

    def _create_split(self, all_files: List[str]) -> List[str]:
        """
        Create train/val/test split based on volume IDs.

        This ensures slices from the same volume stay together to prevent data leakage.
        """
        # Extract unique volume IDs (e.g., "0", "1", "2" from "0_slice_0001.png")
        volume_ids = list(set(f.split('_slice_')[0] for f in all_files))
        volume_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

        # Shuffle volumes
        random.seed(self.seed)
        shuffled_volumes = volume_ids.copy()
        random.shuffle(shuffled_volumes)

        # Calculate split indices
        n_volumes = len(shuffled_volumes)
        n_test = int(n_volumes * self.test_ratio)
        n_val = int(n_volumes * self.val_ratio)
        n_train = n_volumes - n_test - n_val

        # Assign volumes to splits
        train_volumes = set(shuffled_volumes[:n_train])
        val_volumes = set(shuffled_volumes[n_train:n_train + n_val])
        test_volumes = set(shuffled_volumes[n_train + n_val:])

        # Filter files based on split
        if self.split == 'train':
            target_volumes = train_volumes
        elif self.split == 'val':
            target_volumes = val_volumes
        elif self.split == 'test':
            target_volumes = test_volumes
        elif self.split == 'all':
            return all_files
        else:
            raise ValueError(f"Invalid split: {self.split}. Use 'train', 'val', 'test', or 'all'")

        return [f for f in all_files if f.split('_slice_')[0] in target_volumes]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            image: Tensor of shape (1, H, W) - grayscale CT slice
            mask: Tensor of shape (H, W) - binary segmentation mask (0=background, 1=tumor)
        """
        filename = self.files[idx]
        img_path = self.images_dir / filename
        mask_path = self.labels_dir / filename

        # Load image and mask
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)

        # Convert mask to binary (0 or 1)
        mask = (mask > 127).astype(np.int64)

        # Apply transforms
        if self.transform is not None:
            # Albumentations expects (H, W, C) for image
            image_hwc = np.expand_dims(image, axis=-1)
            transformed = self.transform(image=image_hwc, mask=mask)
            image = transformed['image']  # Already (C, H, W) tensor
            mask = transformed['mask']    # Already tensor
        else:
            # Fallback to basic transforms
            image, mask = apply_basic_transforms(
                image, mask,
                img_size=self.img_size,
                is_train=(self.split == 'train')
            )

        # Ensure mask is long type for CrossEntropyLoss
        if isinstance(mask, torch.Tensor):
            mask = mask.long()

        return image, mask

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample."""
        filename = self.files[idx]
        parts = filename.replace('.png', '').split('_slice_')
        return {
            'filename': filename,
            'volume_id': parts[0],
            'slice_id': int(parts[1]) if len(parts) > 1 else 0,
        }

    @property
    def class_names(self) -> List[str]:
        """Get class names."""
        return ['background', 'tumor']

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return 2


def create_dataloaders(
    root: str,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    img_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        root: Path to dataset directory
        batch_size: Batch size
        val_ratio: Validation split ratio
        img_size: Image size
        num_workers: Number of data loading workers
        seed: Random seed
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .augmentations import get_train_transforms, get_val_transforms

    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    train_dataset = LungTumorDataset(
        root=root,
        split='train',
        transform=train_transform,
        val_ratio=val_ratio,
        seed=seed,
        img_size=img_size,
    )

    val_dataset = LungTumorDataset(
        root=root,
        split='val',
        transform=val_transform,
        val_ratio=val_ratio,
        seed=seed,
        img_size=img_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
