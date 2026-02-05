# UNet Lung Tumor Segmentation (PyTorch)

A PyTorch implementation of **Attention U-Net** with **Deep Supervision** for lung tumor segmentation on CT images.

## Architecture

```
Input (1, 512, 512)
    |
[Encoder]                          [Decoder]
DoubleConv ─── 64 ──── AG ──── AttentionUp ── 64 ─── OutConv(n_classes)
    |                                  |                     |
  Down ────── 128 ──── AG ──── AttentionUp ─ 128    DS Head (1/2 res)
    |                                  |                     |
  Down ────── 256 ──── AG ──── AttentionUp ─ 256    DS Head (1/4 res)
    |                                  |                     |
  Down ────── 512 ──── AG ──── AttentionUp ─ 512    DS Head (1/8 res)
    |                                  |
  Down ──── Bottleneck (512) ──────────┘
```

Key features:
- **Attention Gates** on skip connections for focusing on small tumors
- **Deep Supervision** with auxiliary loss at 3 intermediate decoder scales
- **Balanced Focal Tversky Loss** to handle extreme class imbalance (~0.36% tumor pixels)
- **EMA (Exponential Moving Average)** for training stability
- **Gradient Accumulation** for large effective batch sizes on limited VRAM

## Project Structure

```
unet-segment-pytorch/
├── unet/                           # Main package
│   ├── models/
│   │   ├── layers.py               # DoubleConv, Down, Up, AttentionGate, AttentionUp
│   │   └── unet.py                 # UNet, AttentionUNet
│   ├── data/
│   │   ├── dataset.py              # LungTumorDataset (volume-based split)
│   │   └── augmentations.py        # Albumentations pipelines
│   └── utils/
│       ├── loss.py                 # DiceLoss, FocalTverskyLoss, BalancedFocalTverskyLoss, DeepSupervisionLoss
│       ├── metrics.py              # SegmentationMetrics (IoU, Dice, PixelAccuracy)
│       ├── general.py              # Seeds, device, checkpoint, ModelEMA
│       ├── callbacks.py            # EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
│       └── plots.py                # Training curves, prediction visualization
├── configs/
│   └── lung_tumor.yaml             # Training configuration
├── scripts/
│   ├── train.py                    # Training with gradient accumulation + deep supervision
│   └── predict.py                  # Inference (single image or directory)
├── toolkits/
│   ├── download_medical_segmentation.py   # Download dataset from Kaggle
│   └── convert_medical_segmentation.py    # Convert NIfTI to PNG slices
├── requirements.txt
└── setup.py
```

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Prepare Dataset

1. Download the [Medical Image Segmentation](https://www.kaggle.com/datasets/modaresimr/medical-image-segmentation) dataset from Kaggle:

```bash
python toolkits/download_medical_segmentation.py
```

2. Convert NIfTI volumes to PNG slices:

```bash
python toolkits/convert_medical_segmentation.py --task Task006_Lung --output ./dataset
```

The resulting directory structure:

```
dataset/
├── images/          # Grayscale CT slices (PNG)
│   ├── 0_slice_0001.png
│   └── ...
└── labels/          # Binary masks (PNG, 0=background, 255=tumor)
    ├── 0_slice_0001.png
    └── ...
```

### Train

```bash
python scripts/train.py --config configs/lung_tumor.yaml
```

Training outputs are saved to `runs/<experiment_name>/`:
- `weights/best.pt` - Best model (by Tumor Dice)
- `weights/last.pt` - Last checkpoint
- `training_curves.png` - Loss and metric plots
- `predictions.png` - Sample predictions

Key training parameters (see `configs/lung_tumor.yaml`):

| Parameter | Value | Note |
|-----------|-------|------|
| Model | Attention U-Net | With deep supervision |
| Input size | 512x512 | Grayscale CT |
| Batch size | 4 | Effective 32 via gradient accumulation |
| Optimizer | AdamW | lr=5e-5, weight_decay=1e-3 |
| Loss | Balanced Focal Tversky | CE + Focal Tversky, alpha=0.7/beta=0.3 |
| EMA decay | 0.99 | 10-epoch warmup |
| Early stopping | patience=50 | Monitors Tumor Dice |

### Inference

```bash
# Single image
python scripts/predict.py --weights runs/lung_tumor_ds512/weights/best.pt --source image.png

# Directory of images
python scripts/predict.py --weights best.pt --source ./test_images/ --save-overlay

# Custom threshold
python scripts/predict.py --weights best.pt --source image.png --threshold 0.3
```

## Data Handling

- **Volume-based split**: Train/val split by volume ID, not individual slices, to prevent data leakage
- **Augmentation** (training only): HorizontalFlip, VerticalFlip, ShiftScaleRotate, ElasticTransform, GridDistortion, BrightnessContrast, GaussNoise, CoarseDropout
- **Class imbalance**: Tumor pixels are ~0.36% of each image. Addressed via balanced per-pixel CE weighting + Focal Tversky loss

## Loss Functions

| Loss | Description |
|------|-------------|
| `dice` | Dice Loss (foreground only) |
| `ce` | Cross-Entropy |
| `combined` | CE + Dice |
| `focal` | Focal Loss |
| `focal_tversky` | Focal Tversky (alpha/beta control FN/FP trade-off) |
| `balanced_ce` | Per-pixel balanced CE (dynamic weighting by class area) |
| `balanced_focal_tversky` | Balanced CE + Focal Tversky (default) |

Configure in `configs/lung_tumor.yaml` via `loss.type`.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) (Oktay et al., 2018)
- [A Novel Focal Tversky Loss with Improved Attention U-Net for Lesion Segmentation](https://arxiv.org/abs/1810.07842) (Abraham & Khan, 2019)

## License

GPL-3.0
