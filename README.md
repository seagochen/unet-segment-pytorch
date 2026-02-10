# UNet Lung Tumor Segmentation (PyTorch)

A PyTorch implementation of **Attention U-Net** for lung tumor segmentation on CT images.

**Best Result: Tumor Dice 0.8080** (AttentionUNet + DiceBCE loss + Cosine Annealing)

## Architecture

```
Input (1, 512, 512)
    |
[Encoder]                          [Decoder]
DoubleConv ─── 64 ──── AG ──── AttentionUp ── 64 ─── OutConv(n_classes)
    |                                  |
  Down ────── 128 ──── AG ──── AttentionUp ─ 128
    |                                  |
  Down ────── 256 ──── AG ──── AttentionUp ─ 256
    |                                  |
  Down ────── 512 ──── AG ──── AttentionUp ─ 512
    |                                  |
  Down ──── Bottleneck (512) ──────────┘
```

Key features:
- **Attention Gates** on skip connections for focusing on small tumors
- **DiceBCE Loss** (Balanced CE + Dice) for stable training with extreme class imbalance
- **Gradient Accumulation** for large effective batch sizes on limited VRAM
- **Cosine Annealing LR** for smooth convergence

## Project Structure

```
unet-segment-pytorch/
├── unet/                           # Main package
│   ├── models/
│   │   ├── layers.py               # DoubleConv, Down, Up, AttentionGate, AttentionUp
│   │   └── unet.py                 # UNet, AttentionUNet
│   ├── data/
│   │   ├── dataset.py              # LungTumorDataset (volume-based split)
│   │   └── augmentations.py        # Albumentations 2.0 pipelines
│   └── utils/
│       ├── loss.py                 # DiceLoss, BalancedCELoss, DiceBCELoss, DeepSupervisionLoss
│       ├── metrics.py              # SegmentationMetrics (IoU, Dice, PixelAccuracy)
│       ├── general.py              # Seeds, device, config, ModelEMA
│       ├── callbacks.py            # EarlyStopping, ModelCheckpoint
│       └── plots.py                # Training curves, prediction visualization
├── configs/
│   └── lung_tumor.yaml             # Training configuration
├── scripts/
│   ├── train.py                    # Training with gradient accumulation
│   ├── predict.py                  # Inference (single image or directory)
│   └── overfit_test.py             # Sanity check on small sample set
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
- `val_predictions.png` - Sample predictions

Key training parameters (see `configs/lung_tumor.yaml`):

| Parameter | Value | Note |
|-----------|-------|------|
| Model | Attention U-Net | Without deep supervision |
| Input size | 512x512 | Grayscale CT |
| Batch size | 4 | Effective 32 via gradient accumulation |
| Optimizer | AdamW | lr=5e-5, weight_decay=1e-4 |
| Loss | DiceBCE | Balanced CE + Dice, stable for small targets |
| Scheduler | Cosine Annealing | min_lr=1e-6 |
| Early stopping | patience=30 | Monitors Tumor Dice |

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
- **Augmentation** (training only): HorizontalFlip, Affine, ElasticTransform, GridDistortion, BrightnessContrast, GaussNoise, CoarseDropout
- **Class imbalance**: Tumor pixels are ~0.36% of each image. Addressed via balanced per-pixel CE weighting in DiceBCE loss

## Loss Functions

| Loss | Description |
|------|-------------|
| `dice_bce` | Balanced CE + Dice (recommended, most stable) |
| `dice` | Dice Loss (foreground only) |
| `ce` | Cross-Entropy |
| `balanced_ce` | Per-pixel balanced CE (dynamic weighting by class area) |

Configure in `configs/lung_tumor.yaml` via `loss.type`.

## Training Optimization History

This project went through 4 stages of optimization:

| Stage | Best Tumor Dice | Key Change |
|-------|----------------|------------|
| 1 | 0.7480 | Fix EMA/augmentation/lr issues |
| 2 | 0.7815 | Cosine LR (but FocalTversky caused instability) |
| 3 | **0.8080** | **DiceBCE loss — stable, no catastrophic drops** |
| 4 | 0.7968 | LR warmup (no improvement over Stage 3) |

Key lessons:
- **DiceBCE** is the most stable loss for small-target segmentation
- **EMA** causes tumor dice collapse for this task (fragile decision boundary)
- **Deep Supervision** adds noise for small targets (low-res heads can't represent small tumors)
- **Val oscillation is inherent**: Tumor Dice swings 0.1-0.2 between epochs due to tiny tumor size

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) (Oktay et al., 2018)

## License

GPL-3.0
