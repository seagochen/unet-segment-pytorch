#!/usr/bin/env python3
"""
UNet Inference Script for Lung Tumor Segmentation.

Usage:
    # Predict on a single image
    python scripts/predict.py --weights runs/lung_tumor/weights/best.pt --source image.png

    # Predict on a directory
    python scripts/predict.py --weights best.pt --source ./test_images/

    # Save with overlay visualization
    python scripts/predict.py --weights best.pt --source image.png --save-overlay
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unet.models import UNet
from unet.utils.general import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained UNet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, default='./predictions',
                        help='Output directory')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for tumor class')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use')
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save overlay visualization')
    parser.add_argument('--no-save-mask', action='store_true',
                        help='Do not save predicted masks')

    return parser.parse_args()


def load_model(weights_path: str, device: torch.device) -> UNet:
    """
    Load trained model from checkpoint.

    Args:
        weights_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = UNet(
        n_channels=model_config.get('n_channels', 1),
        n_classes=model_config.get('n_classes', 2),
        bilinear=model_config.get('bilinear', True),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def preprocess_image(
    image_path: Path,
    img_size: int = 256,
    mean: float = 0.5,
    std: float = 0.5
) -> tuple:
    """
    Preprocess image for inference.

    Args:
        image_path: Path to input image
        img_size: Target size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Tuple of (preprocessed tensor, original image array)
    """
    # Load image
    image = Image.open(image_path).convert('L')
    original_size = image.size

    # Resize
    image_resized = image.resize((img_size, img_size), Image.BILINEAR)

    # Convert to numpy and normalize
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_normalized = (image_array - mean) / std

    # Convert to tensor (1, 1, H, W)
    tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).float()

    # Keep original for visualization
    original_array = np.array(image, dtype=np.float32) / 255.0

    return tensor, original_array, original_size


def postprocess_mask(
    prediction: torch.Tensor,
    original_size: tuple,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Postprocess model output to binary mask.

    Args:
        prediction: Model output (1, C, H, W)
        original_size: Original image size (W, H)
        threshold: Confidence threshold

    Returns:
        Binary mask array at original resolution
    """
    # Get probabilities for tumor class (class 1)
    probs = torch.softmax(prediction, dim=1)
    tumor_prob = probs[0, 1].cpu().numpy()

    # Apply threshold
    mask = (tumor_prob > threshold).astype(np.uint8) * 255

    # Resize to original size
    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize(original_size, Image.NEAREST)

    return np.array(mask_resized)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple = (255, 0, 0)
) -> np.ndarray:
    """
    Create overlay visualization.

    Args:
        image: Original image (H, W) grayscale
        mask: Binary mask (H, W)
        alpha: Overlay transparency
        color: RGB color for mask overlay

    Returns:
        RGB overlay image
    """
    # Convert grayscale to RGB
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    image_rgb = np.stack([image] * 3, axis=-1)

    # Create colored overlay
    overlay = image_rgb.copy()
    mask_bool = mask > 127

    for i, c in enumerate(color):
        overlay[:, :, i][mask_bool] = c

    # Blend
    result = (1 - alpha) * image_rgb + alpha * overlay
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


@torch.no_grad()
def predict_single(
    model: UNet,
    image_path: Path,
    device: torch.device,
    img_size: int = 256,
    threshold: float = 0.5
) -> tuple:
    """
    Run prediction on a single image.

    Args:
        model: Trained model
        image_path: Path to input image
        device: Device
        img_size: Input size for model
        threshold: Confidence threshold

    Returns:
        Tuple of (mask, original_image, tumor_ratio)
    """
    # Preprocess
    tensor, original, original_size = preprocess_image(image_path, img_size)
    tensor = tensor.to(device)

    # Predict
    output = model(tensor)

    # Postprocess
    mask = postprocess_mask(output, original_size, threshold)

    # Calculate tumor ratio
    tumor_ratio = (mask > 127).sum() / mask.size

    return mask, original, tumor_ratio


def main():
    """Main inference function."""
    args = parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.weights}")
    model = load_model(args.weights, device)
    print("Model loaded successfully")

    # Get input files
    source = Path(args.source)
    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        image_paths = list(source.glob('*.png')) + list(source.glob('*.jpg'))
        image_paths = sorted(image_paths)
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    if len(image_paths) == 0:
        print(f"No images found in {source}")
        return

    print(f"Found {len(image_paths)} images")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_overlay:
        overlay_dir = output_dir / 'overlays'
        overlay_dir.mkdir(exist_ok=True)

    # Process images
    results = []
    pbar = tqdm(image_paths, desc='Processing')

    for image_path in pbar:
        try:
            mask, original, tumor_ratio = predict_single(
                model, image_path, device,
                img_size=args.img_size,
                threshold=args.threshold
            )

            # Save mask
            if not args.no_save_mask:
                mask_path = output_dir / f"{image_path.stem}_mask.png"
                Image.fromarray(mask).save(mask_path)

            # Save overlay
            if args.save_overlay:
                overlay = create_overlay(original, mask)
                overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
                Image.fromarray(overlay).save(overlay_path)

            results.append({
                'image': image_path.name,
                'tumor_ratio': tumor_ratio,
                'has_tumor': tumor_ratio > 0,
            })

            pbar.set_postfix({'tumor': f'{tumor_ratio*100:.1f}%'})

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 50)
    print("Inference complete!")
    print("=" * 50)

    num_with_tumor = sum(1 for r in results if r['has_tumor'])
    print(f"Total images: {len(results)}")
    print(f"Images with tumor: {num_with_tumor} ({100*num_with_tumor/len(results):.1f}%)")

    if results:
        avg_tumor = np.mean([r['tumor_ratio'] for r in results if r['has_tumor']] or [0])
        print(f"Average tumor coverage: {avg_tumor*100:.2f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
