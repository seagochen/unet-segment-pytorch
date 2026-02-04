"""
UNet architecture for image segmentation.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import DoubleConv, Down, Up, OutConv, AttentionUp


class UNet(nn.Module):
    """
    UNet architecture for semantic segmentation.

    The network consists of:
    - Encoder (contracting path): Captures context through downsampling
    - Decoder (expanding path): Enables precise localization through upsampling
    - Skip connections: Preserve spatial information from encoder to decoder

    Args:
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_classes: Number of output classes
        bilinear: If True, use bilinear upsampling; otherwise use transposed convolutions
        base_features: Number of features in the first layer (default: 64)

    Example:
        >>> model = UNet(n_channels=1, n_classes=2)
        >>> x = torch.randn(1, 1, 256, 256)
        >>> output = model(x)  # shape: (1, 2, 256, 256)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        bilinear: bool = True,
        base_features: int = 64
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)

        # Decoder
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)

        # Output
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output logits of shape (N, n_classes, H, W)
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits

    def get_num_params(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class AttentionUNet(nn.Module):
    """
    Attention U-Net for semantic segmentation.

    Adds attention gates to skip connections to help focus on relevant features,
    particularly useful for small object detection (like tumors).

    Reference:
        Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
        https://arxiv.org/abs/1804.03999

    Args:
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_classes: Number of output classes
        bilinear: If True, use bilinear upsampling
        base_features: Number of features in the first layer (default: 64)

    Example:
        >>> model = AttentionUNet(n_channels=1, n_classes=2)
        >>> x = torch.randn(1, 1, 256, 256)
        >>> output = model(x)  # shape: (1, 2, 256, 256)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        bilinear: bool = True,
        base_features: int = 64
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (same as UNet)
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)

        # Decoder with Attention Gates
        self.up1 = AttentionUp(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = AttentionUp(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = AttentionUp(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = AttentionUp(base_features * 2, base_features, bilinear)

        # Output
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output logits of shape (N, n_classes, H, W)
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with attention-weighted skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count the number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_unet(
    n_channels: int = 1,
    n_classes: int = 2,
    bilinear: bool = True,
    pretrained: Optional[str] = None
) -> UNet:
    """
    Factory function to create UNet model.

    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        bilinear: Whether to use bilinear upsampling
        pretrained: Path to pretrained weights (optional)

    Returns:
        UNet model instance
    """
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    return model
