"""
General utility functions.

Includes:
- Random seed setting for reproducibility
- Device detection
- Configuration loading
- Checkpoint save/load
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = '') -> torch.device:
    """
    Get the best available device.

    Args:
        device: Specific device string ('cuda', 'cpu', 'cuda:0', etc.)
                If empty, automatically selects best available.

    Returns:
        torch.device object
    """
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Union[str, Path],
    scheduler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        metrics: Dictionary of current metrics
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        config: Optional configuration dictionary
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        strict: Whether to strictly enforce state_dict key matching

    Returns:
        Dictionary with checkpoint info (epoch, metrics, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
    }


def increment_path(path: Union[str, Path], exist_ok: bool = False) -> Path:
    """
    Increment path by adding a number suffix if it exists.

    E.g., runs/exp -> runs/exp2 -> runs/exp3 -> ...

    Args:
        path: Base path
        exist_ok: If True, return the path even if it exists

    Returns:
        Unique path
    """
    path = Path(path)

    if not path.exists() or exist_ok:
        return path

    # Try incrementing
    suffix = path.suffix
    stem = path.stem

    for n in range(2, 1000):
        new_path = path.parent / f"{stem}{n}{suffix}"
        if not new_path.exists():
            return new_path

    raise RuntimeError(f"Could not find unique path for {path}")


def colorstr(*args) -> str:
    """
    Return colored string for terminal output.

    Example:
        >>> print(colorstr('blue', 'bold', 'hello'))
    """
    colors = {
        'blue': '\033[34m',
        'cyan': '\033[36m',
        'green': '\033[32m',
        'red': '\033[31m',
        'yellow': '\033[33m',
        'bold': '\033[1m',
        'end': '\033[0m',
    }

    *style, text = args
    color_codes = ''.join(colors.get(s, '') for s in style)
    return f"{color_codes}{text}{colors['end']}"
