"""
General utility functions.

Includes:
- Random seed setting for reproducibility
- Device detection
- Configuration loading
- ModelEMA for exponential moving average
"""

import random
from pathlib import Path
from typing import Dict, Any, Union

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
        # benchmark=True is faster for fixed-size inputs (e.g. 512x512)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average of the training model's parameters.
    This helps stabilize training and often produces better final models.

    Reference:
        - Mean teachers are better role models (https://arxiv.org/abs/1703.01780)

    Args:
        model: The model to track
        decay: EMA decay rate (0.999 is common, closer to 1 = more smoothing)
        warmup_steps: Number of steps before EMA starts (use 0 to start immediately)

    Example:
        >>> ema = ModelEMA(model, decay=0.999)
        >>> for batch in dataloader:
        ...     loss = criterion(model(x), y)
        ...     loss.backward()
        ...     optimizer.step()
        ...     ema.update(model)  # Update EMA after each step
        >>> # Use EMA model for validation
        >>> val_results = validate(ema.ema_model, val_loader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

        # Create EMA model (deep copy)
        import copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        """
        Update EMA parameters.

        Args:
            model: The training model with updated parameters
        """
        self.updates += 1

        # Compute effective decay (ramp up during warmup)
        if self.updates <= self.warmup_steps:
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        else:
            decay = self.decay

        # Update EMA parameters
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, ema_param in self.ema_model.named_parameters():
                if name in model_params:
                    model_param = model_params[name]
                    ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

            # Also update buffers (like BatchNorm running stats)
            model_buffers = dict(model.named_buffers())
            for name, ema_buffer in self.ema_model.named_buffers():
                if name in model_buffers:
                    model_buffer = model_buffers[name]
                    ema_buffer.data.copy_(model_buffer.data)

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            'ema_state_dict': self.ema_model.state_dict(),
            'decay': self.decay,
            'updates': self.updates,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.ema_model.load_state_dict(state_dict['ema_state_dict'])
        self.decay = state_dict.get('decay', self.decay)
        self.updates = state_dict.get('updates', 0)
