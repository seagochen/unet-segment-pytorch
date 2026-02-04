"""
Training callbacks.

Includes:
- EarlyStopping: Stop training when metric stops improving
- ModelCheckpoint: Save best and last model checkpoints
"""

from pathlib import Path
from typing import Optional, Union

import torch


class EarlyStopping:
    """
    Early stopping callback.

    Stops training when the monitored metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
        verbose: Whether to print messages

    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     val_dice = validate(model)
        ...     if early_stop(val_dice):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda a, b: a < b - min_delta
        else:  # mode == 'max'
            self.is_better = lambda a, b: a > b + min_delta

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Model checkpoint callback.

    Saves the best model and optionally the last model.

    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor ('val_dice', 'val_loss', etc.)
        mode: 'min' for loss, 'max' for metrics like Dice
        save_last: Whether to also save the last checkpoint
        verbose: Whether to print messages

    Example:
        >>> checkpoint = ModelCheckpoint('./runs/exp1/weights', monitor='val_dice', mode='max')
        >>> for epoch in range(100):
        ...     metrics = validate(model)
        ...     checkpoint.save(model, optimizer, epoch, metrics)
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = 'mean_dice',
        mode: str = 'max',
        save_last: bool = True,
        verbose: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.verbose = verbose

        self.best_score = None

        if mode == 'min':
            self.is_better = lambda a, b: a < b
        else:  # mode == 'max'
            self.is_better = lambda a, b: a > b

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        scheduler: Optional[object] = None,
        config: Optional[dict] = None,
    ) -> bool:
        """
        Save checkpoint if current score is best.

        Args:
            model: Model to save
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Optional scheduler
            config: Optional config dictionary

        Returns:
            True if best model was updated, False otherwise
        """
        # Get current score
        current_score = metrics.get(self.monitor, metrics.get('mean_dice', 0.0))

        # Create checkpoint dict
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

        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / 'last.pt'
            torch.save(checkpoint, last_path)

        # Check if best
        is_best = False
        if self.best_score is None or self.is_better(current_score, self.best_score):
            self.best_score = current_score
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            is_best = True

            if self.verbose:
                print(f"Saved best model: {self.monitor}={current_score:.4f}")

        return is_best

    def load_best(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> dict:
        """
        Load the best checkpoint.

        Args:
            model: Model to load weights into
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        best_path = self.save_dir / 'best.pt'
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint


class ReduceLROnPlateau:
    """
    Wrapper around PyTorch's ReduceLROnPlateau with additional tracking.

    Args:
        optimizer: Optimizer to adjust
        mode: 'min' or 'max'
        factor: Factor to reduce LR by
        patience: Epochs to wait before reducing
        min_lr: Minimum learning rate
        verbose: Whether to print messages
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'max',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        self.num_reductions = 0
        self.last_lr = optimizer.param_groups[0]['lr']

    def step(self, metric: float) -> bool:
        """
        Step the scheduler.

        Args:
            metric: Current metric value

        Returns:
            True if learning rate was reduced
        """
        self.scheduler.step(metric)

        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        reduced = current_lr < self.last_lr

        if reduced:
            self.num_reductions += 1
            if self.verbose:
                print(f"Reducing learning rate to {current_lr:.2e}")
            self.last_lr = current_lr

        return reduced

    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_reductions': self.num_reductions,
            'last_lr': self.last_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        self.num_reductions = state_dict.get('num_reductions', 0)
        self.last_lr = state_dict.get('last_lr', self.last_lr)
