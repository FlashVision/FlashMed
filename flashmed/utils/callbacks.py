"""Training callbacks for monitoring and control."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: "min" for loss, "max" for metrics like AUC
        restore_best: Whether to restore best weights when stopping
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def __call__(self, metric: float, model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop.

        Args:
            metric: Current metric value
            model: Model to save best state

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric
            if model and self.restore_best:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        improved = (
            (self.mode == "max" and metric > self.best_score + self.min_delta) or
            (self.mode == "min" and metric < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = metric
            self.counter = 0
            if model and self.restore_best:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if model and self.restore_best and self.best_state:
                    model.load_state_dict(self.best_state)
                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric name to monitor
        mode: "min" or "max"
        save_best_only: Only save when metric improves
        save_last: Always save the last epoch
    """

    def __init__(
        self,
        save_dir: str = "workspace/checkpoints",
        monitor: str = "val_metric",
        mode: str = "max",
        save_best_only: bool = True,
        save_last: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.best_score = None

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ):
        """Check and potentially save a checkpoint."""
        current = metrics.get(self.monitor, 0.0)

        improved = self.best_score is None or (
            (self.mode == "max" and current > self.best_score) or
            (self.mode == "min" and current < self.best_score)
        )

        if improved:
            self.best_score = current
            self._save(model, optimizer, epoch, metrics, config, "best")

        if self.save_last:
            self._save(model, optimizer, epoch, metrics, config, "last")

        if not self.save_best_only:
            self._save(model, optimizer, epoch, metrics, config, f"epoch_{epoch}")

    def _save(self, model, optimizer, epoch, metrics, config, tag):
        path = self.save_dir / f"flashmed_{tag}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }, path)


class LearningRateLogger:
    """Track and log learning rate changes during training.

    Args:
        log_every: Log every N epochs
    """

    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self.history: list = []

    def __call__(self, optimizer: torch.optim.Optimizer, epoch: int):
        """Log current learning rate."""
        lr = optimizer.param_groups[0]["lr"]
        self.history.append({"epoch": epoch, "lr": lr})

        if epoch % self.log_every == 0:
            print(f"    LR: {lr:.2e}")

    def get_history(self) -> list:
        return self.history


class GradientMonitor:
    """Monitor gradient statistics during training for debugging."""

    def __init__(self, model: nn.Module, log_every: int = 10):
        self.model = model
        self.log_every = log_every
        self.step_count = 0

    def __call__(self) -> Optional[Dict[str, float]]:
        """Compute and optionally log gradient statistics."""
        self.step_count += 1

        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.data.norm(2).item())

        if not grad_norms:
            return None

        stats = {
            "grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "grad_norm_max": max(grad_norms),
            "grad_norm_min": min(grad_norms),
        }

        if self.step_count % self.log_every == 0:
            print(f"    Grad norm — mean: {stats['grad_norm_mean']:.4f}, "
                  f"max: {stats['grad_norm_max']:.4f}")

        return stats
