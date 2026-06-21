"""I/O utilities for saving/loading models and data."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metric: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    path: str = "checkpoint.pth",
):
    """Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metric: Best metric value
        config: Training config dict
        path: Output file path
    """
    ensure_dir(str(Path(path).parent))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": metric,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to map tensors to

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_predictions(predictions: Dict[str, Any], path: str):
    """Save prediction results as JSON."""
    ensure_dir(str(Path(path).parent))

    serializable = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = value.tolist()
        elif hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_predictions(path: str) -> Dict[str, Any]:
    """Load saved predictions from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """Get model size information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "param_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "total_size_mb": param_size_mb + buffer_size_mb,
    }
