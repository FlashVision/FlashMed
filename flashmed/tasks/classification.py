"""Multi-label disease classification task for medical imaging."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import TASKS


@TASKS.register("classification")
class ClassificationTask:
    """Multi-label disease classification from medical images.

    Supports binary cross-entropy with logits for multi-label classification
    (e.g., ChestX-ray14 with 14 possible pathologies per image) and
    standard cross-entropy for single-label tasks.

    Args:
        num_classes: Number of disease classes
        multi_label: Whether to use multi-label (BCE) or single-label (CE) loss
        threshold: Confidence threshold for positive prediction
        label_smoothing: Label smoothing factor
        class_weights: Optional per-class weights for imbalanced datasets
    """

    def __init__(
        self,
        num_classes: int = 14,
        multi_label: bool = True,
        threshold: float = 0.5,
        label_smoothing: float = 0.0,
        class_weights: Optional[List[float]] = None,
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.threshold = threshold
        self.label_smoothing = label_smoothing

        if multi_label:
            weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
            self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        if self.multi_label:
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            return self.criterion(logits, targets)
        return self.criterion(logits, targets)

    def compute_predictions(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert logits to predictions."""
        if self.multi_label:
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            return {"probabilities": probs, "predictions": preds}
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return {"probabilities": probs, "predictions": preds}

    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute classification metrics."""
        from flashmed.analytics.metrics import compute_auc_roc, compute_f1_score

        metrics = {}
        if self.multi_label:
            metrics["auc_roc"] = compute_auc_roc(logits, targets)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            metrics["f1"] = compute_f1_score(preds, targets)
        else:
            preds = torch.argmax(logits, dim=1)
            metrics["accuracy"] = (preds == targets).float().mean().item()
        return metrics

    @staticmethod
    def compute_class_weights(dataset, num_classes: int = 14) -> List[float]:
        """Compute class weights for imbalanced medical datasets."""
        class_counts = np.zeros(num_classes)
        total = len(dataset)
        for i in range(total):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            class_counts += label.astype(float)

        weights = total / (num_classes * class_counts + 1e-7)
        return weights.tolist()
