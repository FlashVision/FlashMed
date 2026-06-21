"""Medical imaging evaluation metrics (AUC-ROC, Dice, sensitivity, specificity, F1)."""

from typing import Optional

import numpy as np
import torch


def compute_auc_roc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean AUC-ROC across all classes (multi-label).

    Args:
        predictions: Model logits or probabilities [N, C]
        targets: Binary target labels [N, C]

    Returns:
        Mean AUC-ROC score across classes
    """
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)

    probs = torch.sigmoid(predictions).numpy() if predictions.requires_grad else torch.sigmoid(predictions).detach().numpy()
    targets_np = targets.numpy() if not targets.requires_grad else targets.detach().numpy()

    num_classes = probs.shape[1]
    auc_scores = []

    for cls in range(num_classes):
        y_true = targets_np[:, cls]
        y_score = probs[:, cls]

        if len(np.unique(y_true)) < 2:
            continue

        auc = _compute_single_auc(y_true, y_score)
        auc_scores.append(auc)

    return float(np.mean(auc_scores)) if auc_scores else 0.0


def _compute_single_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC for a single class using the trapezoidal rule."""
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    y_score_sorted = y_score[desc_indices]

    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.concatenate([distinct_indices, [len(y_true_sorted) - 1]])

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = threshold_indices + 1 - tps

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr = tps / total_pos
    fpr = fps / total_neg

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = _trapz(tpr, fpr)
    return float(auc)


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for single-label classification."""
    if predictions.dim() > 1:
        preds = predictions.argmax(dim=1)
    else:
        preds = predictions
    if targets.dim() > 1:
        targets = targets.argmax(dim=1)
    return float((preds == targets).float().mean())


def compute_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    include_background: bool = False,
) -> float:
    """Compute mean Dice score for segmentation.

    Args:
        predictions: Predicted logits [B, C, ...] or predicted masks [B, ...]
        targets: Ground truth masks [B, ...]
        num_classes: Number of segmentation classes
        include_background: Whether to include class 0 in mean

    Returns:
        Mean Dice score
    """
    if predictions.dim() > targets.dim():
        pred_mask = predictions.argmax(dim=1)
    else:
        pred_mask = predictions

    start_cls = 0 if include_background else 1
    dice_scores = []

    for cls in range(start_cls, num_classes):
        pred_cls = (pred_mask == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            dice_scores.append(1.0)
        else:
            dice = (2.0 * intersection) / (union + 1e-7)
            dice_scores.append(float(dice))

    return float(np.mean(dice_scores))


def compute_sensitivity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute sensitivity (recall / true positive rate) for multi-label classification.

    Sensitivity = TP / (TP + FN)
    """
    probs = torch.sigmoid(predictions)
    preds = (probs > threshold).float()

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()

    sensitivity = tp / (tp + fn + 1e-7)
    return float(sensitivity)


def compute_specificity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute specificity (true negative rate) for multi-label classification.

    Specificity = TN / (TN + FP)
    """
    probs = torch.sigmoid(predictions)
    preds = (probs > threshold).float()

    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()

    specificity = tn / (tn + fp + 1e-7)
    return float(specificity)


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute macro F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    if predictions.dim() > 1 and predictions.shape[1] > 1:
        probs = torch.sigmoid(predictions) if predictions.max() > 1.0 else predictions
        preds = (probs > threshold).float()
    else:
        preds = predictions

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return float(f1.mean())


def compute_cohen_kappa(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Cohen's Kappa for inter-rater agreement."""
    preds = predictions.argmax(dim=1).numpy() if predictions.dim() > 1 else predictions.numpy()
    targs = targets.numpy()

    num_classes = max(preds.max(), targs.max()) + 1
    confusion = np.zeros((num_classes, num_classes))
    for p, t in zip(preds, targs):
        confusion[int(p), int(t)] += 1

    n = confusion.sum()
    po = np.diag(confusion).sum() / n
    pe = (confusion.sum(axis=0) * confusion.sum(axis=1)).sum() / (n * n)

    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))
