"""Organ and lesion segmentation task (2D and 3D)."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import TASKS


class DiceLoss(nn.Module):
    """Generalized Dice Loss for medical image segmentation.

    Handles class imbalance through class-volume weighting.
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)

        if target.dim() == pred.dim() - 1:
            target_one_hot = torch.zeros_like(pred_soft)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        else:
            target_one_hot = target

        start_cls = 0 if self.include_background else 1
        dims = tuple(range(2, pred.dim()))

        intersection = (pred_soft[:, start_cls:] * target_one_hot[:, start_cls:]).sum(dims)
        cardinality = (pred_soft[:, start_cls:] + target_one_hot[:, start_cls:]).sum(dims)

        w = 1.0 / (target_one_hot[:, start_cls:].sum(dims) ** 2 + self.smooth)
        dice = (2.0 * w * intersection + self.smooth) / (w * cardinality + self.smooth)

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for hard example mining in segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


@TASKS.register("segmentation")
class SegmentationTask:
    """2D/3D medical image segmentation task.

    Supports organ segmentation, lesion delineation, and tumor boundaries
    with Dice + CE combined loss and multi-scale evaluation.

    Args:
        num_classes: Number of segmentation classes (including background)
        spatial_dims: 2 for slice-level, 3 for volumetric
        loss_type: "dice_ce", "dice", "focal"
        include_background: Whether to include background in Dice computation
    """

    def __init__(
        self,
        num_classes: int = 4,
        spatial_dims: int = 3,
        loss_type: str = "dice_ce",
        include_background: bool = False,
    ):
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims

        if loss_type == "dice_ce":
            self.dice_loss = DiceLoss(num_classes, include_background=include_background)
            self.ce_loss = nn.CrossEntropyLoss()
            self.criterion = lambda pred, target: 0.5 * self.dice_loss(pred, target) + 0.5 * self.ce_loss(pred, target)
        elif loss_type == "dice":
            self.criterion = DiceLoss(num_classes, include_background=include_background)
        elif loss_type == "focal":
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)

    def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute per-class Dice scores."""
        pred_mask = torch.argmax(pred, dim=1)
        scores = {}

        for cls in range(1, self.num_classes):
            pred_cls = (pred_mask == cls).float()
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            scores[f"dice_class_{cls}"] = dice.item()

        scores["dice_mean"] = np.mean(list(scores.values()))
        return scores

    def compute_hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor, percentile: float = 95.0) -> float:
        """Compute Hausdorff distance (95th percentile) between predicted and target boundaries."""
        pred_np = pred.cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)

        if not pred_np.any() or not target_np.any():
            return float("inf")

        from scipy.ndimage import distance_transform_edt
        pred_border = pred_np ^ distance_transform_edt(pred_np) > 1
        target_border = target_np ^ distance_transform_edt(target_np) > 1

        dt_pred = distance_transform_edt(~pred_border)
        dt_target = distance_transform_edt(~target_border)

        dist_pred_to_target = dt_target[pred_border]
        dist_target_to_pred = dt_pred[target_border]

        if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
            return float("inf")

        hd = max(np.percentile(dist_pred_to_target, percentile), np.percentile(dist_target_to_pred, percentile))
        return float(hd)

    def sliding_window_inference(
        self,
        model: nn.Module,
        volume: torch.Tensor,
        roi_size: Tuple[int, ...] = (96, 96, 96),
        overlap: float = 0.5,
    ) -> torch.Tensor:
        """Run inference on large volumes using sliding window approach."""
        model.eval()
        device = next(model.parameters()).device

        spatial_shape = volume.shape[2:]
        step = [int(r * (1 - overlap)) for r in roi_size]

        output = torch.zeros(volume.shape[0], self.num_classes, *spatial_shape, device=device)
        count = torch.zeros(1, 1, *spatial_shape, device=device)

        ranges = []
        for dim_size, r, s in zip(spatial_shape, roi_size, step):
            positions = list(range(0, dim_size - r + 1, s))
            if positions[-1] + r < dim_size:
                positions.append(dim_size - r)
            ranges.append(positions)

        import itertools
        for positions in itertools.product(*ranges):
            slices = tuple(slice(p, p + r) for p, r in zip(positions, roi_size))
            patch = volume[(slice(None), slice(None)) + slices].to(device)

            with torch.no_grad():
                pred = model(patch)

            output[(slice(None), slice(None)) + slices] += pred
            count[(slice(None), slice(None)) + slices] += 1

        output /= count.clamp(min=1)
        return output
