"""Lesion and nodule detection task for medical imaging."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import TASKS


class AnchorGenerator:
    """Generate anchor boxes for lesion detection at multiple scales."""

    def __init__(
        self,
        sizes: List[float] = (16, 32, 64, 128, 256),
        aspect_ratios: List[float] = (0.5, 1.0, 2.0),
        strides: List[int] = (4, 8, 16, 32, 64),
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides

    def generate(self, feature_map_size: Tuple[int, int], stride: int, size: float) -> torch.Tensor:
        """Generate anchor boxes for a single feature map level."""
        fh, fw = feature_map_size
        anchors = []

        for y in range(fh):
            for x in range(fw):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                for ratio in self.aspect_ratios:
                    w = size * np.sqrt(ratio)
                    h = size / np.sqrt(ratio)
                    anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        return torch.tensor(anchors, dtype=torch.float32)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / (union + 1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """Non-maximum suppression."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = compute_iou(boxes[i:i + 1], boxes[remaining]).squeeze(0)
        mask = ious < iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)


@TASKS.register("detection")
class DetectionTask:
    """Lesion/nodule detection in medical images.

    Detects and localizes abnormalities with bounding boxes and confidence scores.

    Args:
        num_classes: Number of lesion types to detect
        score_threshold: Minimum confidence for a detection
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
    """

    def __init__(
        self,
        num_classes: int = 1,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.anchor_gen = AnchorGenerator()

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        bbox_preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss (classification + regression)."""
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits, targets[0].get("labels", torch.zeros_like(cls_logits))
        )
        bbox_loss = F.smooth_l1_loss(
            bbox_preds, targets[0].get("boxes", torch.zeros_like(bbox_preds))
        )
        return {"cls_loss": cls_loss, "bbox_loss": bbox_loss, "total": cls_loss + bbox_loss}

    def postprocess(
        self,
        cls_logits: torch.Tensor,
        bbox_preds: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process raw model outputs into detections."""
        batch_results = []
        batch_size = cls_logits.shape[0]

        for i in range(batch_size):
            scores = torch.sigmoid(cls_logits[i]).flatten()
            boxes = bbox_preds[i]

            mask = scores > self.score_threshold
            filtered_scores = scores[mask]
            filtered_boxes = boxes[mask]

            if filtered_boxes.numel() > 0:
                keep = nms(filtered_boxes, filtered_scores, self.nms_threshold)
                keep = keep[:self.max_detections]
                batch_results.append({
                    "boxes": filtered_boxes[keep],
                    "scores": filtered_scores[keep],
                })
            else:
                batch_results.append({
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                })

        return batch_results

    def compute_froc(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        fps_per_image: List[float] = (0.125, 0.25, 0.5, 1, 2, 4, 8),
    ) -> Dict[str, float]:
        """Compute Free-Response ROC (FROC) for lesion detection evaluation."""
        all_fps = []
        all_sensitivities = []
        total_lesions = sum(len(t["boxes"]) for t in targets)

        if total_lesions == 0:
            return {"froc_mean": 0.0}

        thresholds = np.linspace(0.0, 1.0, 50)
        for thresh in thresholds:
            tp_count = 0
            fp_count = 0

            for pred, target in zip(predictions, targets):
                pred_mask = pred["scores"] > thresh
                pred_boxes = pred["boxes"][pred_mask]
                gt_boxes = target["boxes"]

                if len(pred_boxes) == 0:
                    continue
                if len(gt_boxes) == 0:
                    fp_count += len(pred_boxes)
                    continue

                ious = compute_iou(pred_boxes, gt_boxes)
                matched = ious.max(dim=1).values > 0.5
                tp_count += matched.sum().item()
                fp_count += (~matched).sum().item()

            sensitivity = tp_count / total_lesions
            fps_avg = fp_count / len(predictions)
            all_sensitivities.append(sensitivity)
            all_fps.append(fps_avg)

        froc_values = []
        for target_fps in fps_per_image:
            sens_at_fps = 0.0
            for fps, sens in zip(all_fps, all_sensitivities):
                if fps <= target_fps:
                    sens_at_fps = max(sens_at_fps, sens)
            froc_values.append(sens_at_fps)

        return {"froc_mean": np.mean(froc_values), "froc_values": froc_values}
