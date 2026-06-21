"""Whole slide image (WSI) analysis for computational pathology."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import TASKS


@TASKS.register("pathology")
class PathologyTask:
    """Whole slide image analysis and tissue classification.

    Supports patch-level classification with multi-instance learning (MIL)
    aggregation for slide-level predictions.

    Args:
        num_classes: Number of tissue/disease classes
        patch_size: Size of patches extracted from WSIs
        magnification: Magnification level for analysis
        mil_aggregation: Aggregation method for MIL ("attention", "mean", "max")
    """

    def __init__(
        self,
        num_classes: int = 9,
        patch_size: int = 256,
        magnification: str = "20x",
        mil_aggregation: str = "attention",
    ):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.magnification = magnification
        self.mil_aggregation = mil_aggregation
        self.criterion = nn.CrossEntropyLoss()

        if mil_aggregation == "attention":
            self.attention_net = AttentionMIL(embed_dim=768, num_classes=num_classes)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)

    def extract_patches(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        overlap: float = 0.0,
    ) -> List[np.ndarray]:
        """Extract tissue patches from a whole slide image.

        Args:
            image: Full WSI or region as numpy array [H, W, C]
            mask: Optional tissue mask to filter background
            overlap: Fractional overlap between patches

        Returns:
            List of patch numpy arrays
        """
        h, w = image.shape[:2]
        step = int(self.patch_size * (1 - overlap))
        patches = []

        for y in range(0, h - self.patch_size + 1, step):
            for x in range(0, w - self.patch_size + 1, step):
                patch = image[y:y + self.patch_size, x:x + self.patch_size]

                if mask is not None:
                    patch_mask = mask[y:y + self.patch_size, x:x + self.patch_size]
                    tissue_ratio = patch_mask.mean()
                    if tissue_ratio < 0.5:
                        continue

                if self._is_tissue(patch):
                    patches.append(patch)

        return patches

    def _is_tissue(self, patch: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if a patch contains tissue (not background)."""
        if patch.ndim == 2:
            return patch.std() > 10

        gray = np.mean(patch, axis=2)
        white_ratio = (gray > 220).mean()
        black_ratio = (gray < 20).mean()

        return white_ratio < 0.8 and black_ratio < 0.8 and gray.std() > threshold * 255

    def aggregate_predictions(
        self,
        patch_features: torch.Tensor,
        patch_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate patch-level predictions to slide-level using MIL.

        Args:
            patch_features: Feature vectors for each patch [N, D]
            patch_logits: Optional patch-level class logits [N, C]

        Returns:
            Slide-level prediction dict
        """
        if self.mil_aggregation == "attention":
            slide_logits, attention_weights = self.attention_net(patch_features.unsqueeze(0))
            return {
                "logits": slide_logits.squeeze(0),
                "attention_weights": attention_weights.squeeze(0),
                "prediction": torch.argmax(slide_logits, dim=-1).item(),
            }
        elif self.mil_aggregation == "max":
            if patch_logits is not None:
                max_logits, _ = patch_logits.max(dim=0)
                return {"logits": max_logits, "prediction": torch.argmax(max_logits).item()}
            max_feat, _ = patch_features.max(dim=0)
            return {"features": max_feat}
        else:
            if patch_logits is not None:
                mean_logits = patch_logits.mean(dim=0)
                return {"logits": mean_logits, "prediction": torch.argmax(mean_logits).item()}
            mean_feat = patch_features.mean(dim=0)
            return {"features": mean_feat}

    def generate_heatmap(
        self,
        image_shape: Tuple[int, int],
        patch_coords: List[Tuple[int, int]],
        patch_scores: np.ndarray,
    ) -> np.ndarray:
        """Generate attention/prediction heatmap over the WSI.

        Args:
            image_shape: (H, W) of original image
            patch_coords: List of (y, x) top-left coordinates
            patch_scores: Score for each patch

        Returns:
            Heatmap array of shape (H, W) normalized to [0, 1]
        """
        heatmap = np.zeros(image_shape[:2], dtype=np.float32)
        count = np.zeros(image_shape[:2], dtype=np.float32)

        for (y, x), score in zip(patch_coords, patch_scores):
            heatmap[y:y + self.patch_size, x:x + self.patch_size] += score
            count[y:y + self.patch_size, x:x + self.patch_size] += 1

        count = np.maximum(count, 1)
        heatmap /= count

        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)

        return heatmap


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning aggregation.

    Learns which patches are most important for the slide-level prediction.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, num_classes: int = 9):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention aggregation.

        Args:
            patch_features: [B, N, D] batch of patch features

        Returns:
            Tuple of (logits [B, C], attention_weights [B, N])
        """
        attn_scores = self.attention(patch_features).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)

        slide_repr = torch.bmm(attn_weights.unsqueeze(1), patch_features).squeeze(1)
        logits = self.classifier(slide_repr)

        return logits, attn_weights
