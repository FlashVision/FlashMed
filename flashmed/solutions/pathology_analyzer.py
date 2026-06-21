"""Pathology analysis solution for whole slide image processing."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


class PathologyAnalyzer:
    """High-level pathology analyzer for whole slide images.

    Provides end-to-end WSI analysis with patch extraction, feature computation,
    MIL aggregation, and heatmap generation.

    Args:
        model_path: Path to trained pathology model
        device: Inference device
        patch_size: Size of patches to extract
        batch_size: Batch size for patch inference
        overlap: Overlap between adjacent patches
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        patch_size: int = 256,
        batch_size: int = 32,
        overlap: float = 0.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap = overlap

        self.model, self.cfg = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str):
        from flashmed.models.flashmed_model import FlashMed
        from flashmed.cfg.config import FlashMedConfig

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg_dict = checkpoint.get("config", {"task": "pathology", "num_classes": 9})
        cfg = FlashMedConfig.from_dict(cfg_dict)

        model = FlashMed(
            task="pathology", num_classes=cfg.num_classes,
            pretrained=False, in_channels=cfg.in_channels, input_size=cfg.input_size,
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        return model.to(self.device), cfg

    def analyze(self, slide_path: str, generate_heatmap: bool = True) -> Dict[str, Any]:
        """Analyze a whole slide image or large tissue section.

        Args:
            slide_path: Path to image file
            generate_heatmap: Whether to generate attention heatmap

        Returns:
            Analysis results including slide-level prediction and details
        """
        image = np.array(Image.open(slide_path).convert("RGB"))
        h, w = image.shape[:2]

        patches, coords = self._extract_patches(image)

        if not patches:
            return {
                "slide_path": slide_path,
                "status": "no_tissue",
                "message": "No tissue patches found in image",
            }

        features, patch_predictions = self._extract_features(patches)
        slide_prediction = self._aggregate_predictions(features, patch_predictions)

        result = {
            "slide_path": slide_path,
            "image_size": (h, w),
            "num_patches": len(patches),
            "slide_prediction": slide_prediction,
            "patch_statistics": self._compute_patch_stats(patch_predictions),
        }

        if generate_heatmap:
            heatmap = self._generate_heatmap(image.shape[:2], coords, patch_predictions)
            result["heatmap"] = heatmap

        return result

    def _extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract tissue patches from the image."""
        from flashmed.tasks.pathology import PathologyTask
        task = PathologyTask(patch_size=self.patch_size)

        h, w = image.shape[:2]
        step = int(self.patch_size * (1 - self.overlap))
        patches = []
        coords = []

        for y in range(0, h - self.patch_size + 1, step):
            for x in range(0, w - self.patch_size + 1, step):
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                if task._is_tissue(patch):
                    patches.append(patch)
                    coords.append((y, x))

        return patches, coords

    @torch.no_grad()
    def _extract_features(
        self, patches: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on patches to get features and predictions."""
        from flashmed.data.transforms import get_medical_transforms

        transform = get_medical_transforms("val", self.patch_size, "pathology")
        all_features = []
        all_preds = []

        for i in range(0, len(patches), self.batch_size):
            batch_patches = patches[i:i + self.batch_size]
            batch_tensors = []
            for patch in batch_patches:
                pil_img = Image.fromarray(patch)
                tensor = transform(pil_img)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(self.device)
            outputs = self.model(batch)

            if hasattr(self.model.backbone, "forward_features"):
                features = self.model.backbone.forward_features(batch)
                all_features.append(features.cpu())

            all_preds.append(outputs.cpu())

        predictions = torch.cat(all_preds, dim=0)
        features = torch.cat(all_features, dim=0) if all_features else predictions
        return features, predictions

    def _aggregate_predictions(
        self, features: torch.Tensor, patch_preds: torch.Tensor
    ) -> Dict[str, Any]:
        """Aggregate patch predictions to slide-level."""
        probs = torch.softmax(patch_preds, dim=1)
        mean_probs = probs.mean(dim=0)
        max_probs, _ = probs.max(dim=0)

        from flashmed.data.datasets import PathMNISTDataset
        labels = PathMNISTDataset.TISSUE_TYPES

        top_class = int(mean_probs.argmax())
        return {
            "predicted_class": top_class,
            "predicted_label": labels[top_class] if top_class < len(labels) else f"class_{top_class}",
            "mean_confidence": float(mean_probs[top_class]),
            "max_confidence": float(max_probs[top_class]),
            "class_distribution": {
                labels[i] if i < len(labels) else f"class_{i}": float(mean_probs[i])
                for i in range(len(mean_probs))
            },
        }

    def _compute_patch_stats(self, patch_preds: torch.Tensor) -> Dict[str, Any]:
        """Compute statistics over patch-level predictions."""
        probs = torch.softmax(patch_preds, dim=1)
        pred_classes = probs.argmax(dim=1)

        unique, counts = torch.unique(pred_classes, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(unique, counts)}

        return {
            "total_patches": len(patch_preds),
            "class_counts": class_counts,
            "max_confidence_mean": float(probs.max(dim=1).values.mean()),
            "entropy_mean": float((-probs * torch.log(probs + 1e-8)).sum(dim=1).mean()),
        }

    def _generate_heatmap(
        self,
        image_shape: Tuple[int, int],
        coords: List[Tuple[int, int]],
        patch_preds: torch.Tensor,
    ) -> np.ndarray:
        """Generate spatial prediction heatmap."""
        from flashmed.tasks.pathology import PathologyTask
        task = PathologyTask(patch_size=self.patch_size)

        probs = torch.softmax(patch_preds, dim=1)
        max_probs = probs.max(dim=1).values.numpy()

        return task.generate_heatmap(image_shape, coords, max_probs)
