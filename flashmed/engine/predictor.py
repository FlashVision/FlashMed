"""Prediction/inference engine for FlashMed models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image


class Predictor:
    """Run inference with trained FlashMed models.

    Supports single images, directories, and DICOM files.

    Args:
        model_path: Path to saved checkpoint
        task: Task type (auto-detected from checkpoint if available)
        device: Inference device
        threshold: Classification threshold for multi-label
    """

    def __init__(
        self,
        model_path: str,
        task: Optional[str] = None,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.model, self.cfg = self._load_model(model_path, task)
        self.model.eval()
        self.task = self.cfg.task

    def _load_model(self, path: str, task: Optional[str] = None):
        from flashmed.models.flashmed_model import FlashMed
        from flashmed.cfg.config import FlashMedConfig

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg_dict = checkpoint.get("config", {})
        if task:
            cfg_dict["task"] = task
        cfg = FlashMedConfig.from_dict(cfg_dict)

        model = FlashMed(
            task=cfg.task, num_classes=cfg.num_classes,
            pretrained=False, in_channels=cfg.in_channels, input_size=cfg.input_size,
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        return model.to(self.device), cfg

    def predict(self, source: str, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run prediction on image(s).

        Args:
            source: Path to image, DICOM file, or directory

        Returns:
            Prediction results (dict for single image, list for directory)
        """
        source_path = Path(source)

        if source_path.is_dir():
            results = []
            for img_file in sorted(source_path.iterdir()):
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".dcm", ".dicom"):
                    result = self._predict_single(str(img_file))
                    result["file"] = str(img_file)
                    results.append(result)
            return results

        return self._predict_single(source)

    def _predict_single(self, path: str) -> Dict[str, Any]:
        """Predict on a single image/DICOM."""
        image_tensor = self._load_and_preprocess(path)

        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))

        if self.task == "classification":
            return self._postprocess_classification(output)
        elif self.task == "segmentation":
            return self._postprocess_segmentation(output)
        elif self.task == "report_gen":
            return self._postprocess_report(output)
        return {"raw_output": output.cpu().numpy().tolist()}

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Load and preprocess image for inference."""
        from flashmed.data.transforms import get_medical_transforms

        if path.lower().endswith((".dcm", ".dicom")):
            from flashmed.data.dicom_utils import dicom_to_pil
            image = dicom_to_pil(path).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")

        transform = get_medical_transforms("val", self.cfg.input_size, self.cfg.modality)
        tensor = transform(image)
        return tensor.unsqueeze(0)

    def _postprocess_classification(self, output: torch.Tensor) -> Dict[str, Any]:
        """Process classification output into readable predictions."""
        if self.cfg.multi_label:
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            from flashmed.data.datasets import ChestXray14Dataset
            labels = ChestXray14Dataset.PATHOLOGIES

            findings = []
            for i, (prob, label) in enumerate(zip(probs, labels)):
                if prob > self.threshold:
                    findings.append({"label": label, "confidence": float(prob)})

            findings.sort(key=lambda x: x["confidence"], reverse=True)
            return {
                "findings": findings,
                "num_findings": len(findings),
                "all_probabilities": {lbl: float(p) for lbl, p in zip(labels, probs)},
            }
        else:
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            return {
                "predicted_class": top_idx,
                "confidence": float(probs[top_idx]),
                "all_probabilities": probs.tolist(),
            }

    def _postprocess_segmentation(self, output: torch.Tensor) -> Dict[str, Any]:
        """Process segmentation output."""
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        return {
            "segmentation_mask": pred_mask,
            "num_classes_found": len(np.unique(pred_mask)),
            "class_distribution": {int(c): float((pred_mask == c).mean()) for c in np.unique(pred_mask)},
        }

    def _postprocess_report(self, output: torch.Tensor) -> Dict[str, Any]:
        """Process report generation output."""
        token_ids = torch.argmax(output, dim=-1).squeeze(0).cpu().numpy()
        return {"generated_tokens": token_ids.tolist()}
