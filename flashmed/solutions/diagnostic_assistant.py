"""Diagnostic assistant for automated medical image analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class DiagnosticAssistant:
    """High-level diagnostic assistant combining classification with interpretability.

    Provides structured diagnostic reports with findings, confidence scores,
    GradCAM attention maps, and differential diagnosis suggestions.

    Args:
        model_path: Path to trained classification model
        device: Inference device
        threshold: Confidence threshold for positive findings
        modality: Default imaging modality
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        threshold: float = 0.5,
        modality: str = "xray",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.modality = modality

        from flashmed.engine.predictor import Predictor
        self.predictor = Predictor(model_path=model_path, task="classification", device=device, threshold=threshold)
        self.model = self.predictor.model

    def diagnose(
        self,
        source: str,
        modality: Optional[str] = None,
        include_gradcam: bool = True,
        include_differential: bool = True,
    ) -> Dict[str, Any]:
        """Run full diagnostic analysis on a medical image.

        Args:
            source: Path to image or DICOM file
            modality: Override default modality
            include_gradcam: Generate attention heatmaps
            include_differential: Include differential diagnosis

        Returns:
            Structured diagnostic report
        """
        modality = modality or self.modality
        prediction = self.predictor.predict(source)

        report = {
            "source": source,
            "modality": modality,
            "findings": prediction.get("findings", []),
            "num_abnormalities": prediction.get("num_findings", 0),
            "overall_status": "ABNORMAL" if prediction.get("num_findings", 0) > 0 else "NORMAL",
        }

        if include_gradcam and prediction.get("findings"):
            top_finding = prediction["findings"][0]["label"]
            heatmap = self._generate_gradcam(source, top_finding)
            report["attention_heatmap"] = heatmap
            report["attention_finding"] = top_finding

        if include_differential and prediction.get("findings"):
            report["differential_diagnosis"] = self._get_differential(prediction["findings"])

        report["confidence_summary"] = self._summarize_confidence(prediction)
        report["recommendation"] = self._generate_recommendation(report)

        return report

    def batch_diagnose(self, image_dir: str, **kwargs) -> List[Dict[str, Any]]:
        """Run diagnosis on a directory of images."""
        results = []
        img_dir = Path(image_dir)
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".dcm"):
                result = self.diagnose(str(img_file), **kwargs)
                results.append(result)
        return results

    def _generate_gradcam(self, source: str, target_class: str) -> np.ndarray:
        """Generate GradCAM heatmap for the target class."""
        from flashmed.utils.visualize import GradCAM

        image_tensor = self.predictor._load_and_preprocess(source).to(self.device)
        self.model.eval()

        from flashmed.data.datasets import ChestXray14Dataset
        class_idx = ChestXray14Dataset.PATHOLOGIES.index(target_class) if target_class in ChestXray14Dataset.PATHOLOGIES else 0

        gradcam = GradCAM(self.model)
        heatmap = gradcam.generate(image_tensor, class_idx=class_idx)
        return heatmap

    def _get_differential(self, findings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate differential diagnosis based on findings."""
        DIFFERENTIAL_MAP = {
            "Atelectasis": ["Mucus plug", "Endobronchial lesion", "Post-surgical"],
            "Cardiomegaly": ["Congestive heart failure", "Pericardial effusion", "Valvular disease"],
            "Effusion": ["Heart failure", "Pneumonia", "Malignancy", "Trauma"],
            "Infiltration": ["Pneumonia", "Pulmonary edema", "Hemorrhage"],
            "Mass": ["Primary lung cancer", "Metastasis", "Benign tumor"],
            "Nodule": ["Granuloma", "Primary cancer", "Metastasis", "Hamartoma"],
            "Pneumonia": ["Bacterial infection", "Viral infection", "Fungal infection"],
            "Pneumothorax": ["Spontaneous", "Traumatic", "Iatrogenic"],
            "Consolidation": ["Lobar pneumonia", "Aspiration", "Hemorrhage"],
            "Edema": ["Cardiogenic pulmonary edema", "ARDS", "Fluid overload"],
            "Emphysema": ["COPD", "Alpha-1 antitrypsin deficiency"],
            "Fibrosis": ["Idiopathic pulmonary fibrosis", "Drug-induced", "Radiation"],
            "Pleural_Thickening": ["Prior infection", "Asbestosis", "Post-inflammatory"],
            "Hernia": ["Hiatal hernia", "Diaphragmatic hernia"],
        }

        differentials = []
        for finding in findings[:3]:
            label = finding["label"]
            if label in DIFFERENTIAL_MAP:
                differentials.append({
                    "finding": label,
                    "confidence": f"{finding['confidence']:.1%}",
                    "differential": ", ".join(DIFFERENTIAL_MAP[label][:3]),
                })
        return differentials

    def _summarize_confidence(self, prediction: Dict[str, Any]) -> str:
        """Generate human-readable confidence summary."""
        findings = prediction.get("findings", [])
        if not findings:
            return "No significant abnormalities detected."

        high_conf = [f for f in findings if f["confidence"] > 0.8]
        med_conf = [f for f in findings if 0.5 <= f["confidence"] <= 0.8]

        parts = []
        if high_conf:
            labels = ", ".join(f["label"] for f in high_conf)
            parts.append(f"High confidence: {labels}")
        if med_conf:
            labels = ", ".join(f["label"] for f in med_conf)
            parts.append(f"Moderate confidence: {labels}")
        return "; ".join(parts)

    def _generate_recommendation(self, report: Dict[str, Any]) -> str:
        """Generate clinical recommendation based on findings."""
        if report["overall_status"] == "NORMAL":
            return "No immediate follow-up recommended based on imaging findings."

        num_findings = report["num_abnormalities"]
        if num_findings >= 3:
            return "Multiple abnormalities detected. Recommend urgent clinical correlation and specialist consultation."
        elif num_findings >= 1:
            return "Abnormality detected. Recommend clinical correlation and follow-up imaging as appropriate."
        return "Findings inconclusive. Consider additional imaging or clinical correlation."
