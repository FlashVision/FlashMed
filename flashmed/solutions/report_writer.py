"""Automated radiology report writer using vision-language models."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


REPORT_TEMPLATE = """RADIOLOGY REPORT
================
Modality: {modality}
Date: {date}

FINDINGS:
{findings}

IMPRESSION:
{impression}

RECOMMENDATION:
{recommendation}
"""


class ReportWriter:
    """Automated radiology report generation from medical images.

    Combines visual analysis with structured report templates to produce
    comprehensive radiology reports.

    Args:
        model_path: Path to trained VLM or classification model
        device: Inference device
        template: Report template format
        max_length: Maximum report length in tokens
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        template: Optional[str] = None,
        max_length: int = 256,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.template = template or REPORT_TEMPLATE
        self.max_length = max_length

        self.model, self.cfg = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str):
        from flashmed.models.flashmed_model import FlashMed
        from flashmed.cfg.config import FlashMedConfig

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg_dict = checkpoint.get("config", {})
        cfg = FlashMedConfig.from_dict(cfg_dict)

        model = FlashMed(
            task=cfg.task, num_classes=cfg.num_classes,
            pretrained=False, in_channels=cfg.in_channels, input_size=cfg.input_size,
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        return model.to(self.device), cfg

    def generate(self, source: str, modality: str = "CXR", patient_info: Optional[Dict] = None) -> str:
        """Generate a structured radiology report from an image.

        Args:
            source: Path to medical image or DICOM file
            modality: Imaging modality (CXR, CT, MRI)
            patient_info: Optional patient context

        Returns:
            Formatted radiology report string
        """
        image_tensor = self._preprocess(source)

        with torch.no_grad():
            if self.cfg.task == "report_gen":
                report_text = self._generate_from_vlm(image_tensor)
            else:
                report_text = self._generate_from_classification(image_tensor)

        from datetime import date
        formatted = self.template.format(
            modality=modality,
            date=date.today().isoformat(),
            findings=report_text["findings"],
            impression=report_text["impression"],
            recommendation=report_text["recommendation"],
        )

        return formatted

    def _preprocess(self, source: str) -> torch.Tensor:
        """Load and preprocess the input image."""
        from flashmed.data.transforms import get_medical_transforms

        if source.lower().endswith((".dcm", ".dicom")):
            from flashmed.data.dicom_utils import dicom_to_pil
            image = dicom_to_pil(source).convert("RGB")
        else:
            image = Image.open(source).convert("RGB")

        transform = get_medical_transforms("val", self.cfg.input_size, self.cfg.modality)
        return transform(image).unsqueeze(0).to(self.device)

    def _generate_from_vlm(self, image_tensor: torch.Tensor) -> Dict[str, str]:
        """Generate report using vision-language model."""
        if hasattr(self.model.backbone, "generate"):
            token_ids = self.model.backbone.generate(image_tensor, max_length=self.max_length)
            text = self._decode_tokens(token_ids[0])
            return self._parse_generated_text(text)
        return self._generate_from_classification(image_tensor)

    def _generate_from_classification(self, image_tensor: torch.Tensor) -> Dict[str, str]:
        """Generate report using classification model outputs."""
        outputs = self.model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

        from flashmed.data.datasets import ChestXray14Dataset
        labels = ChestXray14Dataset.PATHOLOGIES

        findings_list = []
        for prob, label in sorted(zip(probs, labels), reverse=True):
            if prob > 0.5:
                severity = "significant" if prob > 0.8 else "mild"
                findings_list.append(f"- {label}: {severity} ({prob:.0%} confidence)")

        if not findings_list:
            findings_text = "No acute cardiopulmonary abnormality identified."
            impression = "Normal examination."
            recommendation = "No immediate follow-up required."
        else:
            findings_text = "\n".join(findings_list)
            top_findings = [l for p, l in sorted(zip(probs, labels), reverse=True) if p > 0.5][:3]
            impression = f"Findings suggestive of: {', '.join(top_findings)}."
            recommendation = "Clinical correlation recommended. Consider follow-up imaging as clinically indicated."

        return {
            "findings": findings_text,
            "impression": impression,
            "recommendation": recommendation,
        }

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text (simple vocabulary mapping)."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
            return text
        except (ImportError, OSError):
            return " ".join(str(t) for t in token_ids.tolist() if t > 2)

    def _parse_generated_text(self, text: str) -> Dict[str, str]:
        """Parse free-text generation into structured sections."""
        sections = {"findings": "", "impression": "", "recommendation": ""}

        lower = text.lower()
        if "findings:" in lower:
            idx = lower.index("findings:")
            end_idx = lower.find("impression:", idx)
            sections["findings"] = text[idx + 9:end_idx if end_idx > 0 else len(text)].strip()
        else:
            sections["findings"] = text.strip()

        if "impression:" in lower:
            idx = lower.index("impression:")
            end_idx = lower.find("recommendation:", idx)
            sections["impression"] = text[idx + 11:end_idx if end_idx > 0 else len(text)].strip()
        else:
            sections["impression"] = "See findings above."

        if "recommendation:" in lower:
            idx = lower.index("recommendation:")
            sections["recommendation"] = text[idx + 15:].strip()
        else:
            sections["recommendation"] = "Clinical correlation recommended."

        return sections

    def batch_generate(self, image_dir: str, output_dir: Optional[str] = None, **kwargs) -> List[str]:
        """Generate reports for a directory of images."""
        reports = []
        img_dir = Path(image_dir)
        out_dir = Path(output_dir) if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".dcm"):
                report = self.generate(str(img_file), **kwargs)
                reports.append(report)

                if out_dir:
                    report_file = out_dir / f"{img_file.stem}_report.txt"
                    report_file.write_text(report)

        return reports
