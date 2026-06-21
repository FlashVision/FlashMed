"""Validation engine for FlashMed models."""

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator:
    """Evaluate trained FlashMed models on validation/test sets.

    Args:
        model_path: Path to saved checkpoint
        data_dir: Path to validation data
        task: Task type
        device: Evaluation device
        batch_size: Batch size for evaluation
    """

    def __init__(
        self,
        model_path: str,
        data_dir: Optional[str] = None,
        task: str = "classification",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task = task
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.model, self.cfg = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str):
        from flashmed.models.flashmed_model import FlashMed
        from flashmed.cfg.config import FlashMedConfig

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = FlashMedConfig.from_dict(checkpoint.get("config", {"task": self.task}))

        model = FlashMed(
            task=cfg.task, num_classes=cfg.num_classes,
            pretrained=False, in_channels=cfg.in_channels, input_size=cfg.input_size,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model.to(self.device), cfg

    @torch.no_grad()
    def validate(self, data_dir: Optional[str] = None) -> Dict[str, float]:
        """Run full validation and return metrics.

        Returns:
            Dictionary of metric_name -> value
        """
        from flashmed.analytics.metrics import compute_auc_roc, compute_accuracy, compute_sensitivity, compute_specificity
        from flashmed.data.transforms import get_medical_transforms

        data_path = data_dir or self.data_dir or self.cfg.data_dir
        transform = get_medical_transforms("val", self.cfg.input_size, self.cfg.modality)

        from flashmed.data.datasets import ChestXray14Dataset
        dataset = ChestXray14Dataset(root=data_path, split="val", transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        all_preds, all_targets = [], []
        for images, targets in tqdm(loader, desc="Validating"):
            images = images.to(self.device)
            outputs = self.model(images)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)

        if not all_preds:
            return {"auc_roc": 0.0, "accuracy": 0.0}

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        metrics = {}
        if self.cfg.multi_label:
            metrics["auc_roc"] = compute_auc_roc(preds, targets)
            metrics["sensitivity"] = compute_sensitivity(preds, targets)
            metrics["specificity"] = compute_specificity(preds, targets)
        else:
            metrics["accuracy"] = compute_accuracy(preds, targets)

        print("\nValidation Results:")
        for name, val in metrics.items():
            print(f"  {name}: {val:.4f}")

        return metrics
