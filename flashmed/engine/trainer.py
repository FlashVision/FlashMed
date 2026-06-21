"""Training engine for FlashMed medical AI models."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flashmed.cfg.config import FlashMedConfig, get_config


class Trainer:
    """Unified trainer for all FlashMed tasks.

    Supports classification (multi-label), segmentation (2D/3D),
    detection, report generation, and pathology analysis.

    Args:
        config: FlashMedConfig object (overrides other kwargs)
        task: Task type
        epochs: Training epochs
        batch_size: Batch size
        device: Device to train on
        data_dir: Path to dataset
        save_dir: Output directory
        learning_rate: Learning rate
        num_classes: Number of classes
        lora: Enable LoRA fine-tuning
        **kwargs: Additional config overrides
    """

    def __init__(
        self,
        config: Optional[FlashMedConfig] = None,
        task: str = "classification",
        epochs: int = 100,
        batch_size: int = 16,
        device: str = "cuda",
        data_dir: Optional[str] = None,
        save_dir: str = "workspace/train",
        learning_rate: float = 1e-4,
        num_classes: int = 14,
        lora: bool = False,
        **kwargs,
    ):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = get_config(
                task=task, epochs=epochs, batch_size=batch_size,
                device=device, data_dir=data_dir or "data/",
                save_dir=save_dir, learning_rate=learning_rate,
                num_classes=num_classes, lora=lora, **kwargs,
            )

        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_criterion()
        self.scaler = torch.amp.GradScaler("cuda") if self.cfg.mixed_precision and self.device.type == "cuda" else None

        self.best_metric = 0.0
        self.epoch = 0

    def _build_model(self) -> nn.Module:
        from flashmed.models.flashmed_model import FlashMed

        model = FlashMed(
            task=self.cfg.task,
            num_classes=self.cfg.num_classes,
            pretrained=self.cfg.pretrained,
            in_channels=self.cfg.in_channels,
            input_size=self.cfg.input_size,
        )

        if self.cfg.lora:
            from flashmed.models.lora import apply_lora
            model = apply_lora(model, rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha, dropout=self.cfg.lora_dropout)

        return model.to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay)
        return torch.optim.Adam(params, lr=self.cfg.learning_rate)

    def _build_scheduler(self):
        if self.cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)
        elif self.cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        return None

    def _build_criterion(self) -> nn.Module:
        if self.cfg.task == "classification":
            if self.cfg.multi_label:
                return nn.BCEWithLogitsLoss()
            return nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        elif self.cfg.task == "segmentation":
            return DiceCELoss(num_classes=self.cfg.seg_num_classes)
        elif self.cfg.task == "report_gen":
            return nn.CrossEntropyLoss(ignore_index=0)
        return nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

    def _build_dataloaders(self):
        from flashmed.data.transforms import get_medical_transforms

        train_transform = get_medical_transforms("train", self.cfg.input_size, self.cfg.modality)
        val_transform = get_medical_transforms("val", self.cfg.input_size, self.cfg.modality)

        train_dataset = self._get_dataset(self.cfg.train_split, train_transform)
        val_dataset = self._get_dataset(self.cfg.val_split, val_transform)

        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.num_workers,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.batch_size,
            shuffle=False, num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _get_dataset(self, split: str, transform):
        from flashmed.data.datasets import ChestXray14Dataset, PathMNISTDataset
        data_dir = self.cfg.data_dir

        if self.cfg.task == "classification" and self.cfg.modality == "xray":
            return ChestXray14Dataset(root=data_dir, split=split, transform=transform)
        elif self.cfg.task == "pathology":
            return PathMNISTDataset(root=data_dir, split=split, transform=transform)
        return ChestXray14Dataset(root=data_dir, split=split, transform=transform)

    def train(self):
        """Execute the full training loop."""
        print(f"\n{'='*60}")
        print(f"  FlashMed Training — {self.cfg.task}")
        print(f"{'='*60}")
        print(f"  Model:       {self.cfg.model_name}")
        print(f"  Device:      {self.device}")
        print(f"  Epochs:      {self.cfg.epochs}")
        print(f"  Batch size:  {self.cfg.batch_size}")
        print(f"  LR:          {self.cfg.learning_rate}")
        print(f"  LoRA:        {self.cfg.lora}")
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Params:      {params:,}")
        print(f"{'='*60}\n")

        train_loader, val_loader = self._build_dataloaders()

        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            train_loss = self._train_epoch(train_loader)
            val_metric = self._validate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{self.cfg.epochs} | "
                  f"Loss: {train_loss:.4f} | Metric: {val_metric:.4f} | LR: {lr:.2e}")

            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self._save_checkpoint("best")

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

        self._save_checkpoint("last")
        print(f"\n  Training complete. Best metric: {self.best_metric:.4f}")
        print(f"  Checkpoints saved to: {self.save_dir}")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                targets = batch.get("labels", batch.get("label"))
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        from flashmed.analytics.metrics import compute_auc_roc, compute_accuracy

        all_preds = []
        all_targets = []

        for batch in loader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                targets = batch.get("labels", batch.get("label"))
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

            outputs = self.model(images)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

        if not all_preds:
            return 0.0

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        if self.cfg.task == "classification" and self.cfg.multi_label:
            return compute_auc_roc(preds, targets)
        elif self.cfg.task == "segmentation":
            return self._compute_dice(preds, targets)
        return compute_accuracy(preds, targets)

    def _compute_dice(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        preds = torch.argmax(preds, dim=1)
        dice_scores = []
        for cls in range(1, self.cfg.seg_num_classes):
            pred_mask = (preds == cls).float()
            target_mask = (targets == cls).float()
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.append(dice.item())
        return np.mean(dice_scores) if dice_scores else 0.0

    def _save_checkpoint(self, tag: str):
        path = self.save_dir / f"flashmed_{tag}.pth"
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg.to_dict(),
        }, path)


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation tasks."""

    def __init__(self, num_classes: int = 4, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(pred, target)

        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred_soft)
        if target.dim() == pred.dim() - 1:
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        else:
            target_one_hot = target

        dims = tuple(range(2, pred.dim()))
        intersection = (pred_soft * target_one_hot).sum(dims)
        cardinality = (pred_soft + target_one_hot).sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
