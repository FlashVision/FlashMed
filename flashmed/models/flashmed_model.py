"""FlashMed unified model factory for all medical AI tasks."""


import torch
import torch.nn as nn

from flashmed.registry import MODELS


@MODELS.register("FlashMed")
class FlashMed(nn.Module):
    """Unified model entry point for FlashMed.

    Automatically selects the appropriate architecture based on the task.

    Args:
        task: One of "classification", "segmentation", "detection", "report_gen", "pathology"
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        in_channels: Number of input channels
        input_size: Input spatial resolution
        **kwargs: Additional architecture-specific arguments
    """

    def __init__(
        self,
        task: str = "classification",
        num_classes: int = 14,
        pretrained: bool = True,
        in_channels: int = 3,
        input_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.num_classes = num_classes

        if task in ("classification", "detection", "pathology"):
            from flashmed.models.architectures.med_vit import MedViT
            self.backbone = MedViT(
                num_classes=num_classes,
                in_channels=in_channels,
                img_size=input_size,
                pretrained=pretrained,
                multi_label=(task == "classification"),
                **kwargs,
            )
        elif task == "segmentation":
            from flashmed.models.architectures.unet_3d import UNet3D
            spatial_dims = kwargs.pop("spatial_dims", 3)
            self.backbone = UNet3D(
                in_channels=in_channels,
                num_classes=num_classes,
                spatial_dims=spatial_dims,
                **kwargs,
            )
        elif task == "report_gen":
            from flashmed.models.architectures.med_vlm import MedVLM
            self.backbone = MedVLM(
                in_channels=in_channels,
                img_size=input_size,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown task: {task}. Choose from: classification, segmentation, detection, report_gen, pathology")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone(x, **kwargs)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning only the head."""
        if hasattr(self.backbone, "freeze_backbone"):
            self.backbone.freeze_backbone()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", **kwargs) -> "FlashMed":
        """Load a pretrained FlashMed model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_cfg = checkpoint.get("config", {})
        model_cfg.update(kwargs)
        model = cls(pretrained=False, **model_cfg)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(device)
