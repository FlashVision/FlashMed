"""Configuration system for FlashMed training and inference."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FlashMedConfig:
    """Unified configuration for all FlashMed tasks."""

    task: str = "classification"
    model_name: str = "med_vit"
    num_classes: int = 14
    input_size: int = 224
    in_channels: int = 3
    pretrained: bool = True

    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.1

    # Data
    data_dir: str = "data/"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_workers: int = 4

    # Medical-specific
    modality: str = "xray"
    dicom_input: bool = False
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    multi_label: bool = True

    # Segmentation-specific
    seg_num_classes: int = 4
    spatial_dims: int = 2
    roi_size: List[int] = field(default_factory=lambda: [224, 224])

    # LoRA
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Privacy
    federated: bool = False
    differential_privacy: bool = False
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0

    # Output
    save_dir: str = "workspace/train"
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42

    # Extra kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def save_yaml(self, path: str):
        """Save config as YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlashMedConfig":
        """Create config from dictionary, putting unknown keys in extra."""
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        cfg = cls(**known)
        cfg.extra = extra
        return cfg


def load_yaml_config(path: str) -> FlashMedConfig:
    """Load a YAML config file and return a FlashMedConfig."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return FlashMedConfig.from_dict(data)


def get_config(task: str = "classification", **overrides) -> FlashMedConfig:
    """Get a default config for the given task, with optional overrides."""
    defaults = {
        "classification": {"model_name": "med_vit", "num_classes": 14, "multi_label": True, "input_size": 224},
        "segmentation": {"model_name": "unet_3d", "seg_num_classes": 4, "spatial_dims": 3,
                         "roi_size": [96, 96, 96], "input_size": 96},
        "detection": {"model_name": "med_vit", "num_classes": 1, "multi_label": False, "input_size": 512},
        "report_gen": {"model_name": "med_vlm", "num_classes": 0, "input_size": 224},
        "pathology": {"model_name": "med_vit", "num_classes": 9, "input_size": 256, "multi_label": False},
    }

    task_defaults = defaults.get(task, {})
    task_defaults["task"] = task
    task_defaults.update(overrides)
    return FlashMedConfig.from_dict(task_defaults)
