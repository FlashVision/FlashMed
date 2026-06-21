"""Model export engine for FlashMed (ONNX, TorchScript)."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Exporter:
    """Export trained FlashMed models to deployment formats.

    Args:
        model_path: Path to saved checkpoint
        task: Task type
        device: Device for export
    """

    def __init__(self, model_path: str, task: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.cfg = self._load_model(model_path, task)
        self.model.eval()

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

    def export(self, output: str = "model.onnx", format: str = "onnx", **kwargs) -> str:
        """Export model to specified format.

        Args:
            output: Output file path
            format: Export format ("onnx" or "torchscript")

        Returns:
            Path to exported model
        """
        if format == "onnx":
            return self._export_onnx(output, **kwargs)
        elif format == "torchscript":
            return self._export_torchscript(output)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_onnx(self, output: str, opset_version: int = 17, simplify: bool = False) -> str:
        """Export to ONNX format."""
        import onnx

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_shape = self._get_input_shape()
        dummy_input = torch.randn(*input_shape, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        if simplify:
            try:
                import onnxsim
                model_onnx = onnx.load(str(output_path))
                model_simplified, check = onnxsim.simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, str(output_path))
            except ImportError:
                pass

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX exported: {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    def _export_torchscript(self, output: str) -> str:
        """Export to TorchScript format."""
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_shape = self._get_input_shape()
        dummy_input = torch.randn(*input_shape, device=self.device)

        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  TorchScript exported: {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    def _get_input_shape(self) -> Tuple[int, ...]:
        """Determine input shape based on task."""
        if self.cfg.task == "segmentation" and self.cfg.spatial_dims == 3:
            roi = self.cfg.roi_size
            return (1, self.cfg.in_channels, roi[0], roi[1], roi[2])
        return (1, self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size)
