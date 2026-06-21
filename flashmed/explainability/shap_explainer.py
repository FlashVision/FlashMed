"""SHAP-based explainability for FlashMed models."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class SHAPExplainer:
    """SHAP wrapper for feature importance analysis of medical imaging models.

    Provides unified access to DeepSHAP, GradientSHAP, and PartitionSHAP
    methods.  The ``shap`` library is imported lazily so the rest of FlashMed
    can be used without it installed.

    Args:
        model: PyTorch model to explain.
        method: SHAP algorithm — ``"deep"``, ``"gradient"``, or ``"partition"``.
        background_samples: Number of background samples for the explainer.

    Example::

        explainer = SHAPExplainer(model, method="deep", background_samples=50)
        result = explainer.explain(input_tensor, class_idx=0)
        print(result["shap_values"].shape)
    """

    def __init__(self, model: nn.Module, method: str = "deep", background_samples: int = 100):
        if method not in ("deep", "gradient", "partition"):
            raise ValueError(f"Unsupported SHAP method '{method}'. Choose from: deep, gradient, partition")

        self.model = model
        self.method = method
        self.background_samples = background_samples
        self._shap = None
        self._explainer = None

    def _lazy_import_shap(self):
        """Import shap on first use so it remains an optional dependency."""
        if self._shap is None:
            try:
                import shap
                self._shap = shap
            except ImportError:
                raise ImportError(
                    "The 'shap' package is required for SHAPExplainer. Install it with: pip install shap"
                )
        return self._shap

    def _build_explainer(self, background: torch.Tensor):
        """Create the underlying ``shap.Explainer`` for the chosen method.

        Args:
            background: Background dataset tensor used as the reference distribution.
        """
        shap = self._lazy_import_shap()
        self.model.eval()

        if self.method == "deep":
            self._explainer = shap.DeepExplainer(self.model, background)
        elif self.method == "gradient":
            self._explainer = shap.GradientExplainer(self.model, background)
        elif self.method == "partition":
            def model_fn(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    tensor = torch.tensor(x, dtype=torch.float32, device=next(self.model.parameters()).device)
                    out = self.model(tensor)
                return out.cpu().numpy()

            self._explainer = shap.Explainer(model_fn, shap.maskers.Image("inpaint_telea", background.shape[1:]))

    def _ensure_explainer(self, input_tensor: torch.Tensor) -> None:
        """Build the explainer with random background data if not yet initialised.

        Args:
            input_tensor: A sample tensor whose shape is used to create the
                random background distribution.
        """
        if self._explainer is None:
            device = input_tensor.device
            bg = torch.randn(self.background_samples, *input_tensor.shape[1:], device=device)
            self._build_explainer(bg)

    def explain(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Compute SHAP values for a single input.

        Args:
            input_tensor: Input tensor of shape ``[1, C, H, W]``.
            class_idx: Target class index.  When ``None`` the predicted
                class is used.

        Returns:
            Dictionary with keys ``shap_values`` (numpy array matching
            input shape), ``base_values``, and ``expected_value``.
        """
        self._ensure_explainer(input_tensor)
        self.model.eval()

        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                if output.dim() > 1 and output.shape[1] > 1:
                    class_idx = output.argmax(dim=1).item()
                else:
                    class_idx = 0

        if self.method in ("deep", "gradient"):
            shap_values = self._explainer.shap_values(input_tensor)

            if isinstance(shap_values, list):
                sv = shap_values[class_idx]
            else:
                sv = shap_values

            if isinstance(sv, torch.Tensor):
                sv = sv.cpu().numpy()

            expected = self._explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                ev = expected[class_idx] if class_idx < len(expected) else expected[0]
            else:
                ev = expected

            return {
                "shap_values": sv,
                "base_values": sv.mean(),
                "expected_value": float(ev) if not isinstance(ev, float) else ev,
            }

        # partition method
        input_np = input_tensor.cpu().numpy()
        sv = self._explainer(input_np)
        return {
            "shap_values": sv.values if hasattr(sv, "values") else np.array(sv),
            "base_values": sv.base_values if hasattr(sv, "base_values") else 0.0,
            "expected_value": float(sv.base_values.mean()) if hasattr(sv, "base_values") else 0.0,
        }

    def explain_batch(self, inputs: torch.Tensor) -> List[Dict[str, np.ndarray]]:
        """Compute SHAP values for a batch of inputs.

        Args:
            inputs: Batch tensor of shape ``[N, C, H, W]``.

        Returns:
            List of result dictionaries, one per sample (same format as
            :meth:`explain`).
        """
        results = []
        for i in range(inputs.shape[0]):
            results.append(self.explain(inputs[i : i + 1]))
        return results

    def get_feature_importance(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Compute per-pixel feature importance from SHAP values.

        The importance map is the mean of the absolute SHAP values across
        channels, normalised to ``[0, 1]``.

        Args:
            input_tensor: Input tensor of shape ``[1, C, H, W]``.
            class_idx: Target class index.

        Returns:
            2-D numpy array of shape ``[H, W]`` with values in ``[0, 1]``.
        """
        result = self.explain(input_tensor, class_idx)
        sv = result["shap_values"]

        if sv.ndim == 4:
            importance = np.abs(sv[0]).mean(axis=0)
        elif sv.ndim == 3:
            importance = np.abs(sv).mean(axis=0)
        else:
            importance = np.abs(sv)

        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        return importance

    def visualize(
        self,
        input_tensor: torch.Tensor,
        shap_values: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Render a SHAP importance overlay on the input image.

        Args:
            input_tensor: Original input tensor of shape ``[1, C, H, W]``.
            shap_values: SHAP values array (same spatial shape as input).
            save_path: File path to save the figure.  Displayed interactively
                when ``None``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[SHAPExplainer] matplotlib not installed, skipping visualization")
            return

        image = input_tensor[0].cpu().numpy()
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = image.squeeze(-1)

        if shap_values.ndim == 4:
            sv_map = np.abs(shap_values[0]).mean(axis=0)
        elif shap_values.ndim == 3:
            sv_map = np.abs(shap_values).mean(axis=0)
        else:
            sv_map = np.abs(shap_values)

        if sv_map.max() > sv_map.min():
            sv_map = (sv_map - sv_map.min()) / (sv_map.max() - sv_map.min())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cmap_img = "gray" if image.ndim == 2 else None
        axes[0].imshow(image, cmap=cmap_img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(sv_map, cmap="hot")
        axes[1].set_title("SHAP Importance")
        axes[1].axis("off")

        axes[2].imshow(image, cmap=cmap_img)
        axes[2].imshow(sv_map, cmap="jet", alpha=0.4)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.show()


class DeepExplainer:
    """Thin wrapper around ``shap.DeepExplainer`` for deep-learning models.

    Unlike :class:`SHAPExplainer`, this class requires the background data
    to be provided explicitly and only supports the DeepSHAP algorithm.

    Args:
        model: PyTorch model.
        background_data: Background tensor used as the reference distribution.

    Example::

        bg = train_images[:50]
        explainer = DeepExplainer(model, background_data=bg)
        shap_vals = explainer.explain(input_tensor)
    """

    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        self.model = model
        self.background_data = background_data
        self._explainer = None

    def _build(self) -> None:
        """Lazily construct the underlying ``shap.DeepExplainer``."""
        try:
            import shap
        except ImportError:
            raise ImportError("The 'shap' package is required for DeepExplainer. Install it with: pip install shap")
        self.model.eval()
        self._explainer = shap.DeepExplainer(self.model, self.background_data)

    def explain(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute DeepSHAP values for the given input.

        Args:
            input_tensor: Input tensor of shape ``[1, C, H, W]``.

        Returns:
            Numpy array of SHAP values with the same shape as the input.
        """
        if self._explainer is None:
            self._build()

        sv = self._explainer.shap_values(input_tensor)

        if isinstance(sv, list):
            sv = np.stack(sv, axis=0)
        if isinstance(sv, torch.Tensor):
            sv = sv.cpu().numpy()
        return sv
