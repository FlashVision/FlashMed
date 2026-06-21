"""Monte Carlo Dropout for predictive uncertainty estimation."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MCDropout:
    """Monte Carlo Dropout uncertainty estimation.

    Runs multiple stochastic forward passes with dropout enabled at
    inference time and decomposes the resulting variance into aleatoric
    and epistemic components.

    Args:
        model: PyTorch model (should contain ``nn.Dropout`` layers).
        num_samples: Number of stochastic forward passes per prediction.
        dropout_rate: If the model has no dropout layers, temporary dropout
            with this rate is injected before every ``nn.Linear`` layer.

    Example::

        mc = MCDropout(model, num_samples=50)
        result = mc.predict(input_tensor)
        print(f"Epistemic uncertainty: {result['epistemic_uncertainty']:.4f}")
    """

    def __init__(self, model: nn.Module, num_samples: int = 30, dropout_rate: float = 0.1):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self._injected_hooks: list = []

        has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) for m in model.modules())
        if not has_dropout:
            self._inject_dropout()

    def _inject_dropout(self) -> None:
        """Insert ``nn.Dropout`` layers before every ``nn.Linear`` in the model."""
        for name, module in list(self.model.named_children()):
            if isinstance(module, nn.Linear):
                wrapper = nn.Sequential(nn.Dropout(p=self.dropout_rate), module)
                setattr(self.model, name, wrapper)
            elif len(list(module.children())) > 0:
                self._inject_dropout_recursive(module)

    def _inject_dropout_recursive(self, parent: nn.Module) -> None:
        """Recursively inject dropout into nested sub-modules."""
        for name, module in list(parent.named_children()):
            if isinstance(module, nn.Linear):
                wrapper = nn.Sequential(nn.Dropout(p=self.dropout_rate), module)
                setattr(parent, name, wrapper)
            elif len(list(module.children())) > 0:
                self._inject_dropout_recursive(module)

    @staticmethod
    def enable_dropout(model: nn.Module) -> None:
        """Switch all dropout layers in *model* to training mode (stochastic).

        The rest of the model stays in eval mode so batch-norm statistics
        are not updated.

        Args:
            model: The model whose dropout layers should be activated.
        """
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run MC Dropout inference and return uncertainty estimates.

        Args:
            x: Input tensor of shape ``[1, C, H, W]`` (or any batch).

        Returns:
            Dictionary with keys:

            - ``mean_prediction`` — averaged softmax probabilities ``[N, num_classes]``.
            - ``predictive_uncertainty`` — predictive entropy per sample.
            - ``aleatoric_uncertainty`` — expected data (aleatoric) entropy.
            - ``epistemic_uncertainty`` — mutual information (epistemic).
            - ``all_predictions`` — all stochastic predictions ``[T, N, num_classes]``.
            - ``confidence_calibration`` — max mean-probability per sample.
        """
        self.model.eval()
        self.enable_dropout(self.model)

        all_preds = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(x)
                probs = F.softmax(output, dim=-1)
                all_preds.append(probs.cpu().numpy())

        all_preds_np = np.stack(all_preds, axis=0)  # [T, N, C]
        mean_pred = all_preds_np.mean(axis=0)  # [N, C]

        # predictive entropy H[y | x, D]
        pred_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)

        # expected entropy E_{theta}[H[y | x, theta]] (aleatoric)
        sample_entropy = -np.sum(all_preds_np * np.log(all_preds_np + 1e-10), axis=-1)  # [T, N]
        aleatoric = sample_entropy.mean(axis=0)  # [N]

        # mutual information I[y; theta | x, D] (epistemic)
        epistemic = pred_entropy - aleatoric  # [N]
        epistemic = np.maximum(epistemic, 0.0)

        confidence = mean_pred.max(axis=-1)

        return {
            "mean_prediction": mean_pred,
            "predictive_uncertainty": pred_entropy,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "all_predictions": all_preds_np,
            "confidence_calibration": confidence,
        }

    def calibrate(self, val_loader: DataLoader, num_bins: int = 15) -> Dict[str, float]:
        """Compute calibration metrics over a validation set.

        Args:
            val_loader: DataLoader yielding ``(inputs, targets)`` batches.
            num_bins: Number of bins for the reliability diagram.

        Returns:
            Dictionary with keys ``ece`` (Expected Calibration Error),
            ``mce`` (Maximum Calibration Error), ``bin_confidences``,
            ``bin_accuracies``, and ``bin_counts``.
        """
        all_confidences: list = []
        all_predictions: list = []
        all_targets: list = []

        self.model.eval()
        self.enable_dropout(self.model)

        with torch.no_grad():
            for inputs, targets in val_loader:
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)

                preds = []
                for _ in range(self.num_samples):
                    output = self.model(inputs)
                    probs = F.softmax(output, dim=-1)
                    preds.append(probs.cpu().numpy())

                mean_pred = np.stack(preds, axis=0).mean(axis=0)
                confidence = mean_pred.max(axis=-1)
                predicted = mean_pred.argmax(axis=-1)

                all_confidences.append(confidence)
                all_predictions.append(predicted)

                tgt = targets.cpu().numpy()
                if tgt.ndim > 1:
                    tgt = tgt.argmax(axis=-1)
                all_targets.append(tgt)

        confidences = np.concatenate(all_confidences)
        predictions = np.concatenate(all_predictions)
        targets_flat = np.concatenate(all_targets)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bin_confidences = np.zeros(num_bins)
        bin_accuracies = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        for i in range(num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            count = mask.sum()
            bin_counts[i] = count
            if count > 0:
                bin_confidences[i] = confidences[mask].mean()
                bin_accuracies[i] = (predictions[mask] == targets_flat[mask]).mean()

        total = confidences.shape[0]
        ece = np.sum(bin_counts / total * np.abs(bin_accuracies - bin_confidences))
        mce = np.max(np.abs(bin_accuracies - bin_confidences))

        return {
            "ece": float(ece),
            "mce": float(mce),
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }
