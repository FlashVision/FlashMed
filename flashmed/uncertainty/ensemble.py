"""Deep Ensemble uncertainty estimation for FlashMed models."""

import copy
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DeepEnsemble:
    """Deep Ensemble of independently trained models for uncertainty estimation.

    Each ensemble member is initialised with different random weights (via
    *model_fn*) and trained separately.  At prediction time the disagreement
    between members provides a natural measure of epistemic uncertainty.

    Args:
        model_fn: Callable that returns a new ``nn.Module`` instance.
        num_models: Number of ensemble members to create.

    Example::

        def make_model():
            return torchvision.models.resnet18(num_classes=14)

        ens = DeepEnsemble(make_model, num_models=5)
        ens.train(train_loader, val_loader, epochs=20)
        result = ens.predict(input_tensor)
    """

    def __init__(self, model_fn: Callable[[], nn.Module], num_models: int = 5):
        self.model_fn = model_fn
        self.num_models = num_models
        self.models: List[nn.Module] = [model_fn() for _ in range(num_models)]
        self.ensemble_weights: Optional[np.ndarray] = None
        self._device = torch.device("cpu")

    def _to_device(self, device: torch.device) -> None:
        """Move all ensemble members to *device*."""
        self._device = device
        for m in self.models:
            m.to(device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, list]:
        """Train every ensemble member independently.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation loader for tracking performance.
            epochs: Number of training epochs per member.
            lr: Learning rate.
            criterion: Loss function (defaults to ``nn.CrossEntropyLoss``).

        Returns:
            Dictionary mapping member index (as string) to a list of per-epoch
            training losses, plus ``"val_losses"`` if *val_loader* was given.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        device = next(self.models[0].parameters()).device
        self._device = device

        history: Dict[str, list] = {}

        for idx, model in enumerate(self.models):
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            member_losses: list = []
            member_val_losses: list = []

            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                member_losses.append(avg_loss)

                if val_loader is not None:
                    val_loss = self._evaluate(model, val_loader, criterion, device)
                    member_val_losses.append(val_loss)

            history[str(idx)] = member_losses
            if member_val_losses:
                history[f"{idx}_val"] = member_val_losses

        self._compute_weights(val_loader, criterion)

        return history

    @staticmethod
    def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
        """Evaluate a single model on *loader* and return the average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, targets).item()
                num_batches += 1
        model.train()
        return total_loss / max(num_batches, 1)

    def _compute_weights(self, val_loader: Optional[DataLoader], criterion: nn.Module) -> None:
        """Compute per-member weights from validation loss (inverse softmax)."""
        if val_loader is None:
            self.ensemble_weights = np.ones(len(self.models)) / len(self.models)
            return

        losses = []
        for model in self.models:
            loss = self._evaluate(model, val_loader, criterion, self._device)
            losses.append(loss)

        losses_np = np.array(losses)
        inv = 1.0 / (losses_np + 1e-8)
        self.ensemble_weights = inv / inv.sum()

    def add_model(self, model: nn.Module) -> None:
        """Add a pre-trained model to the ensemble.

        Args:
            model: A trained ``nn.Module`` to include in predictions.
        """
        self.models.append(model)
        self.ensemble_weights = np.ones(len(self.models)) / len(self.models)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run ensemble prediction with uncertainty decomposition.

        Args:
            x: Input tensor (any batch shape accepted by the models).

        Returns:
            Dictionary with keys:

            - ``mean_prediction`` — weighted-average softmax probabilities.
            - ``predictive_uncertainty`` — predictive entropy.
            - ``epistemic_uncertainty`` — mutual information across members.
            - ``disagreement`` — mean pairwise KL divergence between members.
            - ``all_predictions`` — stacked predictions ``[M, N, C]``.
            - ``ensemble_weights`` — per-member weight vector.
        """
        device = x.device
        all_preds = []

        for model in self.models:
            model.eval()
            model.to(device)
            output = model(x)
            probs = F.softmax(output, dim=-1)
            all_preds.append(probs.cpu().numpy())

        all_preds_np = np.stack(all_preds, axis=0)  # [M, N, C]

        weights = self.ensemble_weights if self.ensemble_weights is not None else np.ones(len(self.models)) / len(self.models)
        weights_expanded = weights.reshape(-1, 1, 1)
        mean_pred = (all_preds_np * weights_expanded).sum(axis=0)  # [N, C]

        pred_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)

        member_entropy = -np.sum(all_preds_np * np.log(all_preds_np + 1e-10), axis=-1)  # [M, N]
        weighted_member_entropy = (member_entropy * weights.reshape(-1, 1)).sum(axis=0)  # [N]
        epistemic = pred_entropy - weighted_member_entropy
        epistemic = np.maximum(epistemic, 0.0)

        # pairwise KL divergence as a disagreement measure
        num_members = len(self.models)
        disagreement = np.zeros(all_preds_np.shape[1])
        count = 0
        for i in range(num_members):
            for j in range(i + 1, num_members):
                kl = np.sum(all_preds_np[i] * np.log((all_preds_np[i] + 1e-10) / (all_preds_np[j] + 1e-10)), axis=-1)
                disagreement += kl
                count += 1
        if count > 0:
            disagreement /= count

        return {
            "mean_prediction": mean_pred,
            "predictive_uncertainty": pred_entropy,
            "epistemic_uncertainty": epistemic,
            "disagreement": disagreement,
            "all_predictions": all_preds_np,
            "ensemble_weights": weights.copy(),
        }

    def save(self, path: str) -> None:
        """Persist all ensemble members and weights to disk.

        Args:
            path: Directory where model checkpoints will be saved.
        """
        os.makedirs(path, exist_ok=True)
        for idx, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(path, f"member_{idx}.pt"))
        if self.ensemble_weights is not None:
            np.save(os.path.join(path, "ensemble_weights.npy"), self.ensemble_weights)
        torch.save({"num_models": len(self.models)}, os.path.join(path, "ensemble_meta.pt"))

    def load(self, path: str) -> None:
        """Restore ensemble members and weights from disk.

        The number of model instances must match the saved checkpoint count.
        Missing members are created via *model_fn*.

        Args:
            path: Directory containing saved checkpoints.
        """
        meta = torch.load(os.path.join(path, "ensemble_meta.pt"), weights_only=True)
        num_saved = meta["num_models"]

        while len(self.models) < num_saved:
            self.models.append(self.model_fn())

        for idx in range(num_saved):
            ckpt_path = os.path.join(path, f"member_{idx}.pt")
            state = torch.load(ckpt_path, weights_only=True)
            self.models[idx].load_state_dict(state)

        weights_path = os.path.join(path, "ensemble_weights.npy")
        if os.path.exists(weights_path):
            self.ensemble_weights = np.load(weights_path)
        else:
            self.ensemble_weights = np.ones(len(self.models)) / len(self.models)
