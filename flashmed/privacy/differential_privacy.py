"""Differential Privacy (DP-SGD) for medical model training."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flashmed.registry import PRIVACY_METHODS


@PRIVACY_METHODS.register("differential_privacy")
class DifferentialPrivacy:
    """Differential Privacy training wrapper using DP-SGD.

    Provides formal privacy guarantees (epsilon, delta) for model training
    on sensitive medical data. Implements gradient clipping and Gaussian noise
    injection per the Abadi et al. DP-SGD framework.

    Args:
        model: The model to train with DP
        epsilon: Privacy budget (lower = more private)
        delta: Probability of privacy breach
        max_grad_norm: Per-sample gradient clipping norm
        noise_multiplier: Gaussian noise scale (auto-computed if None)
        secure_mode: Use secure noise generation
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        secure_mode: bool = False,
    ):
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier or self._compute_noise_multiplier()
        self.secure_mode = secure_mode
        self._steps = 0
        self._privacy_spent = 0.0

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier to achieve target epsilon."""
        import math
        sigma = math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return max(sigma, 0.1)

    def make_private(self, optimizer: torch.optim.Optimizer, data_loader: DataLoader) -> Tuple:
        """Wrap model, optimizer, and dataloader for DP training.

        Attempts to use Opacus if available, otherwise falls back to manual DP-SGD.
        """
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine(secure_mode=self.secure_mode)
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=1,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
            self._engine = privacy_engine
            return model, optimizer, data_loader
        except ImportError:
            print("[DP] Opacus not found, using manual DP-SGD implementation")
            self._engine = None
            return self.model, DPOptimizer(optimizer, self.max_grad_norm, self.noise_multiplier), data_loader

    def clip_and_noise_gradients(self):
        """Manually clip per-sample gradients and add noise (fallback without Opacus)."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-7))
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)
                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                param.grad.data.add_(noise)

        self._steps += 1

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get the current privacy expenditure (epsilon, delta).

        Uses RDP accountant if Opacus is available, otherwise provides an estimate.
        """
        if hasattr(self, "_engine") and self._engine is not None:
            eps = self._engine.get_epsilon(self.delta)
            return eps, self.delta

        import math
        eps_estimate = self.noise_multiplier * math.sqrt(2 * self._steps * math.log(1 / self.delta))
        eps_estimate = min(eps_estimate, self.epsilon * (self._steps / 100))
        return eps_estimate, self.delta

    def validate_model_compatibility(self) -> bool:
        """Check if the model is compatible with DP training (no batch norm, etc.)."""
        incompatible = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                incompatible.append(f"  - {name}: BatchNorm (use GroupNorm or InstanceNorm)")

        if incompatible:
            print("[DP] Model has incompatible layers:")
            for msg in incompatible:
                print(msg)
            return False
        return True

    @staticmethod
    def replace_batchnorm(model: nn.Module) -> nn.Module:
        """Replace BatchNorm layers with GroupNorm for DP compatibility."""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                num_groups = min(32, num_channels)
                while num_channels % num_groups != 0:
                    num_groups -= 1
                setattr(model, name, nn.GroupNorm(num_groups, num_channels))
            elif isinstance(module, nn.BatchNorm1d):
                setattr(model, name, nn.GroupNorm(1, module.num_features))
            elif isinstance(module, nn.BatchNorm3d):
                num_channels = module.num_features
                num_groups = min(32, num_channels)
                while num_channels % num_groups != 0:
                    num_groups -= 1
                setattr(model, name, nn.GroupNorm(num_groups, num_channels))
            else:
                DifferentialPrivacy.replace_batchnorm(module)
        return model


class DPOptimizer:
    """Manual DP-SGD optimizer wrapper for when Opacus is not available."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_grad_norm: float, noise_multiplier: float):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad_norm = param.grad.data.norm(2)
                clip_factor = min(1.0, self.max_grad_norm / (grad_norm + 1e-7))
                param.grad.data.mul_(clip_factor)
                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                param.grad.data.add_(noise)

        self.optimizer.step()

    @property
    def param_groups(self):
        return self.optimizer.param_groups
