"""Evidential Deep Learning for classification with uncertainty quantification."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialClassifier(nn.Module):
    """Evidential classifier that predicts Dirichlet concentration parameters.

    A lightweight evidential head is appended to an existing backbone.
    The Dirichlet parameters (alpha) encode both class probabilities and
    prediction uncertainty in a single forward pass — no ensembles or
    MC sampling required.

    Args:
        backbone: Feature-extraction model whose final output is a 1-D
            feature vector (or logits).
        num_classes: Number of target classes.
        freeze_backbone: If ``True`` the backbone parameters are frozen
            and only the evidential head is trained.

    Example::

        backbone = torchvision.models.resnet18(num_classes=512)
        model = EvidentialClassifier(backbone, num_classes=14)
        outputs = model(input_tensor)
        loss = model.compute_loss(outputs, targets, epoch=5)
    """

    def __init__(self, backbone: nn.Module, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self._detect_in_features()
        self.evidential_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softplus(),
        )

    def _detect_in_features(self) -> int:
        """Infer the backbone output dimensionality by a dummy forward pass."""
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            try:
                out = self.backbone(dummy)
            except Exception:
                dummy = torch.randn(1, 1, 224, 224)
                out = self.backbone(dummy)
            if out.dim() > 2:
                out = out.view(out.size(0), -1)
            return out.shape[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning Dirichlet parameters and uncertainty.

        Args:
            x: Input tensor of shape ``[N, C, H, W]``.

        Returns:
            Dictionary with keys:

            - ``logits`` — backbone logits ``[N, K]``.
            - ``alpha`` — Dirichlet concentration parameters ``[N, K]``.
            - ``uncertainty`` — dict of ``total``, ``data``, and
              ``distributional`` uncertainty tensors.
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        evidence = self.evidential_head(features)
        alpha = evidence + 1.0  # Dirichlet params >= 1

        strength = alpha.sum(dim=-1, keepdim=True)  # S = sum(alpha)
        probs = alpha / strength  # expected probabilities

        total_uncertainty = self.num_classes / strength.squeeze(-1)

        data_uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        distributional = total_uncertainty - data_uncertainty
        distributional = torch.clamp(distributional, min=0.0)

        return {
            "logits": probs,
            "alpha": alpha,
            "uncertainty": {
                "total": total_uncertainty,
                "data": data_uncertainty,
                "distributional": distributional,
            },
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        epoch: int = 0,
        annealing_epochs: int = 10,
    ) -> torch.Tensor:
        """Type-II maximum likelihood loss with KL-divergence annealing.

        The KL term regularises the Dirichlet towards a uniform prior.
        Its weight ramps linearly from 0 to 1 over the first
        *annealing_epochs* to let the model fit the data before the
        regulariser kicks in.

        Args:
            outputs: Dictionary returned by :meth:`forward`.
            targets: Ground-truth class indices ``[N]`` or one-hot ``[N, K]``.
            epoch: Current training epoch (for annealing schedule).
            annealing_epochs: Epochs over which the KL weight linearly
                ramps from 0 to 1.

        Returns:
            Scalar loss tensor.
        """
        alpha = outputs["alpha"]
        device = alpha.device

        if targets.dim() == 1:
            one_hot = F.one_hot(targets.long(), self.num_classes).float().to(device)
        else:
            one_hot = targets.float().to(device)

        strength = alpha.sum(dim=-1, keepdim=True)

        # type-II ML: E_{Dir}[CE]
        log_likelihood = (
            torch.lgamma(strength)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
            + ((one_hot * (torch.digamma(alpha) - torch.digamma(strength))).sum(dim=-1, keepdim=True))
        )
        mle_loss = -log_likelihood.mean()

        # KL(Dir(alpha) || Dir(1...1))
        annealing = min(1.0, epoch / max(annealing_epochs, 1))
        alpha_tilde = one_hot + (1.0 - one_hot) * (alpha - 1.0) + 1.0

        ones = torch.ones_like(alpha_tilde)
        kl = (
            torch.lgamma(alpha_tilde.sum(dim=-1))
            - torch.lgamma(ones.sum(dim=-1))
            - (torch.lgamma(alpha_tilde) - torch.lgamma(ones)).sum(dim=-1)
            + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde.sum(dim=-1, keepdim=True)))).sum(dim=-1)
        )
        kl_loss = kl.mean()

        return mle_loss + annealing * kl_loss

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run inference and return probabilities with uncertainty decomposition.

        Args:
            x: Input tensor of shape ``[N, C, H, W]``.

        Returns:
            Dictionary with numpy arrays:

            - ``class_probabilities`` — ``[N, K]`` expected class probs.
            - ``predicted_classes`` — ``[N]`` argmax class indices.
            - ``total_uncertainty`` — ``[N]`` total (vacuity) uncertainty.
            - ``data_uncertainty`` — ``[N]`` aleatoric component.
            - ``distributional_uncertainty`` — ``[N]`` epistemic component.
            - ``alpha`` — ``[N, K]`` raw Dirichlet parameters.
        """
        self.eval()
        outputs = self.forward(x)

        probs = outputs["logits"]
        alpha = outputs["alpha"]

        return {
            "class_probabilities": probs.cpu().numpy(),
            "predicted_classes": probs.argmax(dim=-1).cpu().numpy(),
            "total_uncertainty": outputs["uncertainty"]["total"].cpu().numpy(),
            "data_uncertainty": outputs["uncertainty"]["data"].cpu().numpy(),
            "distributional_uncertainty": outputs["uncertainty"]["distributional"].cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
        }
