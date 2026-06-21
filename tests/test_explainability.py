"""Tests for FlashMed explainability and uncertainty modules."""

import torch
import torch.nn as nn
import numpy as np
import pytest


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x, **kwargs):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestGradCAMPlusPlus:
    def test_generate_heatmap(self):
        from flashmed.explainability.gradcam import GradCAMPlusPlus

        model = _TinyClassifier()
        cam = GradCAMPlusPlus(model)
        x = torch.randn(1, 3, 32, 32)
        heatmap = cam.generate(x)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (32, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        cam.cleanup()

    def test_generate_specific_class(self):
        from flashmed.explainability.gradcam import GradCAMPlusPlus

        model = _TinyClassifier()
        cam = GradCAMPlusPlus(model)
        x = torch.randn(1, 3, 32, 32)
        heatmap = cam.generate(x, class_idx=2)
        assert heatmap.shape == (32, 32)
        cam.cleanup()


class TestScoreCAM:
    def test_generate_heatmap(self):
        from flashmed.explainability.gradcam import ScoreCAM

        model = _TinyClassifier()
        cam = ScoreCAM(model)
        x = torch.randn(1, 3, 32, 32)
        heatmap = cam.generate(x)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (32, 32)
        cam.cleanup()


class TestLayerCAM:
    def test_generate_heatmap(self):
        from flashmed.explainability.gradcam import LayerCAM

        model = _TinyClassifier()
        cam = LayerCAM(model)
        x = torch.randn(1, 3, 32, 32)
        heatmap = cam.generate(x)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (32, 32)
        cam.cleanup()


class TestMCDropout:
    def test_predict_uncertainty(self):
        from flashmed.uncertainty.mc_dropout import MCDropout

        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 5),
        )
        mc = MCDropout(model, num_samples=10)
        x = torch.randn(1, 3, 32, 32)
        result = mc.predict(x)

        assert "mean_prediction" in result
        assert "epistemic_uncertainty" in result
        assert "predictive_uncertainty" in result
        assert result["mean_prediction"].shape == (1, 5)


class TestDeepEnsemble:
    def test_predict_with_pretrained(self):
        from flashmed.uncertainty.ensemble import DeepEnsemble

        def make_model():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 8 * 8, 5),
            )

        ensemble = DeepEnsemble(make_model, num_models=3)
        for _ in range(3):
            ensemble.add_model(make_model())

        x = torch.randn(2, 3, 8, 8)
        result = ensemble.predict(x)

        assert "mean_prediction" in result
        assert "epistemic_uncertainty" in result
        assert result["mean_prediction"].shape[0] == 2


class TestEvidentialClassifier:
    def test_forward_and_loss(self):
        from flashmed.uncertainty.evidential import EvidentialClassifier

        backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 64),
            nn.ReLU(),
        )
        model = EvidentialClassifier(backbone, num_classes=5)
        x = torch.randn(4, 3, 32, 32)
        outputs = model(x)

        assert "logits" in outputs
        assert "alpha" in outputs
        assert "uncertainty" in outputs
        assert outputs["logits"].shape == (4, 5)

        targets = torch.randint(0, 5, (4,))
        loss = model.compute_loss(outputs, targets)
        assert loss.requires_grad
