"""Comprehensive test suite for FlashMed covering all architectures,
classification, segmentation, explainability, uncertainty, privacy,
report generation, and CLI."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from flashmed.registry import MODELS, TASKS, PRIVACY_METHODS


# ===================================================================
# Architecture: Medical ViT
# ===================================================================


class TestMedViT:
    def test_forward_multi_label(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(
            num_classes=14,
            in_channels=3,
            img_size=32,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 14)

    def test_forward_features(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(
            num_classes=14,
            in_channels=3,
            img_size=32,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            feats = model(x, return_features=True)
        assert feats.shape == (1, 64)

    def test_freeze_backbone(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(
            num_classes=14,
            in_channels=1,
            img_size=32,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            pretrained=False,
        )
        model.freeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total

    def test_attention_maps(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(
            num_classes=2,
            in_channels=3,
            img_size=32,
            patch_size=16,
            embed_dim=64,
            depth=2,
            num_heads=4,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            attn = model.get_attention_maps(x)
        assert attn.shape[0] == 2


# ===================================================================
# Architecture: 3D UNet
# ===================================================================


class TestUNet3D:
    def test_forward_3d(self):
        from flashmed.models.architectures.unet_3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=4, base_filters=8, depth=2, spatial_dims=3)
        model.eval()
        x = torch.randn(1, 1, 8, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_forward_2d(self):
        from flashmed.models.architectures.unet_3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=3, base_filters=8, depth=2, spatial_dims=2)
        model.eval()
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 32, 32)

    def test_deep_supervision(self):
        from flashmed.models.architectures.unet_3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=4, base_filters=8, depth=2, spatial_dims=3, deep_supervision=True)
        model.train()
        x = torch.randn(1, 1, 8, 8, 8)
        out = model(x)
        assert isinstance(out, tuple)

    def test_get_num_params(self):
        from flashmed.models.architectures.unet_3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=4, base_filters=8, depth=2)
        assert model.get_num_params() > 0


# ===================================================================
# Architecture: MedSAM
# ===================================================================


class TestMedSAM:
    def test_forward(self):
        from flashmed.models.architectures.medsam import MedSAM

        model = MedSAM(
            img_size=32,
            patch_size=16,
            encoder_embed_dim=64,
            encoder_depth=1,
            encoder_num_heads=4,
            decoder_embed_dim=32,
            num_mask_tokens=2,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9]])
        with torch.no_grad():
            masks, iou = model(x, boxes)
        assert masks.shape[0] == 1
        assert masks.shape[1] == 2
        assert iou.shape == (1, 2)

    def test_encode_image(self):
        from flashmed.models.architectures.medsam import MedSAM

        model = MedSAM(
            img_size=32, patch_size=16, encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4, decoder_embed_dim=32
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            emb = model.encode_image(x)
        assert emb.shape[0] == 1
        assert emb.shape[2] == 32

    def test_freeze_encoder(self):
        from flashmed.models.architectures.medsam import MedSAM

        model = MedSAM(
            img_size=32, patch_size=16, encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4, decoder_embed_dim=32
        )
        model.freeze_encoder()
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen > 0


# ===================================================================
# Architecture: nnU-Net
# ===================================================================


class TestNNUNet:
    def test_forward_3d(self):
        from flashmed.models.architectures.nnunet import nnUNet

        model = nnUNet(
            in_channels=1,
            num_classes=4,
            base_filters=8,
            max_filters=32,
            spatial_dims=3,
            deep_supervision=False,
            num_pool_per_axis=(2, 2, 2),
        )
        model.eval()
        x = torch.randn(1, 1, 8, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_fingerprint_dataset(self):
        from flashmed.models.architectures.nnunet import nnUNet

        info = {"median_shape": (64, 64, 64), "spacing": (1.0, 1.0, 1.0), "num_classes": 3, "in_channels": 1}
        config = nnUNet.fingerprint_dataset(info)
        assert "base_filters" in config
        assert config["num_classes"] == 3

    def test_auto_configure(self):
        from flashmed.models.architectures.nnunet import nnUNet

        info = {"median_shape": (32, 32), "spacing": (1.0, 1.0), "num_classes": 2, "in_channels": 1}
        model = nnUNet.auto_configure(info)
        assert model.num_classes == 2


# ===================================================================
# Architecture: SwinUNETR
# ===================================================================


class TestSwinUNETR:
    def test_forward_3d(self):
        from flashmed.models.architectures.swin_unetr import SwinUNETR

        model = SwinUNETR(
            img_size=16,
            in_channels=1,
            num_classes=2,
            embed_dim=12,
            depths=(1, 1),
            num_heads=(3, 6),
            window_size=4,
            spatial_dims=3,
        )
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 2


# ===================================================================
# Architecture: BiomedCLIP
# ===================================================================


class TestBiomedCLIP:
    def test_encode_image(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32,
            patch_size=16,
            vision_embed_dim=64,
            text_embed_dim=64,
            vision_depth=1,
            text_depth=1,
            vision_heads=4,
            text_heads=4,
            projection_dim=32,
        )
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            emb = model.encode_image(x)
        assert emb.shape == (2, 32)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_encode_text(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32,
            patch_size=16,
            vision_embed_dim=64,
            text_embed_dim=64,
            vision_depth=1,
            text_depth=1,
            vision_heads=4,
            text_heads=4,
            projection_dim=32,
            vocab_size=100,
        )
        model.eval()
        ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            emb = model.encode_text(ids)
        assert emb.shape == (2, 32)

    def test_contrastive_loss(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32,
            patch_size=16,
            vision_embed_dim=64,
            text_embed_dim=64,
            vision_depth=1,
            text_depth=1,
            vision_heads=4,
            text_heads=4,
            projection_dim=32,
            vocab_size=100,
        )
        imgs = torch.randn(2, 3, 32, 32)
        ids = torch.randint(0, 100, (2, 10))
        loss = model.contrastive_loss(imgs, ids)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_zero_shot_classify(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32,
            patch_size=16,
            vision_embed_dim=64,
            text_embed_dim=64,
            vision_depth=1,
            text_depth=1,
            vision_heads=4,
            text_heads=4,
            projection_dim=32,
            vocab_size=100,
        )
        model.eval()
        imgs = torch.randn(2, 3, 32, 32)
        class_desc = torch.randint(0, 100, (3, 10))
        with torch.no_grad():
            probs = model.zero_shot_classify(imgs, class_desc)
        assert probs.shape == (2, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)


# ===================================================================
# Classification task (multi-label, binary)
# ===================================================================


class TestClassificationTask:
    def test_multi_label_loss(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=14, multi_label=True)
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()
        loss = task.compute_loss(logits, targets)
        assert loss.dim() == 0

    def test_single_label_loss(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=5, multi_label=False)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = task.compute_loss(logits, targets)
        assert loss.dim() == 0

    def test_compute_predictions_multi_label(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=14, multi_label=True)
        logits = torch.randn(2, 14)
        preds = task.compute_predictions(logits)
        assert "probabilities" in preds
        assert "predictions" in preds
        assert preds["predictions"].shape == (2, 14)

    def test_compute_predictions_single_label(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=5, multi_label=False)
        logits = torch.randn(2, 5)
        preds = task.compute_predictions(logits)
        assert preds["predictions"].shape == (2,)


# ===================================================================
# Segmentation (Dice loss, 3D volumes)
# ===================================================================


class TestSegmentationTask:
    def test_dice_loss(self):
        from flashmed.tasks.segmentation import DiceLoss

        loss_fn = DiceLoss(num_classes=4)
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randint(0, 4, (2, 8, 8, 8))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_focal_loss(self):
        from flashmed.tasks.segmentation import FocalLoss

        loss_fn = FocalLoss()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randint(0, 4, (2, 8, 8))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_compute_dice_score(self):
        from flashmed.tasks.segmentation import SegmentationTask

        task = SegmentationTask(num_classes=4)
        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        scores = task.compute_dice_score(pred, target)
        assert "dice_mean" in scores
        assert 0.0 <= scores["dice_mean"] <= 1.0

    def test_segmentation_task_combined_loss(self):
        from flashmed.tasks.segmentation import SegmentationTask

        task = SegmentationTask(num_classes=4, loss_type="dice_ce")
        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        loss = task.compute_loss(pred, target)
        assert loss.dim() == 0


# ===================================================================
# Explainability: GradCAM++, ScoreCAM, LayerCAM
# ===================================================================


class TestGradCAMPlusPlus:
    def test_generate(self):
        from flashmed.explainability.gradcam import GradCAMPlusPlus

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5)
        )
        gcpp = GradCAMPlusPlus(model)
        x = torch.randn(1, 3, 32, 32)
        heatmap = gcpp.generate(x, class_idx=0)
        assert heatmap.shape == (32, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        gcpp.cleanup()


class TestScoreCAM:
    def test_generate(self):
        from flashmed.explainability.gradcam import ScoreCAM

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5)
        )
        scam = ScoreCAM(model, batch_size=4)
        x = torch.randn(1, 3, 32, 32)
        heatmap = scam.generate(x, class_idx=0)
        assert heatmap.shape == (32, 32)
        scam.cleanup()


class TestLayerCAM:
    def test_generate(self):
        from flashmed.explainability.gradcam import LayerCAM

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5)
        )
        lcam = LayerCAM(model)
        x = torch.randn(1, 3, 32, 32)
        heatmap = lcam.generate(x, class_idx=0)
        assert heatmap.shape == (32, 32)
        lcam.cleanup()


# ===================================================================
# SHAP explainability (mocked)
# ===================================================================


class TestSHAPExplainer:
    def test_init_valid_methods(self):
        from flashmed.explainability.shap_explainer import SHAPExplainer

        model = nn.Linear(10, 5)
        for method in ("deep", "gradient", "partition"):
            explainer = SHAPExplainer(model, method=method)
            assert explainer.method == method

    def test_init_invalid_method(self):
        from flashmed.explainability.shap_explainer import SHAPExplainer

        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Unsupported"):
            SHAPExplainer(model, method="unknown")


# ===================================================================
# Uncertainty: MC Dropout
# ===================================================================


class TestMCDropout:
    def test_predict(self):
        from flashmed.uncertainty.mc_dropout import MCDropout

        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 5))
        mc = MCDropout(model, num_samples=5)
        x = torch.randn(2, 10)
        result = mc.predict(x)
        assert "mean_prediction" in result
        assert result["mean_prediction"].shape == (2, 5)
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "confidence_calibration" in result

    def test_enable_dropout(self):
        from flashmed.uncertainty.mc_dropout import MCDropout

        model = nn.Sequential(nn.Dropout(0.5), nn.Linear(10, 5))
        model.eval()
        MCDropout.enable_dropout(model)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                assert m.training


# ===================================================================
# Uncertainty: Deep Ensembles
# ===================================================================


class TestDeepEnsemble:
    def test_predict(self):
        from flashmed.uncertainty.ensemble import DeepEnsemble

        def make_model():
            return nn.Sequential(nn.Linear(10, 5))

        ens = DeepEnsemble(make_model, num_models=3)
        x = torch.randn(2, 10)
        result = ens.predict(x)
        assert "mean_prediction" in result
        assert result["mean_prediction"].shape == (2, 5)
        assert "epistemic_uncertainty" in result
        assert "disagreement" in result
        assert result["all_predictions"].shape[0] == 3

    def test_add_model(self):
        from flashmed.uncertainty.ensemble import DeepEnsemble

        ens = DeepEnsemble(lambda: nn.Linear(10, 5), num_models=2)
        assert len(ens.models) == 2
        ens.add_model(nn.Linear(10, 5))
        assert len(ens.models) == 3


# ===================================================================
# Uncertainty: Evidential Learning
# ===================================================================


class TestEvidentialClassifier:
    @staticmethod
    def _make_backbone():
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def test_forward(self):
        from flashmed.uncertainty.evidential import EvidentialClassifier

        backbone = self._make_backbone()
        model = EvidentialClassifier(backbone, num_classes=5, freeze_backbone=False)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert "logits" in out
        assert "alpha" in out
        assert "uncertainty" in out
        assert out["alpha"].shape == (2, 5)
        assert (out["alpha"] >= 1.0).all()

    def test_compute_loss(self):
        from flashmed.uncertainty.evidential import EvidentialClassifier

        backbone = self._make_backbone()
        model = EvidentialClassifier(backbone, num_classes=5, freeze_backbone=False)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        targets = torch.tensor([0, 3])
        loss = model.compute_loss(out, targets, epoch=5)
        assert loss.dim() == 0

    def test_predict_with_uncertainty(self):
        from flashmed.uncertainty.evidential import EvidentialClassifier

        backbone = self._make_backbone()
        model = EvidentialClassifier(backbone, num_classes=5, freeze_backbone=False)
        x = torch.randn(2, 3, 32, 32)
        result = model.predict_with_uncertainty(x)
        assert "class_probabilities" in result
        assert "predicted_classes" in result
        assert "total_uncertainty" in result
        assert result["predicted_classes"].shape == (2,)


# ===================================================================
# DICOM Anonymization (mocked)
# ===================================================================


class TestDicomAnonymizer:
    def test_pseudonymize(self):
        from flashmed.privacy.anonymization import DicomAnonymizer

        anon = DicomAnonymizer(method="pseudonymize", salt="test")
        pseudo = anon._pseudonymize("John Doe")
        assert pseudo.startswith("ANON_")
        assert len(pseudo) > 5

    def test_deterministic(self):
        from flashmed.privacy.anonymization import DicomAnonymizer

        anon = DicomAnonymizer(method="pseudonymize", salt="test")
        p1 = anon._pseudonymize("Patient123")
        p2 = anon._pseudonymize("Patient123")
        assert p1 == p2

    def test_different_inputs(self):
        from flashmed.privacy.anonymization import DicomAnonymizer

        anon = DicomAnonymizer(method="pseudonymize", salt="test")
        p1 = anon._pseudonymize("Alice")
        p2 = anon._pseudonymize("Bob")
        assert p1 != p2


# ===================================================================
# Federated Learning
# ===================================================================


class TestFederatedLearner:
    def test_partition_iid(self):
        from flashmed.privacy.federated import FederatedLearner
        from torch.utils.data import TensorDataset

        model = nn.Linear(10, 5)
        fl = FederatedLearner(model, num_clients=3)

        data = TensorDataset(torch.randn(30, 10), torch.randint(0, 5, (30,)))
        subsets = fl.partition_data(data, strategy="iid")
        assert len(subsets) == 3
        total = sum(len(s) for s in subsets)
        assert total == 30

    def test_aggregate(self):
        from flashmed.privacy.federated import FederatedLearner

        model = nn.Linear(10, 5)
        fl = FederatedLearner(model, num_clients=2)
        state1 = {k: v.clone() for k, v in model.state_dict().items()}
        state2 = {k: v.clone() + 1.0 for k, v in model.state_dict().items()}
        fl._aggregate([state1, state2], [10, 10])
        for k in fl.global_state:
            expected = (state1[k] + state2[k]) / 2
            assert torch.allclose(fl.global_state[k], expected, atol=1e-5)


# ===================================================================
# Differential Privacy
# ===================================================================


class TestDifferentialPrivacy:
    def test_compute_noise_multiplier(self):
        from flashmed.privacy.differential_privacy import DifferentialPrivacy

        model = nn.Linear(10, 5)
        dp = DifferentialPrivacy(model, epsilon=8.0, delta=1e-5)
        assert dp.noise_multiplier > 0

    def test_clip_and_noise(self):
        from flashmed.privacy.differential_privacy import DifferentialPrivacy

        model = nn.Linear(10, 5)
        dp = DifferentialPrivacy(model, epsilon=8.0, max_grad_norm=1.0)
        x = torch.randn(2, 10)
        out = model(x)
        out.sum().backward()
        dp.clip_and_noise_gradients()
        assert dp._steps == 1

    def test_validate_model_compatibility(self):
        from flashmed.privacy.differential_privacy import DifferentialPrivacy

        model_ok = nn.Sequential(nn.Linear(10, 5), nn.GroupNorm(1, 5))
        dp = DifferentialPrivacy(model_ok)
        assert dp.validate_model_compatibility()

        model_bad = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5))
        dp2 = DifferentialPrivacy(model_bad)
        assert not dp2.validate_model_compatibility()

    def test_replace_batchnorm(self):
        from flashmed.privacy.differential_privacy import DifferentialPrivacy

        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16), nn.ReLU())
        model = DifferentialPrivacy.replace_batchnorm(model)
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        assert not has_bn

    def test_dp_optimizer(self):
        from flashmed.privacy.differential_privacy import DPOptimizer

        model = nn.Linear(10, 5)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        dp_opt = DPOptimizer(opt, max_grad_norm=1.0, noise_multiplier=0.5)
        x = torch.randn(2, 10)
        out = model(x)
        out.sum().backward()
        dp_opt.step()


# ===================================================================
# Report Generation
# ===================================================================


class TestReportGeneration:
    def test_compute_loss(self):
        from flashmed.tasks.report_gen import ReportGenerationTask

        task = ReportGenerationTask(vocab_size=100, max_length=32)
        logits = torch.randn(2, 20, 100)
        target_ids = torch.randint(0, 100, (2, 20))
        loss = task.compute_loss(logits, target_ids)
        assert loss.dim() == 0

    def test_compute_metrics(self):
        from flashmed.tasks.report_gen import ReportGenerationTask

        task = ReportGenerationTask()
        generated = ["the heart is normal in size", "lungs are clear"]
        references = ["the heart is normal in size and shape", "lungs are clear bilateral"]
        metrics = task.compute_metrics(generated, references)
        assert "bleu-4" in metrics
        assert "rouge-l" in metrics
        assert 0.0 <= metrics["rouge-l"] <= 1.0

    def test_bleu_empty(self):
        from flashmed.tasks.report_gen import ReportGenerationTask

        bleu = ReportGenerationTask._compute_bleu([], ["hello", "world"])
        assert bleu == 0.0

    def test_rouge_l(self):
        from flashmed.tasks.report_gen import ReportGenerationTask

        rouge = ReportGenerationTask._compute_rouge_l(["hello", "world"], ["hello", "world"])
        assert rouge == 1.0


# ===================================================================
# CLI
# ===================================================================


class TestMedCLI:
    def test_main_no_command(self):
        from flashmed.cli import main

        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["flashmed"]):
                main()
        assert exc.value.code == 0


# ===================================================================
# Registry
# ===================================================================


class TestMedRegistries:
    def test_models_registered(self):
        assert "MedViT" in MODELS
        assert "UNet3D" in MODELS
        assert "MedSAM" in MODELS
        assert "nnUNet" in MODELS
        assert "SwinUNETR" in MODELS
        assert "BiomedCLIP" in MODELS

    def test_tasks_registered(self):
        assert "classification" in TASKS
        assert "segmentation" in TASKS
        assert "report_gen" in TASKS

    def test_privacy_registered(self):
        assert "anonymization" in PRIVACY_METHODS
        assert "federated" in PRIVACY_METHODS
        assert "differential_privacy" in PRIVACY_METHODS
