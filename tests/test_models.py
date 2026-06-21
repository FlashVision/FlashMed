"""Tests for FlashMed model architectures."""

import torch
import pytest


class TestMedViT:
    def test_forward_classification(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(num_classes=14, img_size=224, in_channels=3, depth=4, embed_dim=192, num_heads=6)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 14)

    def test_forward_features(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(num_classes=14, img_size=224, in_channels=3, depth=4, embed_dim=192, num_heads=6)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model(x, return_features=True)
        assert features.shape == (1, 192)

    def test_single_channel_input(self):
        from flashmed.models.architectures.med_vit import MedViT

        model = MedViT(num_classes=5, img_size=224, in_channels=1, depth=4, embed_dim=192, num_heads=6)
        model.eval()
        x = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)


class TestUNet3D:
    def test_forward_3d(self):
        from flashmed.models.architectures.unet_3d import UNet3D

        model = UNet3D(in_channels=4, num_classes=4, base_filters=16, depth=3, spatial_dims=3)
        model.eval()
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 32, 32, 32)

    def test_forward_2d(self):
        from flashmed.models.architectures.unet_3d import UNet3D

        model = UNet3D(in_channels=3, num_classes=2, base_filters=16, depth=3, spatial_dims=2)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 64, 64)


class TestMedVLM:
    def test_forward(self):
        from flashmed.models.architectures.med_vlm import MedVLM

        model = MedVLM(in_channels=3, img_size=224, embed_dim=256, vocab_size=1000, num_decoder_layers=2)
        model.eval()
        images = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            logits = model(images, input_ids=input_ids)
        assert logits.shape == (1, 10, 1000)

    def test_generate(self):
        from flashmed.models.architectures.med_vlm import MedVLM

        model = MedVLM(in_channels=3, img_size=224, embed_dim=256, vocab_size=1000, num_decoder_layers=2)
        model.eval()
        images = torch.randn(1, 3, 224, 224)
        tokens = model.generate(images, max_length=20, eos_token_id=102)
        assert tokens.shape[0] == 1
        assert tokens.shape[1] <= 20


class TestFlashMed:
    def test_classification_model(self):
        from flashmed.models.flashmed_model import FlashMed

        model = FlashMed(task="classification", num_classes=14, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 14)

    def test_segmentation_model(self):
        from flashmed.models.flashmed_model import FlashMed

        model = FlashMed(task="segmentation", num_classes=4, pretrained=False, in_channels=4, spatial_dims=3)
        model.eval()
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 32, 32, 32)

    def test_param_count(self):
        from flashmed.models.flashmed_model import FlashMed

        model = FlashMed(task="classification", num_classes=14, pretrained=False)
        assert model.get_num_params() > 0
