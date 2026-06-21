"""Tests for FlashMed model architectures."""

import torch


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


class TestMedSAM:
    def test_forward_with_boxes(self):
        from flashmed.models.architectures.medsam import MedSAM

        model = MedSAM(
            img_size=64, patch_size=16, encoder_embed_dim=96,
            encoder_depth=2, encoder_num_heads=3, decoder_embed_dim=48,
            num_mask_tokens=4, in_channels=3,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9]])
        with torch.no_grad():
            masks, iou_pred = model(x, boxes)
        assert masks.shape == (1, 4, 64, 64)
        assert iou_pred.shape == (1, 4)

    def test_encode_image(self):
        from flashmed.models.architectures.medsam import MedSAM

        model = MedSAM(
            img_size=64, patch_size=16, encoder_embed_dim=96,
            encoder_depth=2, encoder_num_heads=3, decoder_embed_dim=48,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            embeddings = model.encode_image(x)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[2] == 48

    def test_freeze_encoder(self):
        from flashmed.models.architectures.medsam import MedSAM

        model = MedSAM(img_size=64, patch_size=16, encoder_embed_dim=96, encoder_depth=2, encoder_num_heads=3)
        model.freeze_encoder()
        for param in model.encoder_blocks.parameters():
            assert not param.requires_grad


class TestNNUNet:
    def test_forward_3d(self):
        from flashmed.models.architectures.nnunet import nnUNet

        model = nnUNet(
            in_channels=1, num_classes=4, base_filters=16, max_filters=64,
            spatial_dims=3, deep_supervision=False, num_pool_per_axis=(2, 2, 2),
        )
        model.eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 32, 32, 32)

    def test_forward_2d(self):
        from flashmed.models.architectures.nnunet import nnUNet

        model = nnUNet(
            in_channels=3, num_classes=2, base_filters=16, max_filters=64,
            spatial_dims=2, deep_supervision=False, num_pool_per_axis=(2, 2, 2),
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 64, 64)

    def test_deep_supervision(self):
        from flashmed.models.architectures.nnunet import nnUNet

        model = nnUNet(
            in_channels=1, num_classes=3, base_filters=16, max_filters=64,
            spatial_dims=3, deep_supervision=True, num_pool_per_axis=(2, 2, 2),
        )
        model.train()
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert isinstance(out, list)
        assert out[0].shape == (1, 3, 32, 32, 32)

    def test_auto_configure(self):
        from flashmed.models.architectures.nnunet import nnUNet

        dataset_info = {
            "median_shape": [128, 128, 64],
            "num_classes": 5,
            "modalities": 1,
            "spacing": [1.0, 1.0, 2.0],
        }
        model = nnUNet.auto_configure(dataset_info)
        assert model.num_classes == 5
        assert model.in_channels == 1


class TestSwinUNETR:
    def test_forward_3d(self):
        from flashmed.models.architectures.swin_unetr import SwinUNETR

        model = SwinUNETR(
            img_size=32, in_channels=1, num_classes=3, embed_dim=24,
            depths=(2, 2), num_heads=(3, 6), window_size=4, spatial_dims=3,
        )
        model.eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 3

    def test_forward_2d(self):
        from flashmed.models.architectures.swin_unetr import SwinUNETR

        model = SwinUNETR(
            img_size=64, in_channels=3, num_classes=2, embed_dim=24,
            depths=(2, 2), num_heads=(3, 6), window_size=8, spatial_dims=2,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 2


class TestBiomedCLIP:
    def test_forward(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32, patch_size=16, vision_embed_dim=64, text_embed_dim=64,
            vision_depth=2, text_depth=2, vision_heads=4, text_heads=4,
            vocab_size=500, max_text_len=16, projection_dim=32,
        )
        model.eval()
        images = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(0, 500, (2, 16))
        with torch.no_grad():
            logits = model(images, input_ids)
        assert logits.shape == (2, 2)

    def test_zero_shot_classify(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32, patch_size=16, vision_embed_dim=64, text_embed_dim=64,
            vision_depth=2, text_depth=2, vision_heads=4, text_heads=4,
            vocab_size=500, max_text_len=16, projection_dim=32,
        )
        model.eval()
        images = torch.randn(2, 3, 32, 32)
        class_descs = torch.randint(0, 500, (5, 16))
        with torch.no_grad():
            probs = model.zero_shot_classify(images, class_descs)
        assert probs.shape == (2, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_contrastive_loss(self):
        from flashmed.models.architectures.biomed_clip import BiomedCLIP

        model = BiomedCLIP(
            img_size=32, patch_size=16, vision_embed_dim=64, text_embed_dim=64,
            vision_depth=2, text_depth=2, vision_heads=4, text_heads=4,
            vocab_size=500, max_text_len=16, projection_dim=32,
        )
        images = torch.randn(4, 3, 32, 32)
        input_ids = torch.randint(0, 500, (4, 16))
        loss = model.contrastive_loss(images, input_ids)
        assert loss.item() > 0


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


class TestRegistryNewModels:
    def test_new_models_registered(self):
        from flashmed.registry import MODELS
        import flashmed.models.architectures  # noqa: F401

        assert "MedSAM" in MODELS
        assert "nnUNet" in MODELS
        assert "SwinUNETR" in MODELS
        assert "BiomedCLIP" in MODELS
