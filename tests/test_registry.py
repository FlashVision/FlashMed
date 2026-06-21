"""Tests for FlashMed registry system."""

import pytest


class TestRegistry:
    def test_register_and_build(self):
        from flashmed.registry import Registry

        reg = Registry("test")

        @reg.register("MyClass")
        class MyClass:
            def __init__(self, value=0):
                self.value = value

        obj = reg.build("MyClass", value=42)
        assert obj.value == 42

    def test_register_without_name(self):
        from flashmed.registry import Registry

        reg = Registry("test")

        @reg.register()
        class AnotherClass:
            pass

        assert "AnotherClass" in reg

    def test_duplicate_raises(self):
        from flashmed.registry import Registry

        reg = Registry("test")

        @reg.register("Dup")
        class Dup1:
            pass

        with pytest.raises(KeyError):
            @reg.register("Dup")
            class Dup2:
                pass

    def test_build_unknown_raises(self):
        from flashmed.registry import Registry

        reg = Registry("test")
        with pytest.raises(KeyError):
            reg.build("NonExistent")

    def test_list_and_len(self):
        from flashmed.registry import Registry

        reg = Registry("test")

        @reg.register("A")
        class A:
            pass

        @reg.register("B")
        class B:
            pass

        assert len(reg) == 2
        assert "A" in reg.list()
        assert "B" in reg.list()

    def test_global_registries_exist(self):
        from flashmed.registry import MODELS, DATASETS, TASKS, PRIVACY_METHODS

        assert MODELS.name == "models"
        assert DATASETS.name == "datasets"
        assert TASKS.name == "tasks"
        assert PRIVACY_METHODS.name == "privacy_methods"

    def test_models_registered(self):
        from flashmed.models.architectures.med_vit import MedViT  # noqa: F401
        from flashmed.models.architectures.unet_3d import UNet3D  # noqa: F401
        from flashmed.models.flashmed_model import FlashMed  # noqa: F401
        from flashmed.registry import MODELS

        assert "MedViT" in MODELS
        assert "UNet3D" in MODELS
        assert "FlashMed" in MODELS

    def test_datasets_registered(self):
        from flashmed.data.datasets import ChestXray14Dataset, ISICDataset  # noqa: F401
        from flashmed.registry import DATASETS

        assert "ChestXray14" in DATASETS
        assert "ISIC" in DATASETS
        assert "BraTS" in DATASETS
        assert "PathMNIST" in DATASETS
