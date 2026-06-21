"""Example: Train a chest X-ray multi-label classifier using FlashMed."""

from flashmed import FlashMed, Trainer, Predictor
from flashmed.cfg import get_config


def train_xray_classifier():
    """Train a MedViT classifier on ChestX-ray14 dataset."""
    trainer = Trainer(
        task="classification",
        data_dir="data/chestxray14",
        num_classes=14,
        epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        device="cuda",
        save_dir="workspace/xray_cls",
    )
    trainer.train()


def train_with_lora():
    """Fine-tune with LoRA for parameter-efficient training."""
    trainer = Trainer(
        task="classification",
        data_dir="data/chestxray14",
        num_classes=14,
        epochs=30,
        batch_size=16,
        learning_rate=1e-4,
        lora=True,
        device="cuda",
        save_dir="workspace/xray_cls_lora",
    )
    trainer.train()


def predict_xray():
    """Run inference on a chest X-ray."""
    predictor = Predictor(
        model_path="workspace/xray_cls/flashmed_best.pth",
        task="classification",
        device="cuda",
    )

    results = predictor.predict("test_xray.png")
    print("Findings:")
    for finding in results.get("findings", []):
        print(f"  {finding['label']}: {finding['confidence']:.2%}")


def predict_from_config():
    """Train using YAML configuration."""
    from flashmed.cfg import load_yaml_config

    cfg = load_yaml_config("configs/flashmed_xray_cls.yaml")
    trainer = Trainer(config=cfg)
    trainer.train()


if __name__ == "__main__":
    train_xray_classifier()
