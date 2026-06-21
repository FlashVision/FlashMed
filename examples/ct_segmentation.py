"""Example: 3D brain tumor segmentation on BraTS dataset."""

import torch
from flashmed import FlashMed, Trainer
from flashmed.tasks.segmentation import SegmentationTask
from flashmed.cfg import get_config


def train_brats_segmentation():
    """Train 3D UNet for brain tumor segmentation."""
    trainer = Trainer(
        task="segmentation",
        data_dir="data/brats",
        num_classes=4,
        epochs=200,
        batch_size=2,
        learning_rate=1e-4,
        device="cuda",
        save_dir="workspace/brats_seg",
    )
    trainer.train()


def inference_with_sliding_window():
    """Run inference on a full brain volume using sliding window."""
    model = FlashMed.from_pretrained(
        "workspace/brats_seg/flashmed_best.pth",
        device="cuda",
    )

    volume = torch.randn(1, 4, 240, 240, 155)

    seg_task = SegmentationTask(num_classes=4, spatial_dims=3)
    output = seg_task.sliding_window_inference(
        model, volume, roi_size=(96, 96, 96), overlap=0.5,
    )

    pred_mask = output.argmax(dim=1)
    print(f"Segmentation output shape: {pred_mask.shape}")
    print(f"Classes found: {torch.unique(pred_mask).tolist()}")


def evaluate_dice():
    """Compute Dice scores per class."""
    model = FlashMed.from_pretrained(
        "workspace/brats_seg/flashmed_best.pth",
        device="cuda",
    )
    model.eval()

    seg_task = SegmentationTask(num_classes=4, spatial_dims=3)
    dummy_pred = torch.randn(1, 4, 96, 96, 96)
    dummy_target = torch.randint(0, 4, (1, 96, 96, 96))

    dice_scores = seg_task.compute_dice_score(dummy_pred, dummy_target)
    for class_name, score in dice_scores.items():
        print(f"  {class_name}: {score:.4f}")


if __name__ == "__main__":
    train_brats_segmentation()
