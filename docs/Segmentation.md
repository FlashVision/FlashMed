# Segmentation

2D and 3D medical image segmentation for organs and lesions.

## Overview

FlashMed uses a 3D UNet with instance normalization for volumetric segmentation of CT and MRI scans.

## Supported Tasks

- Brain tumor segmentation (BraTS — 4 classes)
- Organ segmentation (CT — liver, spleen, kidneys)
- Lesion delineation (2D/3D)
- Cardiac segmentation (MRI)

## Training

```python
from flashmed import Trainer

trainer = Trainer(
    task="segmentation",
    data_dir="data/brats",
    num_classes=4,
    epochs=200,
    batch_size=2,
    learning_rate=1e-4,
)
trainer.train()
```

## 3D vs 2D

3D volumetric segmentation:
```yaml
task: segmentation
spatial_dims: 3
roi_size: [96, 96, 96]
in_channels: 4  # Multi-modal MRI
```

2D slice segmentation:
```yaml
task: segmentation
spatial_dims: 2
input_size: 512
in_channels: 1
```

## Loss Functions

- **Dice + CE** — Combined loss (default, best for imbalanced classes)
- **Generalized Dice** — Weighted by class volume
- **Focal Loss** — For hard example mining

## Sliding Window Inference

For full-volume inference on large scans:

```python
from flashmed.tasks.segmentation import SegmentationTask

task = SegmentationTask(num_classes=4)
output = task.sliding_window_inference(
    model, volume, roi_size=(96, 96, 96), overlap=0.5,
)
```

## Metrics

- **Dice Score** — Per-class and mean
- **Hausdorff Distance (95th)** — Surface distance accuracy
