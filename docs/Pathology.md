# Pathology

Whole slide image (WSI) analysis for computational pathology.

## Overview

FlashMed supports patch-based analysis of large pathology images with Multiple Instance Learning (MIL) aggregation for slide-level predictions.

## Pipeline

1. **Patch Extraction** — Extract tissue patches, filter background
2. **Feature Extraction** — MedViT encodes each patch
3. **MIL Aggregation** — Attention-weighted combination
4. **Slide Prediction** — Final classification

## Training

```python
from flashmed import Trainer

trainer = Trainer(
    task="pathology",
    data_dir="data/pathmnist",
    num_classes=9,
    epochs=50,
    batch_size=64,
)
trainer.train()
```

## Slide Analysis

```python
from flashmed.solutions import PathologyAnalyzer

analyzer = PathologyAnalyzer(
    model_path="workspace/pathology/flashmed_best.pth",
    patch_size=256,
    batch_size=32,
)

result = analyzer.analyze("slide.png", generate_heatmap=True)
print(f"Prediction: {result['slide_prediction']['predicted_label']}")
print(f"Confidence: {result['slide_prediction']['mean_confidence']:.2%}")
```

## MIL Aggregation Methods

- **Attention** — Learned attention weights (default, best accuracy)
- **Mean** — Simple average pooling
- **Max** — Maximum pooling

## Heatmap Visualization

Generate spatial prediction heatmaps to identify diagnostic regions:

```python
result = analyzer.analyze("slide.png", generate_heatmap=True)
heatmap = result["heatmap"]  # [H, W] normalized attention map
```

## Tissue Types (PathMNIST)

1. Adipose
2. Background
3. Debris
4. Lymphocytes
5. Mucus
6. Smooth Muscle
7. Normal Colon Mucosa
8. Cancer Epithelium
9. Stroma
