# Classification

Multi-label disease classification from medical images.

## Overview

FlashMed's classification module uses MedViT (Medical Vision Transformer) for multi-label disease detection, supporting 14 pathologies from the ChestX-ray14 dataset.

## Supported Datasets

- **ChestX-ray14** — 112K frontal X-rays, 14 pathologies
- **ISIC** — Dermoscopy images, 9 skin lesion types
- **PathMNIST** — Colon pathology, 9 tissue types
- **Custom** — Any ImageFolder-format dataset

## Training

### Python API

```python
from flashmed import Trainer

trainer = Trainer(
    task="classification",
    data_dir="data/chestxray14",
    num_classes=14,
    epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    lora=True,  # Parameter-efficient fine-tuning
)
trainer.train()
```

### YAML Config

```bash
flashmed train --config configs/flashmed_xray_cls.yaml
```

## Multi-Label vs Single-Label

Multi-label (X-rays with multiple findings):
```python
trainer = Trainer(task="classification", num_classes=14)  # BCE loss
```

Single-label (pathology classification):
```python
trainer = Trainer(task="pathology", num_classes=9)  # CE loss
```

## Metrics

- **AUC-ROC** — Primary metric for multi-label
- **Sensitivity** — True positive rate (critical for screening)
- **Specificity** — True negative rate
- **F1 Score** — Harmonic mean of precision and recall

## LoRA Fine-Tuning

Adapt pre-trained models with minimal parameters:

```python
trainer = Trainer(
    task="classification",
    lora=True,
    lora_rank=8,
    lora_alpha=16,
)
```
