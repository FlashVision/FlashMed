# Report Generation

Automated radiology report generation from medical images.

## Overview

FlashMed's MedVLM (Medical Vision-Language Model) combines a visual encoder with a text decoder to generate structured radiology reports from X-rays, CT scans, and MRI.

## Architecture

1. **Visual Encoder** — ViT-based image feature extraction
2. **Cross-Attention** — Text tokens attend to visual features
3. **Text Decoder** — Autoregressive report generation

## Training

```python
from flashmed import Trainer

trainer = Trainer(
    task="report_gen",
    data_dir="data/mimic_cxr",
    epochs=30,
    batch_size=8,
    learning_rate=5e-5,
    lora=True,
)
trainer.train()
```

## Report Generation

```python
from flashmed.solutions import ReportWriter

writer = ReportWriter(model_path="workspace/report_gen/flashmed_best.pth")
report = writer.generate("chest_xray.png", modality="CXR")
print(report)
```

Output:
```
RADIOLOGY REPORT
================
Modality: CXR
Date: 2024-12-01

FINDINGS:
- Cardiomegaly: significant (85% confidence)
- Effusion: mild (62% confidence)

IMPRESSION:
Findings suggestive of: Cardiomegaly, Effusion.

RECOMMENDATION:
Clinical correlation recommended.
```

## Generation Methods

- **Greedy** — Fast, deterministic
- **Beam Search** — Higher quality, configurable width
- **Sampling** — Temperature-controlled diversity

## Metrics

- **BLEU-4** — N-gram precision
- **ROUGE-L** — Longest common subsequence
- **Clinical accuracy** — Finding detection F1
