# FlashMed

[![CI](https://github.com/FlashVision/FlashMed/actions/workflows/ci.yml/badge.svg)](https://github.com/FlashVision/FlashMed/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/flashmed)](https://pypi.org/project/flashmed/)

**Medical AI for imaging analysis, diagnostics, and radiology report generation.** Part of the [FlashVision](https://github.com/FlashVision) family.

## What is FlashMed?

FlashMed is a production-ready medical imaging framework supporting X-ray classification, CT/MRI segmentation, pathology analysis, and automated radiology report generation — with built-in privacy-preserving techniques.

**Key Features:**
- **Multi-task** — Classification, segmentation, detection, report generation, pathology
- **Medical ViT** — Vision Transformer optimized for medical imaging
- **3D UNet** — Volumetric segmentation for CT/MRI
- **Medical VLM** — Vision-language model for report generation
- **Privacy-first** — Federated learning, differential privacy, DICOM anonymization
- **DICOM native** — Load, process, and de-identify DICOM files
- **LoRA fine-tuning** — Parameter-efficient adaptation for medical domains
- **Clinical metrics** — AUC-ROC, Dice, sensitivity, specificity, FROC

## Supported Tasks

| Task | Model | Dataset | Metric |
|------|-------|---------|--------|
| X-ray Classification | MedViT | ChestX-ray14 | AUC-ROC |
| CT/MRI Segmentation | 3D UNet | BraTS | Dice Score |
| Pathology | MedViT + MIL | PathMNIST | Accuracy |
| Report Generation | MedVLM | MIMIC-CXR | BLEU-4, ROUGE-L |
| Lesion Detection | MedViT | Custom | FROC |

## Installation

```bash
pip install flashmed

# Or from source
git clone https://github.com/FlashVision/FlashMed.git
cd FlashMed
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from flashmed import FlashMed, Trainer, Predictor, DiagnosticAssistant

# Train a chest X-ray classifier
trainer = Trainer(
    task="classification",
    data_dir="data/chestxray14",
    num_classes=14,
    epochs=100,
    batch_size=16,
)
trainer.train()

# Inference
predictor = Predictor(model_path="workspace/train/flashmed_best.pth")
results = predictor.predict("chest_xray.png")
for finding in results["findings"]:
    print(f"{finding['label']}: {finding['confidence']:.2%}")

# Diagnostic assistant with GradCAM
assistant = DiagnosticAssistant(model_path="workspace/train/flashmed_best.pth")
diagnosis = assistant.diagnose("xray.dcm", include_gradcam=True)
```

### CLI

```bash
# Verify installation
flashmed check

# Train
flashmed train --task classification --data-dir data/ --num-classes 14

# Predict
flashmed predict --model best.pth --source xray.png --task classification

# Generate report
flashmed report --model vlm.pth --source chest_xray.png

# Run diagnostics
flashmed diagnose --model best.pth --source scan.dcm --modality ct

# Export to ONNX
flashmed export --model best.pth --output model.onnx

# Benchmark
flashmed benchmark --tasks classification,segmentation
```

## Privacy & Compliance

FlashMed includes built-in tools for medical data privacy:

```python
from flashmed.privacy import FederatedLearner, DifferentialPrivacy, DicomAnonymizer

# Federated learning across hospitals
fl = FederatedLearner(model, num_clients=5, rounds=50)
fl.train(client_datasets)

# Differential privacy training
dp = DifferentialPrivacy(model, epsilon=8.0, delta=1e-5)
model, optimizer, loader = dp.make_private(optimizer, data_loader)

# DICOM de-identification
anon = DicomAnonymizer(method="pseudonymize")
anon.anonymize_directory("raw_dicoms/", "anonymized/")
```

## Architecture

```
Input Image [B, C, H, W]
    │
    ▼
┌─────────────────────────┐
│  MedViT / 3D UNet /     │  Task-specific backbone
│  Medical VLM             │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Task Head               │  Classification / Segmentation /
│                          │  Detection / Report Generation
└─────────────────────────┘
    │
    ▼
  Output (logits / mask / report tokens)
```

## Documentation

- [Installation](docs/Installation.md)
- [Quick Start](docs/Quick-Start.md)
- [Classification](docs/Classification.md)
- [Segmentation](docs/Segmentation.md)
- [Report Generation](docs/Report-Generation.md)
- [Pathology](docs/Pathology.md)
- [FAQ](docs/FAQ.md)

## License

MIT License. See [LICENSE](LICENSE) for details.
