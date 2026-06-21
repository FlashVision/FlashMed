# FlashMed Documentation

Welcome to FlashMed — Medical AI for imaging analysis, diagnostics, and radiology report generation.

## Overview

FlashMed provides a unified framework for:

- **Classification** — Multi-label disease detection from X-rays (14 pathologies)
- **Segmentation** — 2D/3D organ and lesion segmentation (BraTS, CT)
- **Detection** — Nodule and lesion localization
- **Report Generation** — Automated radiology reports using VLMs
- **Pathology** — Whole slide image analysis with MIL

## Quick Links

| Page | Description |
|------|-------------|
| [Installation](Installation.md) | Setup instructions |
| [Quick Start](Quick-Start.md) | Get running in 5 minutes |
| [Classification](Classification.md) | X-ray disease classification |
| [Segmentation](Segmentation.md) | CT/MRI segmentation |
| [Report Generation](Report-Generation.md) | Automated radiology reports |
| [Pathology](Pathology.md) | Whole slide image analysis |
| [FAQ](FAQ.md) | Frequently asked questions |

## Design Philosophy

1. **Medical-first** — Built for clinical imaging workflows, not adapted from generic CV
2. **Privacy-preserving** — Federated learning, DP-SGD, DICOM anonymization built-in
3. **Production-ready** — ONNX export, Docker deployment, comprehensive testing
4. **Interpretable** — GradCAM attention maps, structured reports, confidence calibration
