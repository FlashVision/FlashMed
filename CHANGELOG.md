# Changelog

All notable changes to FlashMed will be documented in this file.

## [1.0.0] - 2024-12-01

### Added
- Medical Vision Transformer (MedViT) for classification and pathology
- 3D UNet for volumetric CT/MRI segmentation
- Medical Vision-Language Model (MedVLM) for report generation
- Multi-label disease classification (ChestX-ray14)
- Lesion/nodule detection with FROC evaluation
- Whole slide image analysis with attention-based MIL
- Federated learning for multi-site training
- Differential privacy (DP-SGD) support
- DICOM de-identification following HIPAA Safe Harbor
- LoRA fine-tuning for parameter-efficient adaptation
- GradCAM visualization for clinical interpretability
- Diagnostic assistant with differential diagnosis
- Automated report writer
- Comprehensive metrics: AUC-ROC, Dice, sensitivity, specificity, FROC
- CLI with train, predict, diagnose, report, export, benchmark commands
- Support for ChestX-ray14, ISIC, BraTS, PathMNIST, MIMIC-CXR datasets
- Docker support for reproducible environments
- CI/CD with GitHub Actions
