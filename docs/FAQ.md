# FAQ

## General

**Q: Is FlashMed suitable for clinical use?**
A: FlashMed is a research and development tool. Clinical deployment requires proper validation, regulatory approval, and integration with clinical workflows. Always treat AI outputs as decision support, not diagnostic conclusions.

**Q: What GPU do I need?**
A: Classification and pathology work on GPUs with 8GB+ VRAM. 3D segmentation benefits from 16GB+. CPU-only inference is supported but slower.

**Q: Can I use my own dataset?**
A: Yes. Organize images in a directory structure, create a labels CSV, or use the existing dataset classes as templates for custom loaders.

## Training

**Q: How do I handle class imbalance in medical datasets?**
A: Use class weights (`ClassificationTask.compute_class_weights()`), focal loss for segmentation, or data augmentation. Medical datasets are inherently imbalanced.

**Q: Should I use LoRA?**
A: LoRA is recommended when fine-tuning on small medical datasets (< 10K images). It reduces overfitting while training only ~1-5% of parameters.

**Q: How long does training take?**
A: Chest X-ray classification: ~2-4 hours on a single GPU. BraTS segmentation: ~12-24 hours. Report generation: ~6-12 hours with LoRA.

## Privacy

**Q: How does federated learning work?**
A: Each hospital trains a local model, then only model weights (not data) are aggregated centrally. Patient data never leaves the institution.

**Q: What privacy guarantee does DP-SGD provide?**
A: With epsilon=8.0, delta=1e-5, the model provides formal mathematical guarantees that any single training example has bounded influence on the model output.

**Q: Is the DICOM anonymization HIPAA compliant?**
A: It follows HIPAA Safe Harbor de-identification guidelines, removing all 18 PHI categories. However, compliance also depends on institutional policies and proper deployment.

## Deployment

**Q: How do I export for production?**
A: Use `flashmed export --model best.pth --output model.onnx` for ONNX Runtime deployment, or `--format torchscript` for PyTorch serving.

**Q: Can I run inference on DICOM files directly?**
A: Yes. All inference tools accept `.dcm` files and automatically handle DICOM pixel data extraction, windowing, and preprocessing.
