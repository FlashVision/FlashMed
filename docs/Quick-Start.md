# Quick Start

Get FlashMed running in under 5 minutes.

## 1. Install

```bash
pip install flashmed
flashmed check
```

## 2. Train a Chest X-ray Classifier

```python
from flashmed import Trainer

trainer = Trainer(
    task="classification",
    data_dir="data/chestxray14",
    num_classes=14,
    epochs=50,
    batch_size=16,
)
trainer.train()
```

## 3. Run Inference

```python
from flashmed import Predictor

predictor = Predictor(model_path="workspace/train/flashmed_best.pth")
results = predictor.predict("chest_xray.png")

for finding in results["findings"]:
    print(f"{finding['label']}: {finding['confidence']:.2%}")
```

## 4. Get a Diagnosis

```python
from flashmed import DiagnosticAssistant

assistant = DiagnosticAssistant(model_path="workspace/train/flashmed_best.pth")
report = assistant.diagnose("xray.dcm", include_gradcam=True)
print(report["recommendation"])
```

## 5. CLI Usage

```bash
flashmed train --task classification --data-dir data/ --epochs 50
flashmed predict --model best.pth --source xray.png
flashmed diagnose --model best.pth --source scan.dcm --modality xray
flashmed export --model best.pth --output model.onnx
```

## Next Steps

- [Classification Guide](Classification.md) — Detailed classification tutorial
- [Segmentation Guide](Segmentation.md) — 3D volume segmentation
- [Report Generation](Report-Generation.md) — Automated reports
