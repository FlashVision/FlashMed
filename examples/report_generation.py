"""Example: Automated radiology report generation from chest X-rays."""

from flashmed import FlashMed, Trainer
from flashmed.solutions.report_writer import ReportWriter
from flashmed.solutions.diagnostic_assistant import DiagnosticAssistant


def train_report_model():
    """Train a vision-language model for report generation."""
    trainer = Trainer(
        task="report_gen",
        data_dir="data/mimic_cxr",
        epochs=30,
        batch_size=8,
        learning_rate=5e-5,
        lora=True,
        device="cuda",
        save_dir="workspace/report_gen",
    )
    trainer.train()


def generate_report():
    """Generate a radiology report from a chest X-ray."""
    writer = ReportWriter(
        model_path="workspace/report_gen/flashmed_best.pth",
        device="cuda",
    )

    report = writer.generate("data/test_images/chest_xray_001.png", modality="CXR")
    print(report)


def run_diagnostic_assistant():
    """Use the diagnostic assistant for comprehensive analysis."""
    assistant = DiagnosticAssistant(
        model_path="workspace/xray_cls/flashmed_best.pth",
        device="cuda",
        threshold=0.5,
    )

    diagnosis = assistant.diagnose(
        "data/test_images/chest_xray_001.png",
        modality="xray",
        include_gradcam=True,
        include_differential=True,
    )

    print(f"Status: {diagnosis['overall_status']}")
    print(f"Findings: {diagnosis['num_abnormalities']}")
    for finding in diagnosis.get("findings", []):
        print(f"  - {finding['label']}: {finding['confidence']:.2%}")

    if "differential_diagnosis" in diagnosis:
        print("\nDifferential Diagnosis:")
        for diff in diagnosis["differential_diagnosis"]:
            print(f"  {diff['finding']}: {diff['differential']}")

    print(f"\nRecommendation: {diagnosis['recommendation']}")


if __name__ == "__main__":
    generate_report()
