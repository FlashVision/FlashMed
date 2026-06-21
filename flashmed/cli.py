"""FlashMed CLI — command-line interface for medical AI tasks."""

import argparse
import sys


def _colored(text, color):
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashMed", "bold") + f" v{_get_version()}")
    print(_colored("Medical AI — Imaging, Diagnostics & Report Generation", "blue"))
    print()


def _get_version():
    from flashmed import __version__
    return __version__


def cmd_version(args):
    _print_banner()


def cmd_settings(args):
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    try:
        import monai
        print(f"  MONAI:       {monai.__version__}")
    except ImportError:
        print("  MONAI:       Not installed")
    try:
        import pydicom
        print(f"  pydicom:     {pydicom.__version__}")
    except ImportError:
        print("  pydicom:     Not installed")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashmed  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashmed package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashmed package: {e}")
        errors.append(str(e))

    try:
        from flashmed.engine import Trainer, Predictor, Exporter  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashmed.solutions import DiagnosticAssistant, ReportWriter  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (DiagnosticAssistant, ReportWriter)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashmed.privacy import FederatedLearner, DifferentialPrivacy  # noqa: F401
        print(f"  {_colored('✓', 'green')} privacy (FederatedLearner, DifferentialPrivacy)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} privacy: {e}")
        errors.append(str(e))

    try:
        import torch
        from flashmed.models.flashmed_model import FlashMed
        model = FlashMed(task="classification", num_classes=14, pretrained=False)
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224))
        print(f"  {_colored('✓', 'green')} model forward pass (FlashMed classification, 224px)")
    except Exception as e:
        print(f"  {_colored('✗', 'red')} model forward pass: {e}")
        errors.append(str(e))

    try:
        import pydicom  # noqa: F401
        print(f"  {_colored('✓', 'green')} DICOM support (pydicom)")
    except ImportError:
        print(f"  {_colored('⚠', 'yellow')} pydicom not installed (DICOM features unavailable)")

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (training will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashMed is ready.", "green"))


def cmd_train(args):
    from flashmed.engine.trainer import Trainer

    if args.config:
        from flashmed.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        kwargs = {
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "data_dir": args.data_dir,
            "save_dir": args.save_dir,
        }
        if args.num_classes:
            kwargs["num_classes"] = args.num_classes
        if args.lr:
            kwargs["learning_rate"] = args.lr
        if args.lora:
            kwargs["lora"] = True
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_predict(args):
    from flashmed.engine.predictor import Predictor

    predictor = Predictor(model_path=args.model, task=args.task, device=args.device)
    results = predictor.predict(args.source)

    print(f"\n{_colored('Prediction results:', 'green')}")
    if isinstance(results, dict):
        for key, val in results.items():
            print(f"  {key}: {val}")
    elif isinstance(results, list):
        for item in results[:10]:
            print(f"  {item}")


def cmd_diagnose(args):
    from flashmed.solutions.diagnostic_assistant import DiagnosticAssistant

    assistant = DiagnosticAssistant(model_path=args.model, device=args.device)
    report = assistant.diagnose(args.source, modality=args.modality)

    print(f"\n{_colored('Diagnostic Report:', 'green')}")
    for key, val in report.items():
        print(f"  {key}: {val}")


def cmd_report(args):
    from flashmed.solutions.report_writer import ReportWriter

    writer = ReportWriter(model_path=args.model, device=args.device)
    report_text = writer.generate(args.source)

    print(f"\n{_colored('Generated Report:', 'green')}")
    print(report_text)


def cmd_export(args):
    from flashmed.engine.exporter import Exporter
    exporter = Exporter(model_path=args.model, task=args.task)
    path = exporter.export(output=args.output, format=args.format)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def cmd_benchmark(args):
    from flashmed.analytics.benchmark import Benchmark
    bench = Benchmark(device=args.device)
    bench.run(tasks=args.tasks.split(",") if args.tasks else None)


def main():
    parser = argparse.ArgumentParser(
        prog="flashmed",
        description="FlashMed: Medical AI for imaging, diagnostics, and report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashmed check                                   Verify installation
  flashmed train --task classification --data-dir data/ --num-classes 14
  flashmed predict --model best.pth --source xray.dcm --task classification
  flashmed diagnose --model best.pth --source ct_scan.dcm --modality ct
  flashmed report --model best.pth --source chest_xray.png
  flashmed export --model best.pth --output model.onnx

Documentation: https://github.com/FlashVision/FlashMed
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("version", help="Show version info")
    subparsers.add_parser("settings", help="Show system settings")
    subparsers.add_parser("check", help="Verify installation and run health check")

    train_p = subparsers.add_parser("train", help="Train a medical AI model")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--task", default="classification",
                         choices=["classification", "segmentation", "detection", "report_gen", "pathology"])
    train_p.add_argument("--data-dir", default=None, help="Path to dataset")
    train_p.add_argument("--num-classes", type=int, default=None)
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--save-dir", default="workspace/train", help="Output directory")
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")

    pred_p = subparsers.add_parser("predict", help="Run inference on medical images")
    pred_p.add_argument("--model", required=True, help="Path to checkpoint")
    pred_p.add_argument("--source", required=True, help="Image/DICOM path or directory")
    pred_p.add_argument("--task", default="classification",
                        choices=["classification", "segmentation", "detection", "report_gen", "pathology"])
    pred_p.add_argument("--device", default="cuda")

    diag_p = subparsers.add_parser("diagnose", help="Run diagnostic assistant")
    diag_p.add_argument("--model", required=True, help="Path to checkpoint")
    diag_p.add_argument("--source", required=True, help="Image/DICOM path")
    diag_p.add_argument("--modality", default="xray", choices=["xray", "ct", "mri"])
    diag_p.add_argument("--device", default="cuda")

    rep_p = subparsers.add_parser("report", help="Generate radiology report")
    rep_p.add_argument("--model", required=True, help="Path to checkpoint")
    rep_p.add_argument("--source", required=True, help="Image/DICOM path")
    rep_p.add_argument("--device", default="cuda")

    exp_p = subparsers.add_parser("export", help="Export model")
    exp_p.add_argument("--model", required=True, help="Path to checkpoint")
    exp_p.add_argument("--task", default="classification")
    exp_p.add_argument("--output", default="model.onnx", help="Output path")
    exp_p.add_argument("--format", default="onnx", choices=["onnx", "torchscript"])

    bench_p = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_p.add_argument("--device", default="cuda")
    bench_p.add_argument("--tasks", default=None, help="Comma-separated tasks to benchmark")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "predict": cmd_predict,
        "diagnose": cmd_diagnose,
        "report": cmd_report,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
