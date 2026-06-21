"""Example: Benchmark FlashMed models across tasks and configurations."""

from flashmed.analytics.benchmark import Benchmark
from flashmed.analytics.metrics import compute_auc_roc, compute_dice_score

import torch


def benchmark_all_tasks():
    """Benchmark classification, segmentation, and pathology models."""
    bench = Benchmark(device="cuda", warmup_iterations=10, benchmark_iterations=100)
    results = bench.run(tasks=["classification", "segmentation", "pathology"])
    return results


def compare_model_sizes():
    """Compare different model configurations."""
    bench = Benchmark(device="cuda")

    configs = [
        {"task": "classification", "num_classes": 14, "input_size": 224},
        {"task": "classification", "num_classes": 14, "input_size": 384},
        {"task": "pathology", "num_classes": 9, "input_size": 256},
    ]

    results = bench.compare_models(configs)
    print("\nModel Comparison:")
    print(f"{'Config':<50} {'Params':>12} {'FPS':>8} {'Latency':>10}")
    print("-" * 82)
    for r in results:
        cfg_str = f"{r['config']['task']}@{r['config']['input_size']}"
        print(f"{cfg_str:<50} {r['parameters']:>12,} {r['throughput_fps']:>8.1f} {r['latency_ms']:>8.2f}ms")


def evaluate_metrics():
    """Demonstrate metric computation."""
    num_samples = 100
    num_classes = 14

    predictions = torch.randn(num_samples, num_classes)
    targets = torch.randint(0, 2, (num_samples, num_classes)).float()

    auc = compute_auc_roc(predictions, targets)
    print(f"AUC-ROC: {auc:.4f}")

    seg_pred = torch.randn(4, 4, 96, 96, 96)
    seg_target = torch.randint(0, 4, (4, 96, 96, 96))
    dice = compute_dice_score(seg_pred, seg_target, num_classes=4)
    print(f"Dice Score: {dice:.4f}")


if __name__ == "__main__":
    benchmark_all_tasks()
