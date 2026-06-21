"""Benchmarking suite for FlashMed models across tasks and hardware."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class Benchmark:
    """Comprehensive benchmarking for FlashMed models.

    Measures throughput, latency, memory usage, and accuracy across
    different tasks, model sizes, and hardware configurations.

    Args:
        device: Benchmarking device
        warmup_iterations: Number of warmup forward passes
        benchmark_iterations: Number of timed forward passes
    """

    def __init__(
        self,
        device: str = "cuda",
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def run(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run full benchmark suite.

        Args:
            tasks: List of tasks to benchmark. If None, benchmarks all.

        Returns:
            Dictionary of benchmark results
        """
        if tasks is None:
            tasks = ["classification", "segmentation", "pathology"]

        results = {}
        print(f"\n{'='*70}")
        print("  FlashMed Benchmark Suite")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")

        for task in tasks:
            print(f"  Benchmarking: {task}")
            result = self._benchmark_task(task)
            results[task] = result
            self._print_result(task, result)

        print(f"\n{'='*70}")
        print("  Benchmark Complete")
        print(f"{'='*70}")

        return results

    def _benchmark_task(self, task: str) -> Dict[str, Any]:
        """Benchmark a single task."""
        from flashmed.models.flashmed_model import FlashMed

        configs = {
            "classification": {"num_classes": 14, "input_size": 224, "in_channels": 3},
            "segmentation": {"num_classes": 4, "input_size": 96, "in_channels": 4, "spatial_dims": 3},
            "pathology": {"num_classes": 9, "input_size": 256, "in_channels": 3},
        }

        cfg = configs.get(task, configs["classification"])
        model = FlashMed(task=task, pretrained=False, **cfg).to(self.device)
        model.eval()

        if task == "segmentation":
            input_shape = (1, cfg["in_channels"], cfg["input_size"], cfg["input_size"], cfg["input_size"])
        else:
            input_shape = (1, cfg["in_channels"], cfg["input_size"], cfg["input_size"])

        throughput = self._measure_throughput(model, input_shape)
        latency = self._measure_latency(model, input_shape)
        memory = self._measure_memory(model, input_shape)
        params = sum(p.numel() for p in model.parameters())

        return {
            "parameters": params,
            "throughput_fps": throughput,
            "latency_ms": latency,
            "memory_mb": memory,
            "input_shape": input_shape,
        }

    @torch.no_grad()
    def _measure_throughput(self, model: nn.Module, input_shape: tuple) -> float:
        """Measure inference throughput (images/second)."""
        dummy = torch.randn(*input_shape, device=self.device)

        for _ in range(self.warmup_iterations):
            model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(self.benchmark_iterations):
            model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return self.benchmark_iterations / elapsed

    @torch.no_grad()
    def _measure_latency(self, model: nn.Module, input_shape: tuple) -> float:
        """Measure per-image latency in milliseconds."""
        dummy = torch.randn(*input_shape, device=self.device)

        for _ in range(self.warmup_iterations):
            model(dummy)

        latencies = []
        for _ in range(self.benchmark_iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        return float(np.median(latencies))

    def _measure_memory(self, model: nn.Module, input_shape: tuple) -> float:
        """Measure peak GPU memory usage in MB."""
        if self.device.type != "cuda":
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            return param_size / (1024 * 1024)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        dummy = torch.randn(*input_shape, device=self.device)
        with torch.no_grad():
            model(dummy)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return float(peak_memory)

    def _print_result(self, task: str, result: Dict[str, Any]):
        """Print formatted benchmark result."""
        print(f"    Parameters:  {result['parameters']:,}")
        print(f"    Throughput:  {result['throughput_fps']:.1f} FPS")
        print(f"    Latency:     {result['latency_ms']:.2f} ms")
        print(f"    Memory:      {result['memory_mb']:.1f} MB")
        print()

    def compare_models(self, model_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare multiple model configurations side by side."""
        from flashmed.models.flashmed_model import FlashMed

        results = []
        for cfg in model_configs:
            model = FlashMed(pretrained=False, **cfg).to(self.device)
            model.eval()

            input_size = cfg.get("input_size", 224)
            in_channels = cfg.get("in_channels", 3)
            input_shape = (1, in_channels, input_size, input_size)

            result = {
                "config": cfg,
                "throughput_fps": self._measure_throughput(model, input_shape),
                "latency_ms": self._measure_latency(model, input_shape),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            results.append(result)

        return results
