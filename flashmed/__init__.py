"""FlashMed — Medical AI for imaging analysis, diagnostics, and report generation."""

__version__ = "1.0.0"

from flashmed.models.flashmed_model import FlashMed
from flashmed.engine.trainer import Trainer
from flashmed.engine.predictor import Predictor
from flashmed.engine.exporter import Exporter
from flashmed.solutions.diagnostic_assistant import DiagnosticAssistant
from flashmed.solutions.report_writer import ReportWriter
from flashmed.analytics.benchmark import Benchmark

__all__ = [
    "FlashMed",
    "Trainer",
    "Predictor",
    "Exporter",
    "DiagnosticAssistant",
    "ReportWriter",
    "Benchmark",
    "__version__",
]
