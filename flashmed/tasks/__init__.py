"""FlashMed task definitions for medical AI."""

from flashmed.tasks.classification import ClassificationTask
from flashmed.tasks.segmentation import SegmentationTask
from flashmed.tasks.detection import DetectionTask
from flashmed.tasks.report_gen import ReportGenerationTask
from flashmed.tasks.pathology import PathologyTask

__all__ = [
    "ClassificationTask",
    "SegmentationTask",
    "DetectionTask",
    "ReportGenerationTask",
    "PathologyTask",
]
