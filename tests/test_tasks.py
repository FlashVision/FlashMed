"""Tests for FlashMed task implementations."""

import torch


class TestClassificationTask:
    def test_multi_label_loss(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=14, multi_label=True)
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()
        loss = task.compute_loss(logits, targets)
        assert loss.item() > 0

    def test_single_label_loss(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=9, multi_label=False)
        logits = torch.randn(4, 9)
        targets = torch.randint(0, 9, (4,))
        loss = task.compute_loss(logits, targets)
        assert loss.item() > 0

    def test_predictions(self):
        from flashmed.tasks.classification import ClassificationTask

        task = ClassificationTask(num_classes=14, multi_label=True, threshold=0.5)
        logits = torch.randn(4, 14)
        result = task.compute_predictions(logits)
        assert "probabilities" in result
        assert "predictions" in result
        assert result["probabilities"].shape == (4, 14)


class TestSegmentationTask:
    def test_dice_loss(self):
        from flashmed.tasks.segmentation import DiceLoss

        loss_fn = DiceLoss(num_classes=4)
        pred = torch.randn(2, 4, 32, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32, 32))
        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= 1

    def test_dice_score(self):
        from flashmed.tasks.segmentation import SegmentationTask

        task = SegmentationTask(num_classes=4, spatial_dims=3)
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))
        scores = task.compute_dice_score(pred, target)
        assert "dice_mean" in scores
        assert 0 <= scores["dice_mean"] <= 1


class TestReportGenTask:
    def test_loss_computation(self):
        from flashmed.tasks.report_gen import ReportGenerationTask

        task = ReportGenerationTask(vocab_size=1000, max_length=64)
        logits = torch.randn(2, 64, 1000)
        target_ids = torch.randint(1, 1000, (2, 64))
        loss = task.compute_loss(logits, target_ids)
        assert loss.item() > 0

    def test_bleu_rouge(self):
        from flashmed.tasks.report_gen import ReportGenerationTask

        task = ReportGenerationTask()
        generated = ["the heart size is normal no acute findings"]
        references = ["heart size is normal with no acute cardiopulmonary findings"]
        metrics = task.compute_metrics(generated, references)
        assert "bleu-4" in metrics
        assert "rouge-l" in metrics


class TestDetectionTask:
    def test_nms(self):
        from flashmed.tasks.detection import nms

        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float)
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_iou_computation(self):
        from flashmed.tasks.detection import compute_iou

        boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
        boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
        iou = compute_iou(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([[1.0]]))


class TestMetrics:
    def test_auc_roc(self):
        from flashmed.analytics.metrics import compute_auc_roc

        preds = torch.randn(100, 14)
        targets = torch.randint(0, 2, (100, 14)).float()
        auc = compute_auc_roc(preds, targets)
        assert 0 <= auc <= 1

    def test_sensitivity_specificity(self):
        from flashmed.analytics.metrics import compute_sensitivity, compute_specificity

        preds = torch.randn(100, 14)
        targets = torch.randint(0, 2, (100, 14)).float()
        sens = compute_sensitivity(preds, targets)
        spec = compute_specificity(preds, targets)
        assert 0 <= sens <= 1
        assert 0 <= spec <= 1

    def test_f1_score(self):
        from flashmed.analytics.metrics import compute_f1_score

        preds = torch.randn(50, 14)
        targets = torch.randint(0, 2, (50, 14)).float()
        f1 = compute_f1_score(preds, targets)
        assert 0 <= f1 <= 1
