from __future__ import annotations

import torch

from isic2018.eval.metrics import classification_metrics, segmentation_metrics


def test_perfect_mask_has_unit_dice():
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    # logits that saturate the positive region
    logits = (mask * 20) - 10
    stats = segmentation_metrics(logits, mask)
    assert stats["dice"] > 0.99
    assert stats["iou"] > 0.99


def test_classification_metrics_keys():
    logits = torch.tensor([[4.0, 0.1, 0.0], [0.1, 3.0, 0.0], [0.0, 0.1, 2.5]])
    y = torch.tensor([0, 1, 2])
    names = ("a", "b", "c")
    metrics = classification_metrics(y, logits, names)
    assert metrics["accuracy"] == 1.0
    assert "macro_roc_auc" in metrics
    assert metrics["confusion_matrix"][0][0] == 1
