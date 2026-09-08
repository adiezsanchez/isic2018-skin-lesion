"""Segmentation and classification metrics (NumPy / scikit-learn)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    dims = (2, 3) if preds.ndim == 4 else (1, 2)
    inter = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    iou_den = preds.sum(dim=dims) + targets.sum(dim=dims) - inter
    iou = (inter + 1e-6) / (iou_den + 1e-6)
    acc = (preds == targets).float().mean()
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "pixel_accuracy": float(acc.item()),
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: torch.Tensor | np.ndarray,
    logits: torch.Tensor | np.ndarray,
    class_names: tuple[str, ...] | list[str],
) -> dict:
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)
    if isinstance(logits, torch.Tensor):
        logits_t = logits.detach().cpu()
        proba = torch.softmax(logits_t, dim=1).numpy()
        y_pred = logits_t.argmax(dim=1).numpy()
    else:
        logits_np = np.asarray(logits)
        e = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
        y_pred = proba.argmax(axis=1)

    names = list(class_names)
    n_classes = len(names)
    cm = confusion_matrix(y_true_np, y_pred, labels=list(range(n_classes)))
    report = classification_report(
        y_true_np,
        y_pred,
        labels=list(range(n_classes)),
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    per_class_auc: dict[str, float] = {}
    roc_curves: dict[str, dict[str, list[float]]] = {}
    pr_curves: dict[str, dict[str, list[float]]] = {}
    for i, name in enumerate(names):
        y_bin = (y_true_np == i).astype(int)
        scores = proba[:, i]
        if y_bin.min() == y_bin.max():
            per_class_auc[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_bin, scores)
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        per_class_auc[name] = float(roc_auc_score(y_bin, scores))
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        pr_curves[name] = {"precision": prec.tolist(), "recall": rec.tolist()}

    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred)),
        "macro_f1": float(f1_score(y_true_np, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true_np, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_np, y_pred, average="macro", zero_division=0)),
        "macro_roc_auc": _safe_auc(y_true_np, proba),
        "macro_pr_auc": float(
            average_precision_score(
                np.eye(n_classes)[np.clip(y_true_np, 0, n_classes - 1)],
                proba,
                average="macro",
            )
        )
        if len(np.unique(y_true_np)) > 1
        else float("nan"),
        "per_class_roc_auc": per_class_auc,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "proba": proba.tolist(),
        "y_pred": y_pred.tolist(),
        "y_true": y_true_np.tolist(),
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "class_names": names,
    }
