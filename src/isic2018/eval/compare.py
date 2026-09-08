"""Orchestrate evaluations and the whole-image vs crop vs classical comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from isic2018.constants import CLASS_NAMES
from isic2018.data.datasets import LesionDataset, make_loader
from isic2018.eval.metrics import segmentation_metrics
from isic2018.eval.plots import (
    plot_confusion,
    plot_gradcam_grid,
    plot_metric_bars,
    plot_model_comparison,
    plot_pr,
    plot_roc,
    plot_training_curves,
)
from isic2018.models.gradcam import GradCAM, overlay_cam
from isic2018.models.unet import UNet
from isic2018.train.classify import collect_predictions
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from isic2018.utils import LOGGER, dump_json, get_device


def evaluate_segmentation(
    data_root: Path,
    ckpt: Path,
    figures: Path,
    image_size: int,
    split: str = "test",
) -> dict:
    device = get_device()
    bundle = torch.load(ckpt, map_location=device, weights_only=False)
    model = UNet(base_channels=int(bundle.get("base_channels", 16))).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    ds = LesionDataset(data_root, split, image_size=image_size)
    loader = make_loader(ds, batch_size=8, shuffle=False)
    dices, ious, accs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            stats = segmentation_metrics(logits, batch["mask"].to(device))
            dices.append(stats["dice"])
            ious.append(stats["iou"])
            accs.append(stats["pixel_accuracy"])
    metrics = {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "pixel_accuracy": float(np.mean(accs)),
        "split": split,
    }
    train_log_path = ckpt.parent / "train_log.json"
    if train_log_path.exists():
        import json

        history = json.loads(train_log_path.read_text(encoding="utf-8")).get("history", [])
        plot_training_curves(history, "Task A U-Net training", figures / "task_a_training")
    plot_metric_bars(
        {k: metrics[k] for k in ("dice", "iou", "pixel_accuracy")},
        f"Task A segmentation on {split}",
        figures / "task_a_metrics",
    )
    dump_json(metrics, ckpt.parent / "test_metrics.json")
    LOGGER.info("Task A %s  Dice=%.3f  IoU=%.3f", split, metrics["dice"], metrics["iou"])
    return metrics


def evaluate_classifier(
    data_root: Path,
    ckpt: Path,
    figures: Path,
    image_size: int,
    crop_mode: str,
    split: str = "test",
    pred_mask_dir: Path | None = None,
    tag: str | None = None,
) -> dict:
    tag = tag or f"clf_{crop_mode}"
    result = collect_predictions(
        data_root, ckpt, split, crop_mode, image_size, pred_mask_dir=pred_mask_dir
    )
    model = result.pop("model")
    images = result.pop("images")
    plot_confusion(
        result["confusion_matrix"],
        list(CLASS_NAMES),
        f"Confusion — {tag}",
        figures / f"{tag}_confusion",
    )
    plot_roc(result["roc_curves"], f"ROC — {tag}", figures / f"{tag}_roc")
    plot_pr(result["pr_curves"], f"PR — {tag}", figures / f"{tag}_pr")
    plot_metric_bars(
        {
            "accuracy": result["accuracy"],
            "balanced_acc": result["balanced_accuracy"],
            "macro_f1": result["macro_f1"],
            "macro_AUC": result["macro_roc_auc"] if result["macro_roc_auc"] == result["macro_roc_auc"] else 0.0,
        },
        f"Task B metrics — {tag}",
        figures / f"{tag}_metrics",
    )
    log_path = ckpt.parent / "train_log.json"
    if log_path.exists():
        import json

        history = json.loads(log_path.read_text(encoding="utf-8")).get("history", [])
        plot_training_curves(history, f"Task B training — {tag}", figures / f"{tag}_training")

    # Grad-CAM on a handful of test images.
    device = get_device()
    model.to(device)
    n = min(8, images.shape[0])
    tensor = torch.from_numpy(images[:n]).to(device)
    with GradCAM(model) as cam:
        cams, preds = cam(tensor)
    overlays = np.stack(
        [overlay_cam(images[i], cams[i]) for i in range(n)],
        axis=0,
    )
    titles = [
        f"{result['ids'][i]}→{CLASS_NAMES[int(preds[i])]}"
        for i in range(n)
    ]
    plot_gradcam_grid(images[:n], overlays, titles, figures / f"{tag}_gradcam")
    result.pop("logits", None)
    dump_json(
        {k: v for k, v in result.items() if k not in {"proba", "roc_curves", "pr_curves", "report"}},
        ckpt.parent / "test_metrics.json",
    )
    LOGGER.info(
        "Task B %s  acc=%.3f  bal-acc=%.3f  AUC=%.3f",
        tag,
        result["accuracy"],
        result["balanced_accuracy"],
        result["macro_roc_auc"],
    )
    return result


def compare_methods(
    cnn_results: list[dict[str, Any]],
    classical: dict[str, Any],
    figures: Path,
) -> dict:
    rows = []
    for item in cnn_results:
        rows.append(
            {
                "name": item.get("name", item.get("crop_mode", "cnn")),
                "accuracy": item.get("accuracy"),
                "balanced_accuracy": item.get("balanced_accuracy"),
                "macro_roc_auc": item.get("macro_roc_auc"),
            }
        )
    for key in ("random_forest", "xgboost"):
        if key in classical:
            block = classical[key]
            # Reconstruct balanced accuracy from stored arrays if needed.
            y_true = np.asarray(block["y_true"])
            y_pred = np.asarray(block["y_pred"])
            rows.append(
                {
                    "name": key,
                    "accuracy": block["accuracy"],
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "macro_roc_auc": block["macro_roc_auc"],
                }
            )
            plot_confusion(
                confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))),
                list(CLASS_NAMES),
                f"Confusion — {key}",
                figures / f"{key}_confusion",
            )
    plot_model_comparison(rows, figures / "method_comparison")
    payload = {"rows": rows}
    dump_json(payload, figures.parent / "runs" / "comparison.json")
    return payload
