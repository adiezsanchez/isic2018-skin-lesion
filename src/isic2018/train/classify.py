"""Task B: transfer-learning CNN with class-imbalance handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from isic2018.constants import CLASS_NAMES
from isic2018.data.datasets import (
    CropMode,
    LesionDataset,
    inverse_freq_weights,
    make_loader,
)
from isic2018.eval.metrics import classification_metrics
from isic2018.models.classifiers import build_classifier, freeze_backbone
from isic2018.models.losses import classification_loss
from isic2018.utils import LOGGER, dump_json, ensure_dir, get_device, seed_everything


def train_classifier(
    data_root: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    backbone: str,
    pretrained: bool,
    crop_mode: CropMode,
    imbalance: str,
    out_dir: Path,
    freeze_epochs: int = 0,
    pred_mask_dir: Path | None = None,
    seed: int = 7,
) -> dict:
    seed_everything(seed)
    device = get_device()
    LOGGER.info("Task B CNN (%s, crop=%s, imbalance=%s) on %s", backbone, crop_mode, imbalance, device)

    train_ds = LesionDataset(
        data_root,
        "train",
        image_size=image_size,
        crop_mode=crop_mode,
        augment=True,
        pred_mask_dir=pred_mask_dir,
        seed=seed,
    )
    val_ds = LesionDataset(
        data_root,
        "val",
        image_size=image_size,
        crop_mode=crop_mode,
        augment=False,
        pred_mask_dir=pred_mask_dir,
    )
    counts = train_ds.class_counts()
    weights = inverse_freq_weights(counts).to(device)
    use_sampler = imbalance == "sampler"
    train_loader = make_loader(
        train_ds, batch_size, shuffle=True, class_counts=counts, use_sampler=use_sampler
    )
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    model = build_classifier(len(CLASS_NAMES), backbone=backbone, pretrained=pretrained).to(device)
    if freeze_epochs > 0:
        freeze_backbone(model, True)
    criterion = classification_loss(imbalance, weights)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_auc = -1.0
    out_dir = ensure_dir(out_dir)
    ckpt_path = out_dir / "clf_best.pt"

    for epoch in range(1, epochs + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            freeze_backbone(model, False)
            optim = torch.optim.Adam(model.parameters(), lr=lr * 0.3)
            LOGGER.info("Unfroze backbone at epoch %d", epoch)

        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"clf {crop_mode} {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * images.size(0)
        train_loss = running / max(len(train_ds), 1)

        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                logits = model(images)
                logits_all.append(logits.cpu())
                labels_all.append(batch["label"])
        val_metrics = classification_metrics(
            torch.cat(labels_all), torch.cat(logits_all), CLASS_NAMES
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_roc_auc": val_metrics["macro_roc_auc"],
            }
        )
        LOGGER.info(
            "epoch %02d  loss=%.4f  acc=%.3f  bal-acc=%.3f  AUC=%.3f",
            epoch,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            val_metrics["macro_roc_auc"],
        )
        auc = val_metrics["macro_roc_auc"]
        score = auc if auc == auc else val_metrics["balanced_accuracy"]  # NaN-safe
        if score >= best_auc:
            best_auc = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": backbone,
                    "pretrained": pretrained,
                    "crop_mode": crop_mode,
                    "image_size": image_size,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )

    payload = {
        "history": history,
        "best_val_auc": best_auc,
        "checkpoint": str(ckpt_path),
        "crop_mode": crop_mode,
        "class_counts_train": counts.tolist(),
    }
    dump_json(payload, out_dir / "train_log.json")
    return payload


@torch.no_grad()
def collect_predictions(
    data_root: Path,
    ckpt: Path,
    split: str,
    crop_mode: CropMode,
    image_size: int,
    pred_mask_dir: Path | None = None,
    batch_size: int = 8,
) -> dict:
    device = get_device()
    bundle = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_classifier(
        len(CLASS_NAMES),
        backbone=bundle.get("backbone", "resnet18"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    ds = LesionDataset(
        data_root,
        split,
        image_size=image_size,
        crop_mode=crop_mode,
        pred_mask_dir=pred_mask_dir,
    )
    loader = make_loader(ds, batch_size, shuffle=False)
    logits_all, labels_all, ids = [], [], []
    images_cpu = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        logits_all.append(logits.cpu())
        labels_all.append(batch["label"])
        ids.extend(list(batch["image_id"]))
        images_cpu.append(batch["image"])
    logits_t = torch.cat(logits_all)
    labels_t = torch.cat(labels_all)
    metrics = classification_metrics(labels_t, logits_t, CLASS_NAMES)
    metrics["ids"] = ids
    metrics["logits"] = logits_t.numpy().tolist()
    metrics["y_true"] = labels_t.numpy().tolist()
    metrics["images"] = torch.cat(images_cpu).numpy()
    metrics["model"] = model
    metrics["checkpoint"] = str(ckpt)
    metrics["crop_mode"] = crop_mode
    return metrics
