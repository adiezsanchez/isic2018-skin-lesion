"""Task A: train a U-Net lesion segmenter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from isic2018.data.datasets import LesionDataset, make_loader
from isic2018.eval.metrics import segmentation_metrics
from isic2018.models.losses import BCEDiceLoss
from isic2018.models.unet import UNet
from isic2018.utils import LOGGER, dump_json, ensure_dir, get_device, seed_everything


def train_unet(
    data_root: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    base_channels: int,
    dice_weight: float,
    out_dir: Path,
    seed: int = 7,
) -> dict:
    seed_everything(seed)
    device = get_device()
    LOGGER.info("Task A U-Net on %s", device)
    train_ds = LesionDataset(data_root, "train", image_size=image_size, augment=True, seed=seed)
    val_ds = LesionDataset(data_root, "val", image_size=image_size, augment=False)
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    model = UNet(base_channels=base_channels).to(device)
    criterion = BCEDiceLoss(dice_weight=dice_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_dice = -1.0
    out_dir = ensure_dir(out_dir)
    ckpt_path = out_dir / "unet_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"seg epoch {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * images.size(0)
        train_loss = running / max(len(train_ds), 1)

        model.eval()
        dice_vals, iou_vals = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                stats = segmentation_metrics(logits, masks)
                dice_vals.append(stats["dice"])
                iou_vals.append(stats["iou"])
        val_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
        val_iou = float(np.mean(iou_vals)) if iou_vals else 0.0
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice, "val_iou": val_iou}
        )
        LOGGER.info(
            "epoch %02d  loss=%.4f  val Dice=%.3f  val IoU=%.3f",
            epoch,
            train_loss,
            val_dice,
            val_iou,
        )
        if val_dice >= best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "base_channels": base_channels,
                    "image_size": image_size,
                    "val_dice": val_dice,
                },
                ckpt_path,
            )

    payload = {"history": history, "best_val_dice": best_dice, "checkpoint": str(ckpt_path)}
    dump_json(payload, out_dir / "train_log.json")
    return payload


@torch.no_grad()
def predict_masks(
    data_root: Path,
    ckpt: Path,
    split: str,
    out_dir: Path,
    image_size: int,
    batch_size: int = 8,
) -> Path:
    device = get_device()
    bundle = torch.load(ckpt, map_location=device, weights_only=True)
    model = UNet(base_channels=int(bundle.get("base_channels", 16))).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    ds = LesionDataset(data_root, split, image_size=image_size, augment=False)
    loader = make_loader(ds, batch_size, shuffle=False)
    out_dir = ensure_dir(out_dir)
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8) * 255
        for i, image_id in enumerate(batch["image_id"]):
            Image.fromarray(preds[i, 0]).save(out_dir / f"{image_id}.png")
    return out_dir
