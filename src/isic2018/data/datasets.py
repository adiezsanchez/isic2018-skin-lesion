"""Datasets, lesion crops, and simple geometric transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from isic2018.constants import CLASS_NAMES, CLASS_TO_IDX

CropMode = Literal["whole", "crop_gt", "crop_pred"]


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image)


def load_mask(path: Path, size: int | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size is not None:
        mask = mask.resize((size, size), Image.NEAREST)
    arr = np.asarray(mask)
    return (arr > 127).astype(np.uint8)


def lesion_bbox(mask: np.ndarray, margin: float = 0.15) -> tuple[int, int, int, int]:
    """Axis-aligned box around the positive mask, with a relative margin."""
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(ys) == 0:
        return 0, 0, w, h
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    y0 = max(0, int(y0 - margin * bh))
    x0 = max(0, int(x0 - margin * bw))
    y1 = min(h, int(y1 + margin * bh))
    x1 = min(w, int(x1 + margin * bw))
    return x0, y0, x1, y1


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = lesion_bbox(mask, margin=margin)
    crop_img = Image.fromarray(image[y0:y1, x0:x1]).resize((size, size), Image.BILINEAR)
    crop_mask = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )
    return np.asarray(crop_img), (np.asarray(crop_mask) > 127).astype(np.uint8)


def to_image_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB → float CHW in [0, 1]."""
    arr = image.astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32)[None].copy())


def random_augment(
    image: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Cheap numpy augmentations (no albumentations dependency)."""
    if rng.random() < 0.5:
        image = np.ascontiguousarray(np.fliplr(image))
        if mask is not None:
            mask = np.ascontiguousarray(np.fliplr(mask))
    if rng.random() < 0.5:
        image = np.ascontiguousarray(np.flipud(image))
        if mask is not None:
            mask = np.ascontiguousarray(np.flipud(mask))
    k = int(rng.integers(0, 4))
    if k:
        image = np.ascontiguousarray(np.rot90(image, k))
        if mask is not None:
            mask = np.ascontiguousarray(np.rot90(mask, k))
    # Mild color jitter.
    if rng.random() < 0.7:
        scale = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
        bias = rng.uniform(-12, 12, size=3).astype(np.float32)
        image = np.clip(image.astype(np.float32) * scale + bias, 0, 255).astype(np.uint8)
    return image, mask


class LesionDataset(Dataset):
    """Paired RGB / mask / diagnosis table used by both Task A and Task B."""

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int = 128,
        crop_mode: CropMode = "whole",
        augment: bool = False,
        pred_mask_dir: Path | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.crop_mode = crop_mode
        self.augment = augment
        self.pred_mask_dir = Path(pred_mask_dir) if pred_mask_dir else None
        self.rng = np.random.default_rng(seed)
        labels = pd.read_csv(self.root / "labels.csv")
        self.frame = labels[labels["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise FileNotFoundError(f"No rows for split={split!r} in {self.root / 'labels.csv'}")

    def __len__(self) -> int:
        return len(self.frame)

    def class_counts(self) -> torch.Tensor:
        counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for idx in self.frame["class_idx"].tolist():
            counts[int(idx)] += 1
        return torch.tensor(counts, dtype=torch.float32)

    def _mask_path(self, row: pd.Series) -> Path | None:
        if self.crop_mode == "crop_pred" and self.pred_mask_dir is not None:
            candidate = self.pred_mask_dir / f"{row['image_id']}.png"
            if candidate.exists():
                return candidate
        rel = row.get("mask_path")
        if isinstance(rel, str) and rel:
            path = self.root / rel
            if path.exists():
                return path
        fallback = self.root / "masks" / self.split / f"{row['image_id']}.png"
        return fallback if fallback.exists() else None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.frame.iloc[index]
        image = load_rgb(self.root / row["image_path"], size=None)
        mask_path = self._mask_path(row)
        mask = load_mask(mask_path, size=None) if mask_path is not None else np.ones(image.shape[:2], np.uint8)

        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = np.asarray(Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR))
            mask = np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                )
            )
            mask = (mask > 127).astype(np.uint8)

        if self.crop_mode in {"crop_gt", "crop_pred"}:
            image, mask = crop_to_mask(image, mask, size=self.image_size)

        if self.augment:
            image, mask = random_augment(image, mask, self.rng)

        return {
            "image": to_image_tensor(image),
            "mask": to_mask_tensor(mask),
            "label": torch.tensor(int(row["class_idx"]), dtype=torch.long),
            "image_id": str(row["image_id"]),
            "class_name": str(row["class_name"]),
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    class_counts: torch.Tensor | None = None,
    use_sampler: bool = False,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    sampler = None
    if use_sampler and shuffle and class_counts is not None:
        labels = [int(dataset.frame.iloc[i]["class_idx"]) for i in range(len(dataset))]  # type: ignore[attr-defined]
        weight_per_class = (1.0 / class_counts.clamp_min(1.0)).numpy()
        weights = [float(weight_per_class[y]) for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def inverse_freq_weights(counts: torch.Tensor) -> torch.Tensor:
    weights = counts.sum() / counts.clamp_min(1.0)
    return (weights / weights.mean()).to(dtype=torch.float32)
