"""Napari viewer for RGB dermoscopy, GT / predicted masks, and Grad-CAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from isic2018.data.datasets import LesionDataset, load_mask, load_rgb
from isic2018.utils import LOGGER


def _stack_split(dataset: LesionDataset, pred_mask_dir: Path | None, max_n: int) -> dict[str, np.ndarray]:
    images, masks, preds, labels = [], [], [], []
    for i, row in dataset.frame.iterrows():
        if len(images) >= max_n:
            break
        image = load_rgb(dataset.root / row["image_path"], size=dataset.image_size)
        mask_path = dataset.root / row["mask_path"]
        mask = load_mask(mask_path, size=dataset.image_size) if mask_path.exists() else np.zeros(image.shape[:2], np.uint8)
        pred = np.zeros_like(mask)
        if pred_mask_dir is not None:
            p = Path(pred_mask_dir) / f"{row['image_id']}.png"
            if p.exists():
                pred = load_mask(p, size=dataset.image_size)
        images.append(image)
        masks.append(mask)
        preds.append(pred)
        labels.append(f"{row['image_id']} | {row['class_name']}")
    return {
        "images": np.stack(images, axis=0),
        "masks": np.stack(masks, axis=0),
        "preds": np.stack(preds, axis=0),
        "labels": labels,
    }


def launch_napari(
    data_root: Path,
    split: str = "test",
    image_size: int = 128,
    pred_mask_dir: Path | None = None,
    gradcam_dir: Path | None = None,
    max_n: int = 16,
) -> None:
    """Open an interactive Napari viewer (requires a Qt display)."""
    try:
        import napari
    except Exception as exc:  # pragma: no cover - import environment
        raise SystemExit(
            "Napari failed to import. On a headless machine this is expected.\n"
            "Linux: install system OpenGL and run inside a desktop session, or\n"
            "  export QT_QPA_PLATFORM=offscreen   # for smoke tests only\n"
            "Windows: use native Windows (not a headless SSH session).\n"
            f"Original error: {exc}"
        ) from exc

    ds = LesionDataset(data_root, split, image_size=image_size)
    payload = _stack_split(ds, pred_mask_dir, max_n=max_n)
    LOGGER.info("Launching Napari with %d slices from %s/%s", payload["images"].shape[0], data_root, split)

    viewer = napari.Viewer(title="ISIC 2018 pedagogical viewer (synthetic or local data)")
    viewer.add_image(
        payload["images"],
        name="dermoscopy (RGB)",
        rgb=True,
        metadata={"labels": payload["labels"]},
    )
    viewer.add_labels(payload["masks"].astype(np.int32), name="ground-truth lesion", opacity=0.4)
    viewer.add_labels(payload["preds"].astype(np.int32), name="U-Net prediction", opacity=0.4)

    if gradcam_dir is not None and Path(gradcam_dir).exists():
        cams = []
        for row in ds.frame.itertuples():
            p = Path(gradcam_dir) / f"{row.image_id}.png"
            if p.exists():
                cams.append(np.asarray(Image.open(p).convert("L")))
            if len(cams) >= payload["images"].shape[0]:
                break
        if cams:
            viewer.add_image(
                np.stack(cams, axis=0).astype(np.float32) / 255.0,
                name="Grad-CAM",
                colormap="magma",
                opacity=0.5,
                blending="additive",
            )

    # Status bar: class names for the current slice.
    def _on_step(_event=None) -> None:
        idx = int(viewer.dims.current_step[0]) if viewer.dims.ndim >= 3 else 0
        idx = min(idx, len(payload["labels"]) - 1)
        viewer.status = payload["labels"][idx]

    viewer.dims.events.current_step.connect(_on_step)
    _on_step()
    napari.run()
