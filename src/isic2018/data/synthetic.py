"""Synthetic dermoscopy-like images for the offline pedagogical demo.

These images are **not** ISIC data. They are generated from noise, ellipses,
and class-specific palettes so the rest of the pipeline (U-Net, CNN, Grad-CAM,
Napari, Plotly) can run without downloading the challenge archives.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter
from tqdm import tqdm

from isic2018.constants import (
    CLASS_NAMES,
    CLASS_PALETTES,
    CLASS_TO_IDX,
    DEMO_CLASS_PRIOR,
    SKIN_TONES,
)
from isic2018.utils import LOGGER, Paths, ensure_dir, seed_everything


def _smooth_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    noise = rng.standard_normal(shape)
    return gaussian_filter(noise, sigma=sigma, mode="reflect")


def _radial_vignette(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    return np.clip(1.3 - 0.85 * r**2, 0.15, 1.0)


def _lesion_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
    irregularity: float,
) -> np.ndarray:
    """Signed-distance ellipse warped by smooth noise → irregular blob."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = h * rng.uniform(0.42, 0.58)
    cx = w * rng.uniform(0.42, 0.58)
    ry = h * rng.uniform(0.16, 0.32)
    rx = w * rng.uniform(0.16, 0.34)
    sdf = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 - 1.0
    warp = irregularity * _smooth_noise((h, w), rng, sigma=max(h / 18, 2.0))
    blob = sdf + warp < 0
    blob = binary_fill_holes(blob)
    # Keep the largest connected component.
    from skimage.measure import label

    labeled = label(blob)
    if labeled.max() == 0:
        return blob.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return labeled == keep


def _draw_hairs(image: np.ndarray, rng: np.random.Generator, n_hairs: int) -> np.ndarray:
    h, w, _ = image.shape
    canvas = image.copy()
    for _ in range(n_hairs):
        n_pts = rng.integers(4, 8)
        ys = rng.uniform(0, h, size=n_pts)
        xs = rng.uniform(0, w, size=n_pts)
        # Sort along the dominant axis so the stroke is a long curve, not a scribble.
        order = np.argsort(xs if rng.random() > 0.5 else ys)
        ys, xs = ys[order], xs[order]
        t = np.linspace(0, 1, n_pts)
        tt = np.linspace(0, 1, int(max(h, w) * 1.2))
        y_s = np.interp(tt, t, ys)
        x_s = np.interp(tt, t, xs)
        thickness = rng.uniform(0.6, 1.8)
        color = rng.uniform([15, 8, 5], [50, 30, 22])
        for y, x in zip(y_s, x_s):
            yy = int(np.clip(y, 0, h - 1))
            xx = int(np.clip(x, 0, w - 1))
            rad = max(int(thickness), 1)
            y0, y1 = max(0, yy - rad), min(h, yy + rad + 1)
            x0, x1 = max(0, xx - rad), min(w, xx + rad + 1)
            canvas[y0:y1, x0:x1] = 0.65 * canvas[y0:y1, x0:x1] + 0.35 * color
    return canvas


def _paint_lesion(
    image: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    palette = CLASS_PALETTES[class_name]
    h, w, _ = image.shape
    out = image.copy()
    noise = _smooth_noise((h, w), rng, sigma=max(h / 22, 1.5))
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-8)
    base = np.array(palette["lesion"], dtype=np.float32)
    accent = np.array(palette["accent"], dtype=np.float32)
    rim = np.array(palette["rim"], dtype=np.float32)

    from skimage.morphology import dilation, disk

    rim_band = dilation(mask, disk(max(h // 40, 2))) & ~mask
    color = base[None, None, :] * (1.0 - 0.45 * noise[..., None]) + accent[None, None, :] * (
        0.45 * noise[..., None]
    )
    # Melanoma: extra chromatic variegation.
    if class_name == "MEL":
        blotch = gaussian_filter(rng.random((h, w)), sigma=max(h / 16, 2))
        blotch = blotch > np.quantile(blotch, 0.72)
        color[blotch] = np.array([25, 12, 10], dtype=np.float32)
        color[~blotch] = 0.7 * color[~blotch] + 0.3 * rim
    elif class_name == "BCC":
        # Pearly pink + fake arborizing vessels (red filaments).
        veins = _smooth_noise((h, w), rng, sigma=1.2)
        vein_mask = (np.abs(veins) < 0.08) & mask
        color[vein_mask] = np.array([140, 25, 35], dtype=np.float32)
    elif class_name == "VASC":
        color = np.where(
            noise[..., None] > 0.55,
            np.array([90, 10, 30], dtype=np.float32),
            color,
        )
    elif class_name in {"BKL", "DF"}:
        # Stuck-on / central pale area.
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = np.argwhere(mask).mean(axis=0)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        core = dist < (0.28 * max(h, w) * rng.uniform(0.5, 1.0))
        color[core & mask] = accent
    elif class_name == "AKIEC":
        scale = rng.random((h, w)) > 0.82
        color[scale] = np.array([220, 200, 170], dtype=np.float32)

    out[mask] = color[mask]
    out[rim_band] = 0.55 * out[rim_band] + 0.45 * rim
    return np.clip(out, 0, 255)


def generate_one(
    class_name: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    h = w = int(size)
    skin = np.array(SKIN_TONES[int(rng.integers(0, len(SKIN_TONES)))], dtype=np.float32)
    image = np.ones((h, w, 3), dtype=np.float32) * skin
    texture = _smooth_noise((h, w), rng, sigma=max(h / 30, 1.5))
    image += 18.0 * texture[..., None]
    # Pore-like speckles.
    speck = rng.random((h, w)) > 0.992
    image[speck] *= 0.75

    irregularity = {
        "MEL": 0.55,
        "NV": 0.18,
        "BCC": 0.35,
        "AKIEC": 0.40,
        "BKL": 0.28,
        "DF": 0.22,
        "VASC": 0.20,
    }[class_name]
    mask = _lesion_mask(h, w, rng, irregularity=irregularity)
    image = _paint_lesion(image, mask, class_name, rng)
    image = _draw_hairs(image, rng, n_hairs=int(rng.integers(1, 5)))

    # Specular gel highlights.
    for _ in range(int(rng.integers(0, 3))):
        cy = int(rng.integers(h // 5, 4 * h // 5))
        cx = int(rng.integers(w // 5, 4 * w // 5))
        ry, rx = int(rng.integers(2, 6)), int(rng.integers(3, 8))
        yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
        blob = (yy / (ry + 1e-6)) ** 2 + (xx / (rx + 1e-6)) ** 2 <= 1
        y0, x0 = max(cy - ry, 0), max(cx - rx, 0)
        patch = image[y0 : cy + ry + 1, x0 : cx + rx + 1]
        by, bx = blob.shape
        patch = patch[:by, :bx]
        blob = blob[: patch.shape[0], : patch.shape[1]]
        patch[blob] = np.clip(patch[blob] * 0.4 + 200, 0, 255)
        image[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] = patch

    vignette = _radial_vignette(h, w)[..., None]
    image = image * vignette
    image = np.clip(image, 0, 255).astype(np.uint8)
    mask_u8 = (mask.astype(np.uint8)) * 255
    return image, mask_u8


def _allocate_counts(
    per_class: int,
    imbalanced: bool,
    rng: np.random.Generator,
) -> dict[str, int]:
    if not imbalanced:
        return {name: int(per_class) for name in CLASS_NAMES}
    # Keep NV as the majority class, rare classes as 1–2 samples when per_class is small.
    weights = np.array([DEMO_CLASS_PRIOR[n] for n in CLASS_NAMES], dtype=np.float64)
    total = per_class * len(CLASS_NAMES)
    raw = weights / weights.sum() * total
    counts = np.maximum(1, np.round(raw).astype(int))
    # Fix rounding so the total matches.
    while int(counts.sum()) < total:
        counts[int(rng.choice(np.arange(len(CLASS_NAMES)), p=weights / weights.sum()))] += 1
    while int(counts.sum()) > total:
        idx = int(np.argmax(counts))
        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break
    return {name: int(c) for name, c in zip(CLASS_NAMES, counts)}


def generate_demo_dataset(
    out_dir: Path | None = None,
    image_size: int = 128,
    per_class: int = 8,
    val_per_class: int = 2,
    test_per_class: int = 2,
    imbalanced: bool = True,
    seed: int = 7,
    force: bool = False,
) -> Path:
    """Write a small synthetic dataset under ``data/demo``."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    paths = Paths.from_root()
    out_dir = Path(out_dir) if out_dir is not None else paths.data_demo
    labels_path = out_dir / "labels.csv"
    meta_path = out_dir / "metadata.json"
    if labels_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("image_size", -1)) == int(image_size):
            LOGGER.info("Reusing existing demo dataset at %s", out_dir)
            return out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ensure_dir(out_dir)
    rows: list[dict] = []
    split_specs = {
        "train": _allocate_counts(per_class, imbalanced, rng),
        "val": {n: val_per_class for n in CLASS_NAMES},
        "test": {n: test_per_class for n in CLASS_NAMES},
    }

    for split, counts in split_specs.items():
        img_dir = ensure_dir(out_dir / "images" / split)
        mask_dir = ensure_dir(out_dir / "masks" / split)
        tasks = [(cls, i) for cls, n in counts.items() for i in range(n)]
        for class_name, i in tqdm(tasks, desc=f"synthetic {split}", leave=False):
            image, mask = generate_one(class_name, image_size, rng)
            image_id = f"{class_name}_{split}_{i:03d}"
            Image.fromarray(image).save(img_dir / f"{image_id}.png")
            Image.fromarray(mask).save(mask_dir / f"{image_id}.png")
            rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "class_name": class_name,
                    "class_idx": CLASS_TO_IDX[class_name],
                    "image_path": str(Path("images") / split / f"{image_id}.png"),
                    "mask_path": str(Path("masks") / split / f"{image_id}.png"),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "labels.csv", index=False)
    meta = {
        "synthetic": True,
        "not_isic": True,
        "image_size": image_size,
        "seed": seed,
        "imbalanced": imbalanced,
        "n_images": int(len(frame)),
        "counts": {f"{a}/{b}": int(v) for (a, b), v in frame.groupby(["split", "class_name"]).size().items()},
    }
    (out_dir / "README.txt").write_text(
        "SYNTHETIC dermoscopy-like images for the pedagogical demo.\n"
        "These are NOT ISIC 2018 images and must not be treated as clinical data.\n"
        "Download the official challenge archives with: pixi run download --agree\n"
        f"See {out_dir / 'metadata.json'} for generation settings.\n",
        encoding="utf-8",
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote %d synthetic images to %s", len(frame), out_dir)
    return out_dir
