"""PIL montages of synthetic demo images (not matplotlib plots)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isic2018.constants import CLASS_NAMES
from isic2018.data.datasets import LesionDataset, load_mask, load_rgb
from isic2018.utils import ensure_dir


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def write_class_gallery(data_root: Path, out_path: Path, split: str = "train", tile: int = 96) -> Path:
    ds = LesionDataset(data_root, split, image_size=tile)
    cols = len(CLASS_NAMES)
    rows = 2
    pad = 8
    header = 36
    canvas = Image.new("RGB", (cols * (tile + pad) + pad, rows * (tile + pad) + header + pad), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)
    font = _font(12)
    draw.text((pad, 8), "Synthetic dermoscopy-like demo (NOT ISIC data)", fill=(40, 30, 24), font=font)

    used = {name: 0 for name in CLASS_NAMES}
    for _, row in ds.frame.iterrows():
        name = str(row["class_name"])
        slot = used[name]
        if slot >= rows:
            continue
        used[name] = slot + 1
        img = load_rgb(ds.root / row["image_path"], size=tile)
        mask = load_mask(ds.root / row["mask_path"], size=tile)
        rgb = Image.fromarray(img)
        # Draw mask contour in lime.
        from skimage.segmentation import find_boundaries

        bounds = find_boundaries(mask.astype(bool), mode="outer")
        arr = np.array(rgb)
        arr[bounds] = (80, 220, 90)
        x = pad + CLASS_NAMES.index(name) * (tile + pad)
        y = header + pad + slot * (tile + pad)
        canvas.paste(Image.fromarray(arr), (x, y))
        if slot == 0:
            draw.text((x, y - 14), name, fill=(90, 30, 20), font=font)

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    canvas.save(out_path)
    return out_path


def write_overlay_strip(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray | None,
    out_path: Path,
    title: str = "",
) -> Path:
    tiles = [Image.fromarray(image)]
    gt = image.copy()
    gt[mask.astype(bool)] = (0.55 * gt[mask.astype(bool)] + np.array([40, 180, 70])).astype(np.uint8)
    tiles.append(Image.fromarray(gt))
    if pred is not None:
        pr = image.copy()
        pr[pred.astype(bool)] = (0.55 * pr[pred.astype(bool)] + np.array([40, 90, 200])).astype(np.uint8)
        tiles.append(Image.fromarray(pr))
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * len(tiles) + 12, h + 22), (250, 248, 244))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * w + 6, 18))
    ImageDraw.Draw(canvas).text((8, 2), title or "image | GT overlay | pred overlay", fill=(30, 30, 30), font=_font(11))
    ensure_dir(Path(out_path).parent)
    canvas.save(out_path)
    return out_path
