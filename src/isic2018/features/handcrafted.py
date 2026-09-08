"""ABCD-inspired handcrafted features from a dermoscopy image + lesion mask.

These are classical computer-vision descriptors used before CNNs became the
default: Asymmetry, Border, Color, Diameter / texture (the ABCD rule of
dermoscopy, plus GLCM / LBP texture).
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image


FEATURE_NAMES: tuple[str, ...] = (
    "area",
    "perimeter",
    "equivalent_diameter",
    "major_axis",
    "minor_axis",
    "eccentricity",
    "solidity",
    "extent",
    "circularity",
    "asymmetry_h",
    "asymmetry_v",
    "border_irregularity",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "std_h",
    "std_s",
    "std_v",
    "mean_l",
    "mean_a",
    "mean_b_lab",
    "color_variegation",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "lbp_energy",
    "lbp_entropy",
)


def _largest_region(mask: np.ndarray):
    labeled = label(mask.astype(bool))
    if labeled.max() == 0:
        return None
    props = regionprops(labeled)
    return max(props, key=lambda p: p.area)


def _asymmetry(mask: np.ndarray) -> tuple[float, float]:
    """XOR of the mask with its horizontal / vertical flips, normalized."""
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0, 0.0
    h_flip = np.fliplr(m)
    v_flip = np.flipud(m)
    return float(np.logical_xor(m, h_flip).mean()), float(np.logical_xor(m, v_flip).mean())


def _border_irregularity(mask: np.ndarray, prop) -> float:
    hull = convex_hull_image(mask.astype(bool))
    hull_perim = regionprops(hull.astype(np.uint8))[0].perimeter if hull.any() else 1.0
    if hull_perim <= 0:
        return 0.0
    return float(prop.perimeter / hull_perim)


def extract_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a 1-D feature vector aligned with ``FEATURE_NAMES``.

    ``image`` is uint8 RGB HWC; ``mask`` is {0,1} HW. If the mask is empty,
    features are computed on the whole image so classical models still run.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if mask.sum() < 8:
        mask = np.ones(image.shape[:2], dtype=np.uint8)
    prop = _largest_region(mask)
    if prop is None:
        zeros = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        return zeros

    area = float(prop.area)
    perim = float(max(prop.perimeter, 1.0))
    circularity = float(4.0 * np.pi * area / (perim**2 + 1e-8))
    asym_h, asym_v = _asymmetry(mask)
    border_irr = _border_irregularity(mask, prop)

    lesion = image[mask.astype(bool)]
    hsv = rgb2hsv(image)
    lab = rgb2lab(image)
    hsv_px = hsv[mask.astype(bool)]
    lab_px = lab[mask.astype(bool)]

    mean_rgb = lesion.mean(axis=0)
    std_rgb = lesion.std(axis=0)
    mean_hsv = hsv_px.mean(axis=0)
    std_hsv = hsv_px.std(axis=0)
    mean_lab = lab_px.mean(axis=0)
    # Number of coarse color bins occupied inside the lesion (variegation).
    quant = (lesion.astype(np.int32) // 32)
    color_var = float(len({tuple(row) for row in quant})) / 512.0

    gray = np.mean(image, axis=2).astype(np.uint8)
    gray_lesion = gray.copy()
    gray_lesion[~mask.astype(bool)] = 0
    # Quantize for a small GLCM.
    gray_q = (gray_lesion // 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=[2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    homo = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    corr = float(graycoprops(glcm, "correlation").mean())

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp[mask.astype(bool)], bins=10, range=(0, 10), density=True)
    lbp_energy = float((lbp_hist**2).sum())
    lbp_entropy = float(-(lbp_hist * np.log(lbp_hist + 1e-8)).sum())

    values = [
        area / mask.size,
        perim / (mask.shape[0] + mask.shape[1]),
        float(prop.equivalent_diameter_area) / max(mask.shape),
        float(prop.axis_major_length) / max(mask.shape),
        float(prop.axis_minor_length) / max(mask.shape),
        float(prop.eccentricity),
        float(prop.solidity),
        float(prop.extent),
        circularity,
        asym_h,
        asym_v,
        border_irr,
        *(mean_rgb / 255.0),
        *(std_rgb / 255.0),
        *mean_hsv,
        *std_hsv,
        mean_lab[0] / 100.0,
        mean_lab[1] / 128.0,
        mean_lab[2] / 128.0,
        color_var,
        contrast / 50.0,
        homo,
        energy,
        corr,
        lbp_energy,
        lbp_entropy,
    ]
    return np.asarray(values, dtype=np.float32)
