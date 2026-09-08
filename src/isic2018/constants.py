"""ISIC 2018 pedagogical constants.

Task A in this repo maps to official ISIC 2018 Task 1 (lesion boundary
segmentation). Task B maps to official Task 3 (7-class disease diagnosis).
"""

from __future__ import annotations

# Official ISIC 2018 Task 3 class order (CSV column order).
CLASS_NAMES: tuple[str, ...] = (
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

CLASS_LABELS: dict[str, str] = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / Bowen's (intraepithelial carcinoma)",
    "BKL": "Benign keratosis (solar lentigo / SK / LPLK)",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
}

# HAM10000 / ISIC 2018 Task 3 is heavily imbalanced: NV dominates.
# These relative frequencies are used only to synthesize a realistic demo
# prior; they are not the official counts.
DEMO_CLASS_PRIOR: dict[str, float] = {
    "MEL": 0.11,
    "NV": 0.67,
    "BCC": 0.05,
    "AKIEC": 0.03,
    "BKL": 0.11,
    "DF": 0.01,
    "VASC": 0.02,
}

# Approximate dermoscopic palettes (RGB, 0-255) used by the synthetic generator.
CLASS_PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "MEL": {
        "lesion": (42, 22, 16),
        "accent": (92, 38, 22),
        "rim": (120, 70, 40),
    },
    "NV": {
        "lesion": (110, 70, 40),
        "accent": (150, 100, 60),
        "rim": (170, 130, 90),
    },
    "BCC": {
        "lesion": (198, 120, 110),
        "accent": (160, 50, 50),
        "rim": (220, 170, 150),
    },
    "AKIEC": {
        "lesion": (180, 90, 70),
        "accent": (210, 160, 120),
        "rim": (200, 140, 110),
    },
    "BKL": {
        "lesion": (130, 85, 45),
        "accent": (240, 230, 210),
        "rim": (160, 120, 80),
    },
    "DF": {
        "lesion": (120, 75, 50),
        "accent": (230, 220, 200),
        "rim": (150, 110, 80),
    },
    "VASC": {
        "lesion": (150, 30, 50),
        "accent": (90, 15, 35),
        "rim": (200, 90, 110),
    },
}

SKIN_TONES: tuple[tuple[int, int, int], ...] = (
    (232, 190, 160),
    (214, 168, 138),
    (196, 140, 108),
    (168, 118, 88),
    (140, 92, 68),
)

ISIC_CHALLENGE_DATA = "https://challenge.isic-archive.com/data/"
ISIC_CHALLENGE_2018 = "https://challenge.isic-archive.com/landing/2018/"
ISIC_S3_BASE = "https://isic-challenge-data.s3.amazonaws.com/2018"

# Official public archives. This project never vendors the bytes.
ISIC_ARCHIVES: dict[str, dict[str, str]] = {
    "task1_train_images": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task1-2_Training_Input.zip",
        "task": "1",
        "kind": "images",
        "split": "train",
    },
    "task1_train_masks": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task1_Training_GroundTruth.zip",
        "task": "1",
        "kind": "masks",
        "split": "train",
    },
    "task1_val_images": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task1-2_Validation_Input.zip",
        "task": "1",
        "kind": "images",
        "split": "val",
    },
    "task1_val_masks": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task1_Validation_GroundTruth.zip",
        "task": "1",
        "kind": "masks",
        "split": "val",
    },
    "task3_train_images": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task3_Training_Input.zip",
        "task": "3",
        "kind": "images",
        "split": "train",
    },
    "task3_train_labels": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task3_Training_GroundTruth.zip",
        "task": "3",
        "kind": "labels",
        "split": "train",
    },
    "task3_val_images": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task3_Validation_Input.zip",
        "task": "3",
        "kind": "images",
        "split": "val",
    },
    "task3_val_labels": {
        "url": f"{ISIC_S3_BASE}/ISIC2018_Task3_Validation_GroundTruth.zip",
        "task": "3",
        "kind": "labels",
        "split": "val",
    },
}

CITATIONS: tuple[str, ...] = (
    "Codella N. et al. Skin Lesion Analysis Toward Melanoma Detection 2018: "
    "A Challenge Hosted by ISIC. arXiv:1902.03368.",
    "Tschandl P., Rosendahl C., Kittler H. The HAM10000 dataset, a large "
    "collection of multi-source dermatoscopic images of common pigmented skin "
    "lesions. Sci. Data 5, 180161 (2018).",
)
