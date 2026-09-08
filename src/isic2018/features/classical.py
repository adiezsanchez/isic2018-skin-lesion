"""Random Forest and XGBoost on ABCD / texture features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from isic2018.constants import CLASS_NAMES
from isic2018.data.datasets import LesionDataset, load_mask, load_rgb
from isic2018.features.handcrafted import FEATURE_NAMES, extract_features
from isic2018.utils import LOGGER, dump_json, ensure_dir


def _stack_split(dataset: LesionDataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, ids = [], [], []
    root = dataset.root
    for _, row in tqdm(dataset.frame.iterrows(), total=len(dataset.frame), desc="features", leave=False):
        image = load_rgb(root / row["image_path"], size=dataset.image_size)
        mask_path = root / row["mask_path"]
        mask = load_mask(mask_path, size=dataset.image_size) if mask_path.exists() else np.ones(image.shape[:2], np.uint8)
        xs.append(extract_features(image, mask))
        ys.append(int(row["class_idx"]))
        ids.append(str(row["image_id"]))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64), ids


def train_classical_models(
    data_root: Path,
    image_size: int,
    out_dir: Path,
    n_estimators: int = 200,
    max_depth: int = 6,
    seed: int = 7,
) -> dict:
    train_ds = LesionDataset(data_root, "train", image_size=image_size, crop_mode="whole")
    test_ds = LesionDataset(data_root, "test", image_size=image_size, crop_mode="whole")
    x_train, y_train, _ = _stack_split(train_ds)
    x_test, y_test, test_ids = _stack_split(test_ds)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    xgb.fit(x_train_s, y_train)

    results: dict = {"n_features": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES)}
    for name, model, xt in (
        ("random_forest", rf, x_test),
        ("xgboost", xgb, x_test_s),
    ):
        proba = model.predict_proba(xt)
        pred = proba.argmax(axis=1)
        try:
            auc = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            auc = float("nan")
        acc = float((pred == y_test).mean())
        results[name] = {
            "accuracy": acc,
            "macro_roc_auc": auc,
            "y_true": y_test.tolist(),
            "y_pred": pred.tolist(),
            "proba": proba.tolist(),
            "ids": test_ids,
        }
        LOGGER.info("%s  acc=%.3f  macro-AUC=%.3f", name, acc, auc)

    out_dir = ensure_dir(out_dir)
    dump_json(results, out_dir / "classical_metrics.json")
    np.savez(
        out_dir / "classical_arrays.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=np.array(FEATURE_NAMES),
    )
    return results
