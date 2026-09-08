"""Shared helpers: paths, seeds, device selection, config I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger("isic2018")


def repo_root() -> Path:
    """Return the repository root (directory that contains pixi.toml)."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pixi.toml").exists():
            return candidate
    return Path.cwd()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Kaleido/Chrome chatter drowns training logs during PNG export.
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("choreographer").setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_device(prefer_cuda: bool = True):
    """Select CUDA when a GPU is visible, otherwise CPU.

    CUDA-enabled PyTorch wheels still import and train on CPU if no driver
    is present; this helper never crashes in that situation.
    """
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_report() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": str(get_device()),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_capability"] = ".".join(
            str(v) for v in torch.cuda.get_device_capability(idx)
        )
    return info


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return data


def dump_json(payload: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Paths:
    root: Path
    data_demo: Path
    data_isic: Path
    figures: Path
    runs: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "Paths":
        root = Path(root) if root is not None else repo_root()
        return cls(
            root=root,
            data_demo=root / "data" / "demo",
            data_isic=root / "data" / "isic2018",
            figures=root / "results" / "figures",
            runs=root / "results" / "runs",
        )
