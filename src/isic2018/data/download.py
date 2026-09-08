"""Opt-in downloader for official ISIC 2018 challenge archives.

This repository never vendors ISIC images. Running the downloader requires
an explicit ``--agree`` flag acknowledging the challenge terms.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from isic2018.constants import ISIC_ARCHIVES, ISIC_CHALLENGE_2018, ISIC_CHALLENGE_DATA
from isic2018.utils import LOGGER, Paths, ensure_dir

TERMS = f"""
ISIC 2018 data is NOT redistributed by this repository.

Official pages:
  {ISIC_CHALLENGE_DATA}
  {ISIC_CHALLENGE_2018}

Typical license for the 2018 challenge aggregates: CC-BY-NC.
You must cite Codella et al. (arXiv:1902.03368) and Tschandl et al. (HAM10000).

Re-run with --agree to download the public S3 archives into data/isic2018/
(git-ignored). This can be tens of gigabytes.
"""


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        LOGGER.info("Already present: %s", dest)
        return dest
    LOGGER.info("Downloading %s → %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def _extract(archive: Path, dest_dir: Path) -> Path:
    ensure_dir(dest_dir)
    marker = dest_dir / ".extracted"
    if marker.exists():
        return dest_dir
    LOGGER.info("Extracting %s", archive.name)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    marker.write_text(archive.name, encoding="utf-8")
    return dest_dir


def download_isic(
    dest: Path | None = None,
    tasks: tuple[str, ...] = ("1", "3"),
    agree: bool = False,
    keep_zip: bool = True,
) -> Path:
    if not agree:
        raise SystemExit(TERMS.strip())

    paths = Paths.from_root()
    dest = Path(dest) if dest is not None else paths.data_isic
    raw = ensure_dir(dest / "raw")
    for name, spec in ISIC_ARCHIVES.items():
        if spec["task"] not in tasks:
            continue
        filename = spec["url"].rsplit("/", 1)[-1]
        archive = _download(spec["url"], raw / filename)
        extract_dir = dest / f"task{spec['task']}" / spec["kind"] / spec["split"]
        _extract(archive, extract_dir)
        if not keep_zip:
            archive.unlink(missing_ok=True)
    LOGGER.info("ISIC 2018 archives extracted under %s", dest)
    LOGGER.info("Organize / train with configs/isic.yaml after inspecting folder names.")
    return dest


def which_isic_layout(root: Path) -> dict[str, Path]:
    """Best-effort discovery of extracted Task 1 / Task 3 folders."""
    found: dict[str, Path] = {}
    for path in Path(root).rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if "task1" in name and "groundtruth" in name:
            found.setdefault("task1_masks", path)
        elif "task1-2" in name and "input" in name:
            found.setdefault("task1_images", path)
        elif "task3" in name and "input" in name:
            found.setdefault("task3_images", path)
    return found
