# ISIC 2018 skin-lesion segmentation & classification

A **pedagogical** PyTorch project for the [ISIC 2018](https://challenge.isic-archive.com/landing/2018/)
challenge: **U-Net lesion segmentation** (Task A / official Task 1) and
**7-class dermoscopy diagnosis** (Task B / official Task 3) with transfer
learning, class imbalance, ROC-AUC, Grad-CAM, and a head-to-head of
**whole-image CNN vs segment-then-crop CNN vs handcrafted features → Random Forest / XGBoost**.

| Piece | Choice |
| --- | --- |
| Environment | **`pixi.toml` only** (no `environment.yml`, no `requirements.txt`) |
| Platforms | **`linux-64`** and **`win-64`** with **NVIDIA CUDA** declared for the solver |
| Deep learning | **PyTorch + torchvision** (`pytorch-gpu`, CUDA 12.6) — **not TensorFlow** |
| Interactive viz | **Napari** (RGB + GT mask + U-Net mask + Grad-CAM) |
| Plots | **Plotly only** (ROC, PR, confusion, metrics, training curves) |

This repository **does not ship ISIC images**. A small **synthetic
dermoscopy-like** set plus Plotly figures under `results/figures/` let you run
the full pipeline offline. Official archives are downloaded only after you
pass `--agree`.

---

## What you will learn

1. **Task A — U-Net segmentation.** Encoder–decoder with skip connections,
   BCE+Dice loss, Dice / IoU reporting.
2. **Task B — transfer classification.** ResNet-18, optional ImageNet init,
   inverse-frequency loss weights (or a `WeightedRandomSampler`, or focal loss).
3. **Imbalance.** HAM10000 / ISIC Task 3 is dominated by nevus (`NV`). The
   synthetic generator can mimic that prior so metrics are not “accuracy of
   predicting NV”.
4. **ROC-AUC and PR.** One-vs-rest curves in Plotly, plus macro ROC-AUC.
5. **Grad-CAM.** A ~40-line implementation (hooks on the last conv map) so you
   can read the math and the code together.
6. **Whole image vs lesion crop.** Train the same CNN on the full dermoscopy
   field vs a box cropped from the GT mask vs a box cropped from the **U-Net
   prediction** (the realistic two-stage pipeline).
7. **Classical vs deep.** ABCD-inspired shape / colour / GLCM / LBP features
   fed to Random Forest and XGBoost vs the CNN.

Synthetic images are **not clinical data**. They exist so the tooling
(Pixi, CUDA PyTorch, Napari, Plotly) is exercisable without a 10+ GB
download.

---

## Quick start

```bash
# 1. Install Pixi (once per machine)
# Linux:
curl -fsSL https://pixi.sh/install.sh | bash
# Windows (PowerShell):
# iwr -useb https://pixi.sh/install.ps1 | iex

# 2. From this repo:
pixi install          # CUDA PyTorch + Napari + Plotly (first run is large)
pixi run info         # Torch version, CUDA availability, GPU name
pixi run test         # CPU unit tests
pixi run demo         # generate data → train → Plotly figures
pixi run viz          # Napari (needs a display)
```

The demo writes HTML/PNG figures to [`results/figures/`](results/figures/).
Open any `*.html` in a browser; they use Plotly.js from CDN.

Tiny laptop / CI smoke run:

```bash
pixi run demo-smoke
```

---

## Install: Linux, Windows, CUDA, Napari

### Pixi

Pixi is the only environment tool this project uses. It solves **conda-forge**
packages into a lockfile (`pixi.lock`) for both platforms.

```bash
pixi --version
pixi install
```

`pixi.toml` declares CUDA on linux-64 and win-64 so the solver picks **GPU**
PyTorch (`pytorch-gpu`) when an NVIDIA driver is present. CPU platforms are
listed second so `pixi install` still works on machines without a GPU
(the conda subdirs remain `linux-64` and `win-64` only):

```toml
platforms = [
  { name = "linux-64-cuda", platform = "linux-64", cuda = "12.0" },
  { name = "win-64-cuda", platform = "win-64", cuda = "12.0" },
  { name = "linux-64-cpu", platform = "linux-64" },
  { name = "win-64-cpu", platform = "win-64" },
]
```

`cuda-version = "12.6.*"` (CUDA target) pins the CUDA *toolkit* that
`libtorch` is built against. You still need a recent **NVIDIA driver** for
`torch.cuda.is_available()` to be true. On a GPU machine Pixi selects the
CUDA platform automatically; otherwise it falls back to `pytorch-cpu`.

Force a specific solve (useful in CI):

```bash
pixi run --platform linux-64-cuda info    # requires CONDA_OVERRIDE_CUDA=12.0 if no driver
pixi run --platform linux-64-cpu info
CONDA_OVERRIDE_CUDA=12.0 pixi install --platform linux-64-cuda
```

Check what Pixi thinks your machine provides:

```bash
pixi info
# look for a `__cuda=12.x` virtual package
nvidia-smi          # driver / GPU
pixi run info       # torch.cuda.is_available(), device name
```

### NVIDIA CUDA (Linux)

1. Install a current proprietary NVIDIA driver for your GPU
   (Ubuntu: `ubuntu-drivers`, Fedora: RPMFusion, etc.).
2. You do **not** need a system-wide CUDA toolkit. Conda-forge ships
   `cuda-version` / `cudatoolkit` next to `pytorch-gpu`.
3. Driver too old for CUDA 12? Either upgrade the driver or lower
   `cuda-version` in `pixi.toml` (for example `12.1.*`) and run
   `pixi lock && pixi install`.

### NVIDIA CUDA (Windows)

1. Install the [NVIDIA Game Ready / Studio driver](https://www.nvidia.com/Download/index.aspx).
2. Install Pixi in **native Windows** (PowerShell). WSL2 also works if the
   Windows NVIDIA driver is installed and CUDA in WSL is enabled.
3. `pixi install` from the repo; then `pixi run info`.

Do **not** mix the legacy `pytorch` conda channel with `conda-forge` in this
manifest. The Pixi docs recommend conda-forge `pytorch-gpu` plus a platform
`cuda = "12.0"` declaration — that is what this repo does.

### Napari (interactive viewer)

Napari needs a **Qt display**:

| Environment | What to do |
| --- | --- |
| Linux desktop | `pixi run viz` — uses the bundled `pyqt` 5 from conda-forge |
| Linux SSH / CI | No window. Use Plotly HTML, or `export QT_QPA_PLATFORM=offscreen` only for import smoke tests |
| Windows | Run from a normal desktop session, not a headless service account |
| WSL2 | Use [WSLg](https://github.com/microsoft/wslg) so Qt can open a window |

Headless machines can still train and write Plotly figures; Napari is the
interactive inspection step.

If Napari fails with an OpenGL / `libGL` error on Linux, install your
distribution’s OpenGL stack (e.g. `libgl1` / `mesa`) — those are *system*
libraries, not Pixi packages.

---

## Usage

All workflows are Pixi tasks wrapping `python -m isic2018.cli`. Extra CLI
flags go after `--`.

| Task | Command |
| --- | --- |
| Hardware report | `pixi run info` |
| Synthetic data | `pixi run generate-demo` |
| Official download | `pixi run download -- --agree` |
| Train U-Net | `pixi run train-seg` |
| Train CNN | `pixi run train-clf -- --crop-mode whole` |
| Train RF / XGBoost | `pixi run train-classical` |
| Evaluate + figures | `pixi run eval-seg` / `pixi run eval-clf` |
| Compare methods | `pixi run compare` |
| Napari | `pixi run viz` |
| Full demo | `pixi run demo` |
| Tests | `pixi run test` |

### Demo data vs real ISIC

```bash
# Offline pedagogical path (default)
pixi run demo -- --config configs/demo.yaml

# After you have accepted ISIC terms and downloaded archives:
pixi run train-seg -- --config configs/isic.yaml --data data/isic2018
pixi run train-clf -- --config configs/isic.yaml --data data/isic2018 --crop-mode whole
```

See [`data/README.md`](data/README.md) for S3 URLs, folder layout, and
citations. **Never commit `data/isic2018/`.**

---

## Methods (short)

### Task A — U-Net

A 4-level U-Net (`src/isic2018/models/unet.py`) predicts a binary lesion mask.
The loss is a mix of per-pixel BCE and soft Dice. Report **Dice** and **IoU**
on the test split (ISIC Task 1 also used a thresholded Jaccard index).

### Task B — CNN

ResNet-18 with a 7-way head (`MEL, NV, BCC, AKIEC, BKL, DF, VASC`). Demo
training starts from random weights so it stays offline; `configs/isic.yaml`
turns on ImageNet initialization and a short backbone freeze.

Imbalance modes (`classification.imbalance` in YAML):

- `weights` — inverse-frequency `CrossEntropyLoss` (default)
- `sampler` — `WeightedRandomSampler`
- `focal` — multi-class focal loss
- `none` — vanilla CE

### Whole image vs segment-crop

| `crop_mode` | Input to the CNN |
| --- | --- |
| `whole` | Full dermoscopy field (including skin, hair, gel) |
| `crop_gt` | Axis-aligned box around the **ground-truth** mask (oracle crop) |
| `crop_pred` | Same box around the **U-Net** mask (two-stage system) |

Cropping removes a lot of healthy skin and can help small lesions; it also
inherits segmentation errors. The comparison figure
`results/figures/method_comparison.html` puts these next to RF / XGBoost.

### Handcrafted features

`src/isic2018/features/handcrafted.py` implements an ABCD-flavoured vector:
area / perimeter / circularity / asymmetry, colour stats in RGB / HSV / Lab,
GLCM contrast–homogeneity, LBP energy–entropy. Random Forest uses
`class_weight="balanced"`; XGBoost sees standardized features.

### Grad-CAM

`src/isic2018/models/gradcam.py` stores last-conv activations \(A^k\),
backprops the class score, GAP-pools the gradients into \(\alpha_k\), and
forms \(\mathrm{ReLU}(\sum_k \alpha_k A^k)\). Overlays are blended in NumPy
(no matplotlib).

---

## Project layout

```
pixi.toml                 # sole environment spec (linux-64 + win-64, CUDA)
configs/demo.yaml         # fast synthetic run
configs/isic.yaml         # starting point for real archives
src/isic2018/
  cli.py                  # pixi run *  →  python -m isic2018.cli
  data/                   # synthetic generator, datasets, opt-in downloader
  models/                 # U-Net, ResNet-18, losses, Grad-CAM
  features/               # ABCD + RF/XGBoost
  train/                  # Task A / Task B loops (AMP on CUDA)
  eval/                   # metrics + Plotly figures
  viz/                    # Napari + PIL montages
data/demo/                # synthetic images (not ISIC)
data/isic2018/            # gitignored official download
results/figures/          # Plotly HTML/PNG
tests/                    # CPU tests, no ISIC download
```

---

## Figures

After `pixi run demo` you should see (among others):

- `demo_gallery.png` — synthetic examples with mask contours (PIL montage)
- `task_a_overlay_example.png` — original | GT overlay | U-Net overlay
- `task_a_metrics.html` / `.png` — Dice, IoU, pixel accuracy
- `task_a_training.html` — U-Net loss / Dice curves
- `clf_whole_roc.html`, `clf_whole_pr.html`, `clf_whole_confusion.html`
- `clf_*_gradcam.html` — Plotly image grid of Grad-CAM overlays
- `random_forest_confusion.html`, `xgboost_confusion.html`
- `method_comparison.html` — the headline bar chart

The numbers that ship with this repo were produced by `pixi run demo` on
**synthetic** 128×128 images (84 samples, 6 epochs, randomly initialized
ResNet-18). They illustrate the *pipeline*, not ISIC leaderboard
performance. On this tiny set, ABCD features + Random Forest often beat the
CNN — a useful reminder that data size and pretraining matter. Use
`configs/isic.yaml` and ImageNet weights on the real archives.

Matplotlib may appear as a *transitive* dependency of Napari / scikit-image.
This project never calls `matplotlib.pyplot` for ROC, PR, confusion, or
metric plots.

---

## Citation

If you use **ISIC 2018 / HAM10000** images:

> Codella N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018:
> A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)*.
> arXiv:1902.03368.

> Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin
> lesions.* Scientific Data 5, 180161 (2018).

U-Net: Ronneberger, Fischer, Brox, MICCAI 2015. Grad-CAM: Selvaraju et al.,
ICCV 2017.

---

## License

MIT for **code** in this repository (see [`LICENSE`](LICENSE)). ISIC images
remain under their own terms; this project does not grant rights to them.
