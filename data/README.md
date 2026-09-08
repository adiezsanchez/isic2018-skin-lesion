# Data directory

This folder is a **layout**, not a dataset dump.

## What is (and is not) in git

| Path | In git? | Contents |
| --- | --- | --- |
| `data/demo/` | yes, after `pixi run generate-demo` | Synthetic dermoscopy-**like** PNG images + masks. **Not ISIC.** |
| `data/isic2018/` | **no** (gitignored) | Official ISIC 2018 archives you download yourself |

This repository **does not redistribute** ISIC challenge images, masks, or
CSV labels. That would violate the challenge terms (the 2018 aggregates are
typically CC-BY-NC and require citation).

## Synthetic demo (default)

```bash
pixi run generate-demo
```

Writes `data/demo/images/{train,val,test}/`, matching masks, and `labels.csv`.
Every demo image has **both** a lesion mask (Task A) and a 7-class diagnosis
(Task B) so the whole-image vs segment-crop comparison can run offline.

## Official ISIC 2018 (opt-in)

1. Read the challenge pages:
   - https://challenge.isic-archive.com/data/
   - https://challenge.isic-archive.com/landing/2018/
2. Agree to the terms, then:

```bash
pixi run download -- --agree
```

Archives are pulled from the public ISIC S3 bucket
(`https://isic-challenge-data.s3.amazonaws.com/2018/…`) and extracted under
`data/isic2018/` (gitignored). Expect **tens of gigabytes**.

Point training at that tree with `configs/isic.yaml`:

```bash
pixi run train-seg -- --config configs/isic.yaml --data data/isic2018
```

### Official task mapping

| This repo | ISIC 2018 | Files |
| --- | --- | --- |
| Task A | Task 1 — lesion boundary segmentation | `ISIC_*.jpg` + `ISIC_*_segmentation.png` |
| Task B | Task 3 — 7-class diagnosis | images + ground-truth CSV (`MEL,NV,BCC,AKIEC,BKL,DF,VASC`) |

Task 1 (~2.6k training images) and Task 3 / HAM10000 (~10k images) are
**different sets**. The synthetic demo gives every image both a mask and a
label so students can compare pipelines without joining IDs. On real data,
segment-then-crop classification uses a U-Net mask (or a Task 1 mask when the
same `ISIC_*` id exists).

## Citation (required if you use ISIC images)

- Codella et al., *Skin Lesion Analysis Toward Melanoma Detection 2018*, arXiv:1902.03368
- Tschandl, Rosendahl, Kittler, *The HAM10000 dataset*, Sci. Data 5:180161 (2018)
