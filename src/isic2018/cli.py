"""Command-line entry point: ``python -m isic2018.cli <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isic2018 import __version__
from isic2018.constants import CITATIONS, CLASS_NAMES, ISIC_CHALLENGE_2018, ISIC_CHALLENGE_DATA
from isic2018.utils import (
    Paths,
    cuda_report,
    dump_json,
    load_yaml,
    repo_root,
    seed_everything,
    setup_logging,
)


def _paths() -> Paths:
    return Paths.from_root()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML config (optional)")
    parser.add_argument("--data", type=Path, default=None, help="Dataset root (default: data/demo)")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")


def _merge(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.config)
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("image_size", 128)
    cfg.setdefault("seed", 7)
    if args.epochs is not None:
        cfg.setdefault("segmentation", {})
        cfg.setdefault("classification", {})
        cfg["segmentation"]["epochs"] = args.epochs
        cfg["classification"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("segmentation", {})
        cfg.setdefault("classification", {})
        cfg["segmentation"]["batch_size"] = args.batch_size
        cfg["classification"]["batch_size"] = args.batch_size
    return cfg


def _data_root(args: argparse.Namespace, paths: Paths) -> Path:
    return Path(args.data) if getattr(args, "data", None) else paths.data_demo


def cmd_info(_args: argparse.Namespace) -> None:
    report = cuda_report()
    print(f"isic2018-skin-lesion {__version__}")
    print(f"repo: {repo_root()}")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("classes:", ", ".join(CLASS_NAMES))
    print("challenge data:", ISIC_CHALLENGE_DATA)
    print("challenge 2018:", ISIC_CHALLENGE_2018)
    print("This repo does not ship ISIC images.")
    for cite in CITATIONS:
        print("cite:", cite)


def cmd_generate_demo(args: argparse.Namespace) -> None:
    from isic2018.data.synthetic import generate_demo_dataset

    cfg = _merge(args)
    demo = cfg.get("demo", {})
    per_class = getattr(args, "per_class", None) or demo.get("per_class", 8)
    out = generate_demo_dataset(
        out_dir=_data_root(args, _paths()),
        image_size=cfg["image_size"],
        per_class=int(per_class),
        val_per_class=int(demo.get("val_per_class", 2)),
        test_per_class=int(demo.get("test_per_class", 2)),
        imbalanced=bool(demo.get("imbalanced", True)),
        seed=cfg["seed"],
    )
    print(f"demo dataset: {out}")


def cmd_download(args: argparse.Namespace) -> None:
    from isic2018.data.download import TERMS, download_isic

    if not args.agree:
        raise SystemExit(TERMS.strip())
    dest = download_isic(
        dest=args.dest,
        tasks=tuple(args.tasks.split(",")),
        agree=True,
        keep_zip=not args.delete_zip,
    )
    print(f"extracted under {dest}")


def cmd_train_seg(args: argparse.Namespace) -> None:
    from isic2018.train.segment import train_unet

    cfg = _merge(args)
    seed_everything(cfg["seed"])
    seg = cfg.get("segmentation", {})
    paths = _paths()
    train_unet(
        data_root=_data_root(args, paths),
        image_size=int(cfg["image_size"]),
        epochs=int(seg.get("epochs", 6)),
        batch_size=int(seg.get("batch_size", 8)),
        lr=float(seg.get("lr", 1e-3)),
        base_channels=int(seg.get("base_channels", 16)),
        dice_weight=float(seg.get("dice_weight", 0.5)),
        out_dir=paths.runs / "task_a_unet",
        seed=cfg["seed"],
    )


def cmd_eval_seg(args: argparse.Namespace) -> None:
    from isic2018.eval.compare import evaluate_segmentation

    cfg = _merge(args)
    paths = _paths()
    ckpt = args.checkpoint or (paths.runs / "task_a_unet" / "unet_best.pt")
    evaluate_segmentation(
        data_root=_data_root(args, paths),
        ckpt=Path(ckpt),
        figures=paths.figures,
        image_size=int(cfg["image_size"]),
        split=args.split,
    )


def cmd_train_clf(args: argparse.Namespace) -> None:
    from isic2018.train.classify import train_classifier

    cfg = _merge(args)
    seed_everything(cfg["seed"])
    clf = cfg.get("classification", {})
    paths = _paths()
    crop_mode = args.crop_mode or clf.get("crop_mode", "whole")
    pred_dir = paths.runs / "task_a_unet" / "pred_masks" if crop_mode == "crop_pred" else None
    train_classifier(
        data_root=_data_root(args, paths),
        image_size=int(cfg["image_size"]),
        epochs=int(clf.get("epochs", 6)),
        batch_size=int(clf.get("batch_size", 8)),
        lr=float(clf.get("lr", 1e-3)),
        backbone=str(clf.get("backbone", "resnet18")),
        pretrained=bool(clf.get("pretrained", False)),
        crop_mode=crop_mode,
        imbalance=str(clf.get("imbalance", "weights")),
        freeze_epochs=int(clf.get("freeze_epochs", 0)),
        out_dir=paths.runs / f"task_b_{crop_mode}",
        pred_mask_dir=pred_dir,
        seed=cfg["seed"],
    )


def cmd_eval_clf(args: argparse.Namespace) -> None:
    from isic2018.eval.compare import evaluate_classifier

    cfg = _merge(args)
    paths = _paths()
    crop_mode = args.crop_mode or cfg.get("classification", {}).get("crop_mode", "whole")
    ckpt = args.checkpoint or (paths.runs / f"task_b_{crop_mode}" / "clf_best.pt")
    pred_dir = paths.runs / "task_a_unet" / "pred_masks" if crop_mode == "crop_pred" else None
    evaluate_classifier(
        data_root=_data_root(args, paths),
        ckpt=Path(ckpt),
        figures=paths.figures,
        image_size=int(cfg["image_size"]),
        crop_mode=crop_mode,
        split=args.split,
        pred_mask_dir=pred_dir,
        tag=f"clf_{crop_mode}",
    )


def cmd_train_classical(args: argparse.Namespace) -> None:
    from isic2018.features.classical import train_classical_models

    cfg = _merge(args)
    paths = _paths()
    classical = cfg.get("classical", {})
    train_classical_models(
        data_root=_data_root(args, paths),
        image_size=int(cfg["image_size"]),
        out_dir=paths.runs / "classical",
        n_estimators=int(classical.get("n_estimators", 200)),
        max_depth=int(classical.get("max_depth", 6)),
        seed=cfg["seed"],
    )


def cmd_compare(args: argparse.Namespace) -> None:
    from isic2018.eval.compare import compare_methods, evaluate_classifier, evaluate_segmentation
    from isic2018.train.segment import predict_masks

    cfg = _merge(args)
    paths = _paths()
    data = _data_root(args, paths)
    size = int(cfg["image_size"])

    unet_ckpt = paths.runs / "task_a_unet" / "unet_best.pt"
    if unet_ckpt.exists():
        evaluate_segmentation(data, unet_ckpt, paths.figures, size, split="test")
        pred_dir = predict_masks(
            data, unet_ckpt, "test", paths.runs / "task_a_unet" / "pred_masks_test", size
        )
        predict_masks(data, unet_ckpt, "train", paths.runs / "task_a_unet" / "pred_masks", size)
        predict_masks(data, unet_ckpt, "val", paths.runs / "task_a_unet" / "pred_masks", size)
    else:
        pred_dir = None

    cnn_results = []
    for mode in ("whole", "crop_gt", "crop_pred"):
        ckpt = paths.runs / f"task_b_{mode}" / "clf_best.pt"
        if not ckpt.exists():
            continue
        tag = f"clf_{mode}"
        metrics = evaluate_classifier(
            data,
            ckpt,
            paths.figures,
            size,
            crop_mode="crop_gt" if mode == "crop_pred" and pred_dir is None else mode,
            pred_mask_dir=paths.runs / "task_a_unet" / "pred_masks" if mode == "crop_pred" else None,
            tag=tag,
        )
        metrics["name"] = {
            "whole": "CNN whole-image",
            "crop_gt": "CNN crop (GT mask)",
            "crop_pred": "CNN crop (U-Net mask)",
        }[mode]
        cnn_results.append(metrics)

    classical_path = paths.runs / "classical" / "classical_metrics.json"
    classical = json.loads(classical_path.read_text(encoding="utf-8")) if classical_path.exists() else {}
    compare_methods(cnn_results, classical, paths.figures)


def cmd_viz(args: argparse.Namespace) -> None:
    from isic2018.viz.napari_viewer import launch_napari

    cfg = _merge(args)
    paths = _paths()
    launch_napari(
        data_root=_data_root(args, paths),
        split=args.split,
        image_size=int(cfg["image_size"]),
        pred_mask_dir=paths.runs / "task_a_unet" / "pred_masks_test",
        max_n=args.max_n,
    )


def cmd_demo(args: argparse.Namespace) -> None:
    """End-to-end pedagogical run on synthetic data, writing Plotly figures."""
    from isic2018.data.synthetic import generate_demo_dataset
    from isic2018.eval.compare import compare_methods, evaluate_classifier, evaluate_segmentation
    from isic2018.features.classical import train_classical_models
    from isic2018.train.classify import train_classifier
    from isic2018.train.segment import predict_masks, train_unet
    from isic2018.viz.montage import write_class_gallery, write_overlay_strip
    from isic2018.data.datasets import LesionDataset, load_mask, load_rgb

    cfg = _merge(args)
    seed_everything(cfg["seed"])
    paths = _paths()
    demo = cfg.get("demo", {})
    per_class = int(getattr(args, "per_class", None) or demo.get("per_class", 8))
    if getattr(args, "per_class", None):
        val_n = test_n = max(1, per_class // 4)
    else:
        val_n = int(demo.get("val_per_class", 2))
        test_n = int(demo.get("test_per_class", 2))
    data = generate_demo_dataset(
        out_dir=_data_root(args, paths),
        image_size=int(cfg["image_size"]),
        per_class=per_class,
        val_per_class=val_n,
        test_per_class=test_n,
        imbalanced=bool(demo.get("imbalanced", True)),
        seed=cfg["seed"],
    )
    write_class_gallery(data, paths.figures / "demo_gallery.png")

    seg = cfg.get("segmentation", {})
    clf = cfg.get("classification", {})
    classical_cfg = cfg.get("classical", {})
    size = int(cfg["image_size"])
    epochs = int(seg.get("epochs", 6))

    train_unet(
        data_root=data,
        image_size=size,
        epochs=epochs,
        batch_size=int(seg.get("batch_size", 8)),
        lr=float(seg.get("lr", 1e-3)),
        base_channels=int(seg.get("base_channels", 16)),
        dice_weight=float(seg.get("dice_weight", 0.5)),
        out_dir=paths.runs / "task_a_unet",
        seed=cfg["seed"],
    )
    unet_ckpt = paths.runs / "task_a_unet" / "unet_best.pt"
    evaluate_segmentation(data, unet_ckpt, paths.figures, size, split="test")
    pred_train = predict_masks(data, unet_ckpt, "train", paths.runs / "task_a_unet" / "pred_masks", size)
    predict_masks(data, unet_ckpt, "val", paths.runs / "task_a_unet" / "pred_masks", size)
    pred_test = predict_masks(data, unet_ckpt, "test", paths.runs / "task_a_unet" / "pred_masks_test", size)

    # One qualitative overlay strip.
    ds = LesionDataset(data, "test", image_size=size)
    row = ds.frame.iloc[0]
    img = load_rgb(data / row["image_path"], size=size)
    gt = load_mask(data / row["mask_path"], size=size)
    pred = load_mask(pred_test / f"{row['image_id']}.png", size=size)
    write_overlay_strip(img, gt, pred, paths.figures / "task_a_overlay_example.png", title=f"{row['image_id']}  GT vs U-Net")

    cnn_results = []
    for mode in ("whole", "crop_gt", "crop_pred"):
        train_classifier(
            data_root=data,
            image_size=size,
            epochs=int(clf.get("epochs", epochs)),
            batch_size=int(clf.get("batch_size", 8)),
            lr=float(clf.get("lr", 1e-3)),
            backbone=str(clf.get("backbone", "resnet18")),
            pretrained=bool(clf.get("pretrained", False)),
            crop_mode=mode,
            imbalance=str(clf.get("imbalance", "weights")),
            freeze_epochs=int(clf.get("freeze_epochs", 0)),
            out_dir=paths.runs / f"task_b_{mode}",
            pred_mask_dir=pred_train if mode == "crop_pred" else None,
            seed=cfg["seed"],
        )
        metrics = evaluate_classifier(
            data,
            paths.runs / f"task_b_{mode}" / "clf_best.pt",
            paths.figures,
            size,
            crop_mode=mode,
            pred_mask_dir=pred_train if mode == "crop_pred" else None,
            tag=f"clf_{mode}",
        )
        metrics["name"] = {
            "whole": "CNN whole-image",
            "crop_gt": "CNN crop (GT mask)",
            "crop_pred": "CNN crop (U-Net mask)",
        }[mode]
        cnn_results.append(metrics)

    classical = train_classical_models(
        data_root=data,
        image_size=size,
        out_dir=paths.runs / "classical",
        n_estimators=int(classical_cfg.get("n_estimators", 200)),
        max_depth=int(classical_cfg.get("max_depth", 6)),
        seed=cfg["seed"],
    )
    compare_methods(cnn_results, classical, paths.figures)
    summary = {
        "data": str(data),
        "figures": str(paths.figures),
        "cuda": cuda_report(),
        "task_a_checkpoint": str(unet_ckpt),
        "note": "Figures are Plotly HTML/PNG under results/figures/. Open Napari with: pixi run viz",
    }
    dump_json(summary, paths.runs / "demo_summary.json")
    print(json.dumps(summary, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isic2018",
        description="Pedagogical ISIC 2018 segmentation + classification (PyTorch, CUDA, Napari, Plotly).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Print Torch / CUDA / class names")
    info.add_argument("--cuda", action="store_true", help="ignored; CUDA is always reported")
    _add_common(info)
    info.set_defaults(func=cmd_info)

    gen = sub.add_parser("generate-demo", help="Write synthetic dermoscopy-like images")
    _add_common(gen)
    gen.add_argument("--per-class", type=int, default=None)
    gen.set_defaults(func=cmd_generate_demo)

    dl = sub.add_parser("download", help="Download official ISIC 2018 archives (opt-in)")
    _add_common(dl)
    dl.add_argument("--agree", action="store_true", help="Acknowledge ISIC terms of use")
    dl.add_argument("--dest", type=Path, default=None)
    dl.add_argument("--tasks", default="1,3", help="Comma-separated official task numbers")
    dl.add_argument("--delete-zip", action="store_true")
    dl.set_defaults(func=cmd_download)

    ts = sub.add_parser("train-seg", help="Train Task A U-Net")
    _add_common(ts)
    ts.set_defaults(func=cmd_train_seg)

    es = sub.add_parser("eval-seg", help="Evaluate Task A U-Net and write Plotly figures")
    _add_common(es)
    es.add_argument("--checkpoint", type=Path, default=None)
    es.add_argument("--split", default="test")
    es.set_defaults(func=cmd_eval_seg)

    tc = sub.add_parser("train-clf", help="Train Task B CNN")
    _add_common(tc)
    tc.add_argument("--crop-mode", choices=["whole", "crop_gt", "crop_pred"], default=None)
    tc.set_defaults(func=cmd_train_clf)

    ec = sub.add_parser("eval-clf", help="Evaluate Task B CNN (ROC, PR, Grad-CAM)")
    _add_common(ec)
    ec.add_argument("--crop-mode", choices=["whole", "crop_gt", "crop_pred"], default=None)
    ec.add_argument("--checkpoint", type=Path, default=None)
    ec.add_argument("--split", default="test")
    ec.set_defaults(func=cmd_eval_clf)

    tcl = sub.add_parser("train-classical", help="Train RF / XGBoost on ABCD features")
    _add_common(tcl)
    tcl.set_defaults(func=cmd_train_classical)

    cmp_ = sub.add_parser("compare", help="Whole-image vs crop vs classical comparison")
    _add_common(cmp_)
    cmp_.set_defaults(func=cmd_compare)

    viz = sub.add_parser("viz", help="Open Napari (image / mask / prediction)")
    _add_common(viz)
    viz.add_argument("--split", default="test")
    viz.add_argument("--max-n", type=int, default=16)
    viz.set_defaults(func=cmd_viz)

    demo = sub.add_parser("demo", help="Full synthetic pipeline + Plotly figures")
    _add_common(demo)
    demo.add_argument("--per-class", type=int, default=None)
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
