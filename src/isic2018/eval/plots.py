"""All scientific plots are Plotly. Matplotlib is never used to draw figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from isic2018.utils import LOGGER, ensure_dir


ISIC_TEMPLATE = dict(
    layout=go.Layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        colorway=[
            "#c0392b",
            "#8e6b3a",
            "#e07470",
            "#d68910",
            "#b9770e",
            "#7d6608",
            "#922b21",
        ],
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
)


def save_figure(fig: go.Figure, path: Path, width: int = 900, height: int = 560) -> list[Path]:
    """Write HTML always; PNG when Kaleido is available."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        colorway=ISIC_TEMPLATE["layout"].colorway,
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    written = [html_path]
    png_path = path.with_suffix(".png")
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=2)
        written.append(png_path)
    except Exception as exc:  # kaleido missing / headless quirks
        LOGGER.warning("PNG export skipped for %s (%s). HTML written.", png_path.name, exc)
    return written


def plot_training_curves(history: list[dict[str, Any]], title: str, path: Path) -> list[Path]:
    fig = go.Figure()
    if not history:
        fig.add_annotation(text="No history", showarrow=False)
        return save_figure(fig, path)
    epochs = [h["epoch"] for h in history]
    keys = [k for k in history[0] if k != "epoch"]
    for key in keys:
        fig.add_trace(go.Scatter(x=epochs, y=[h[key] for h in history], mode="lines+markers", name=key))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Value")
    return save_figure(fig, path)


def plot_confusion(cm: Iterable[Iterable[float]], labels: list[str], title: str, path: Path) -> list[Path]:
    matrix = np.asarray(list(cm), dtype=float)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="YlOrRd",
            text=matrix.astype(int),
            texttemplate="%{text}",
            hovertemplate="true=%{y}<br>pred=%{x}<br>n=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        width=640,
        height=560,
    )
    return save_figure(fig, path, width=640, height=560)


def plot_roc(roc_curves: dict[str, dict[str, list[float]]], title: str, path: Path) -> list[Path]:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash", color="#999"))
    )
    for name, curve in roc_curves.items():
        fig.add_trace(
            go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=name)
        )
    fig.update_layout(
        title=title,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return save_figure(fig, path)


def plot_pr(pr_curves: dict[str, dict[str, list[float]]], title: str, path: Path) -> list[Path]:
    fig = go.Figure()
    for name, curve in pr_curves.items():
        fig.add_trace(
            go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", name=name)
        )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return save_figure(fig, path)


def plot_metric_bars(series: dict[str, float], title: str, path: Path) -> list[Path]:
    names = list(series.keys())
    values = [series[k] for k in names]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color="#c0392b", text=[f"{v:.3f}" for v in values], textposition="outside"))
    fig.update_layout(title=title, yaxis_title="Score", yaxis=dict(range=[0, 1.05]))
    return save_figure(fig, path)


def plot_model_comparison(rows: list[dict[str, Any]], path: Path) -> list[Path]:
    """Grouped bars: accuracy / balanced acc / macro ROC-AUC per method."""
    methods = [r["name"] for r in rows]
    fig = go.Figure()
    for metric, pretty in (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_roc_auc", "Macro ROC-AUC"),
    ):
        fig.add_trace(
            go.Bar(
                name=pretty,
                x=methods,
                y=[r.get(metric, float("nan")) for r in rows],
            )
        )
    fig.update_layout(
        title="Whole-image CNN vs segment-crop CNN vs handcrafted RF / XGBoost",
        barmode="group",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
    )
    return save_figure(fig, path, width=980, height=560)


def plot_gradcam_grid(
    images: np.ndarray,
    overlays: np.ndarray,
    titles: list[str],
    path: Path,
    max_n: int = 8,
) -> list[Path]:
    n = min(len(titles), max_n, images.shape[0])
    fig = make_subplots(rows=2, cols=n, subplot_titles=titles[:n] + [""] * n)
    for i in range(n):
        img = np.transpose(images[i], (1, 2, 0))
        if img.max() <= 1.5:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        fig.add_trace(go.Image(z=img), row=1, col=i + 1)
        fig.add_trace(go.Image(z=overlays[i]), row=2, col=i + 1)
    fig.update_layout(title="Grad-CAM (row 1: input, row 2: overlay)", height=420, width=140 * n + 80)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return save_figure(fig, path, width=140 * n + 80, height=420)
