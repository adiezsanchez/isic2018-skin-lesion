from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from isic2018.eval.plots import plot_confusion, save_figure


def test_plotly_html_is_written(tmp_path: Path):
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], name="diag"))
    written = save_figure(fig, tmp_path / "curve")
    html = [p for p in written if p.suffix == ".html"]
    assert html and html[0].exists()
    text = html[0].read_text(encoding="utf-8")
    assert "plotly" in text.lower()


def test_confusion_html(tmp_path: Path):
    written = plot_confusion([[3, 1], [0, 4]], ["NV", "MEL"], "cm", tmp_path / "cm")
    assert any(p.suffix == ".html" for p in written)
