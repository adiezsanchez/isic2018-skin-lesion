from __future__ import annotations


def test_napari_imports():
    import napari

    assert hasattr(napari, "Viewer")
