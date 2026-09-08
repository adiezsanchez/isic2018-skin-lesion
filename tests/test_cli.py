from __future__ import annotations

import pytest

from isic2018.cli import main
from isic2018.data.download import download_isic


def test_download_requires_explicit_agreement():
    with pytest.raises(SystemExit):
        download_isic(agree=False)


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "train-seg" in out
    assert "demo" in out
