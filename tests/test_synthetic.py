from __future__ import annotations

import numpy as np

from isic2018.constants import CLASS_NAMES
from isic2018.data.synthetic import generate_one


def test_generate_one_mask_and_shape():
    rng = np.random.default_rng(0)
    for name in CLASS_NAMES:
        image, mask = generate_one(name, size=64, rng=rng)
        assert image.shape == (64, 64, 3)
        assert mask.shape == (64, 64)
        assert image.dtype == np.uint8
        assert mask.dtype == np.uint8
        assert mask.max() == 255
        assert mask.min() == 0
        assert int((mask > 0).sum()) > 20
