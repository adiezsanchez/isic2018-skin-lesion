from __future__ import annotations

import numpy as np

from isic2018.data.synthetic import generate_one
from isic2018.features.handcrafted import FEATURE_NAMES, extract_features


def test_feature_vector_length_and_finite():
    rng = np.random.default_rng(1)
    image, mask = generate_one("MEL", size=64, rng=rng)
    feats = extract_features(image, (mask > 0).astype(np.uint8))
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feats).all()
