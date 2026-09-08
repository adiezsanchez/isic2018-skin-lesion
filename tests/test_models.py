from __future__ import annotations

import torch

from isic2018.models.classifiers import build_classifier
from isic2018.models.unet import UNet


def test_unet_forward_shape():
    model = UNet(base_channels=8)
    x = torch.rand(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_resnet18_classifier_forward():
    model = build_classifier(num_classes=7, pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 7)
