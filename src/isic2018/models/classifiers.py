"""ImageNet-initialized CNN classifiers for ISIC 2018 Task B."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


BACKBONES = {"resnet18"}


def build_classifier(
    num_classes: int = 7,
    backbone: str = "resnet18",
    pretrained: bool = False,
) -> nn.Module:
    """Transfer-learning classifier.

    For the synthetic demo, ``pretrained=False`` keeps the run offline and
    fast. For real ISIC images, ImageNet initialization is the default in
    ``configs/isic.yaml``.
    """
    if backbone != "resnet18":
        raise ValueError(f"Unsupported backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def last_conv_layer(model: nn.Module) -> nn.Module:
    """ResNet18's last spatial feature map lives on ``layer4``."""
    if hasattr(model, "layer4"):
        return model.layer4
    raise AttributeError("Cannot locate last conv layer for Grad-CAM")


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    for name, param in model.named_parameters():
        if name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze
