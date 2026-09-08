"""Segmentation and classification losses, including imbalance-aware options."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (2, 3) if probs.ndim == 4 else (1, 2)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Standard U-Net objective: pixel BCE + soft Dice."""

    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dsc = dice_loss(logits, targets)
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dsc


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al.) for the long-tailed Task B prior."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        log_pt = log_p.gather(1, targets[:, None]).squeeze(1)
        pt = p.gather(1, targets[:, None]).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()


def classification_loss(
    imbalance: str,
    class_weights: torch.Tensor | None,
) -> nn.Module:
    if imbalance == "focal":
        return FocalLoss(weight=class_weights)
    if imbalance in {"weights", "sampler", "none"}:
        weight = class_weights if imbalance == "weights" else None
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown imbalance mode {imbalance!r}")
