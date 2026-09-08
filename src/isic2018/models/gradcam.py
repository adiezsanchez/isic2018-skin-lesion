"""Grad-CAM (Selvaraju et al., 2017) for the Task B CNN.

The implementation is intentionally short so students can read it:

1. Register a forward hook on the last conv layer to stash activations A.
2. Register a backward hook to stash ∂y_c / ∂A.
3. Channel weights α_k = GAP(∂y_c / ∂A^k).
4. CAM = ReLU(Σ_k α_k A^k), upsampled to the input resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from isic2018.models.classifiers import last_conv_layer


def _magma_rgb(values: np.ndarray) -> np.ndarray:
    """Tiny magma-like LUT so heatmaps do not depend on matplotlib."""
    x = np.clip(values, 0.0, 1.0)
    r = np.clip(2.2 * x - 0.15, 0, 1)
    g = np.clip(1.4 * x - 0.45, 0, 1) * 0.55 + 0.05 * x
    b = np.clip(1.8 * (0.5 - np.abs(x - 0.35)), 0, 1) * 0.9 + 0.15 * (1 - x)
    rgb = np.stack([r, g, b], axis=-1)
    return (255.0 * rgb).astype(np.uint8)


def overlay_cam(image_chw: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] CAM onto an RGB image in [0, 1] CHW or HWC."""
    if image_chw.ndim == 3 and image_chw.shape[0] in {1, 3}:
        image = np.transpose(image_chw, (1, 2, 0))
    else:
        image = image_chw
    image_u8 = np.clip(image * 255.0 if image.max() <= 1.5 else image, 0, 255).astype(np.uint8)
    heat = _magma_rgb(cam)
    blend = (1.0 - alpha) * image_u8.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


@dataclass
class GradCAM:
    model: nn.Module
    target_layer: nn.Module | None = None

    def __post_init__(self) -> None:
        self.model.eval()
        self.target_layer = self.target_layer or last_conv_layer(self.model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        cam = F.relu(cam)
        b, _, h, w = cam.shape
        flat = cam.view(b, -1)
        min_v = flat.min(dim=1, keepdim=True).values
        max_v = flat.max(dim=1, keepdim=True).values
        cam = (flat - min_v) / (max_v - min_v + 1e-8)
        return cam.view(b, 1, h, w)

    def __call__(
        self,
        images: torch.Tensor,
        class_idx: int | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (cams[B,H,W], predicted class indices)."""
        self.model.zero_grad(set_to_none=True)
        images = images.requires_grad_(True)
        logits = self.model(images)
        pred = logits.argmax(dim=1)
        if class_idx is None:
            target = pred
        elif isinstance(class_idx, int):
            target = torch.full_like(pred, class_idx)
        else:
            target = class_idx.to(pred.device)
        score = logits.gather(1, target[:, None]).sum()
        score.backward()
        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = self._normalize(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze(1).detach().cpu().numpy(), pred.detach().cpu().numpy()
