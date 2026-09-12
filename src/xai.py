"""
Grad-CAM visualization for crack detection models.

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) using
PyTorch forward / backward hooks — no extra runtime dependency beyond
torch + matplotlib (already in requirements.txt).

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", IJCV 2019.

Usage:
    from src.xai import gradcam_heatmap, overlay_heatmap
    heatmap = gradcam_heatmap(model, image_tensor, target_class=1)
    overlay_pil = overlay_heatmap(image_pil, heatmap)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _resolve_target_layer(model: nn.Module, target_layer: str) -> nn.Module:
    """Locate a submodule by its dotted attribute path (e.g. ``layer4``)."""
    module: nn.Module = model
    for attr in target_layer.split("."):
        if not hasattr(module, attr):
            raise ValueError(
                f"Target layer '{target_layer}' not found — '{attr}' missing."
            )
        module = getattr(module, attr)
    return module


def gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: str = "layer4",
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model
        A model in ``eval()`` mode. For ResNet18 built by
        :func:`src.models.build_resnet18`, the default ``"layer4"`` is the
        last residual block whose 3×3 convolutions carry the deepest
        spatial signal.
    image_tensor
        Shape ``(1, 3, H, W)``, float32, ImageNet-normalised for ResNet18
        (use :func:`src.data.normalize_imagenet` on a batch first).
    target_class
        Class index to explain. ``None`` (default) explains the model's
        predicted class — index ``0`` = NoCrack, ``1`` = Crack.
    target_layer
        Dotted path of the convolutional layer whose feature maps drive the
        explanation.

    Returns
    -------
    heatmap
        ``np.ndarray`` of shape ``(H, W)`` and dtype ``float32``,
        normalised to ``[0, 1]``. The hottest cell is the part of the image
        most responsible for the prediction.
    """
    model.eval()

    target = _resolve_target_layer(model, target_layer)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module, _inputs, output):
        activations.append(output.detach())

    def bwd_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target.register_forward_hook(fwd_hook)
    bwd_handle = target.register_full_backward_hook(bwd_hook)

    try:
        output = model(image_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        model.zero_grad()
        score = output[0, target_class]
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Hooks did not fire — check target_layer path.")

        act = activations[0]               # (1, C, h, w)
        grad = gradients[0]               # (1, C, h, w)
        weights = grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=image_tensor.shape[2:], mode="bilinear", align_corners=False,
        )
        heatmap = cam[0, 0].cpu().numpy().astype(np.float32)

        # Min-max normalisation → [0, 1]
        lo, hi = float(heatmap.min()), float(heatmap.max())
        if hi > lo:
            heatmap = (heatmap - lo) / (hi - lo)
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def overlay_heatmap(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on top of an RGB image.

    Parameters
    ----------
    image_pil
        Original PIL image (any size).
    heatmap
        ``np.ndarray`` of shape ``(h, w)`` in ``[0, 1]`` — typically
        :func:`gradcam_heatmap` output.
    alpha
        Blend weight of the heatmap (0 = original image, 1 = pure heatmap).
    colormap
        Matplotlib colormap name. Defaults to ``"jet"`` (blue → red, classic
        Grad-CAM). Other good choices: ``"turbo"``, ``"hot"``, ``"inferno"``.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless — required when Streamlit renders offscreen
    import matplotlib.cm as cm

    cmap = getattr(cm, colormap)

    # Upsample heatmap to image resolution
    w, h = image_pil.size
    heat_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    heat_resized = (
        np.array(Image.fromarray(heat_uint8).resize((w, h), Image.BILINEAR)) / 255.0
    )

    heat_rgb = (cmap(heat_resized)[:, :, :3] * 255).astype(np.uint8)

    img_arr = np.asarray(image_pil.convert("RGB"))
    blended = ((1.0 - alpha) * img_arr + alpha * heat_rgb).astype(np.uint8)
    return Image.fromarray(blended)
