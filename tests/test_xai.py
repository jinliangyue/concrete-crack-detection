"""
Smoke tests for Grad-CAM (src.xai).

Uses a randomly initialised CNN so the tests don't depend on a trained
checkpoint or the SDNET2018 dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import _resolve_target_layer, gradcam_heatmap, overlay_heatmap  # noqa: E402


@pytest.fixture(scope="module")
def cnn32():
    """A small randomly initialised CNN for fast deterministic tests."""
    from src.models import build_cnn

    torch.manual_seed(0)
    model = build_cnn(img_size=32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def resnet18_random():
    """ResNet18 with random init (no ImageNet download) and a 2-class head."""
    import torchvision.models as tvm

    torch.manual_seed(0)
    model = tvm.resnet18(weights=None)
    # Match src.models.build_resnet18's 2-class head so the layer4 path is valid
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model


def test_resolve_target_layer_valid(resnet18_random):
    """Happy path: 'layer4' resolves to a real nn.Module on ResNet18."""
    layer = _resolve_target_layer(resnet18_random, "layer4")
    assert isinstance(layer, torch.nn.Module)


def test_resolve_target_layer_nested(resnet18_random):
    """Nested dotted paths resolve to deep submodules."""
    layer = _resolve_target_layer(resnet18_random, "layer4.1.conv2")
    assert isinstance(layer, torch.nn.Module)


def test_resolve_target_layer_missing(resnet18_random):
    """Missing attribute raises ValueError with the offending name."""
    with pytest.raises(ValueError, match="totally.made.up"):
        _resolve_target_layer(resnet18_random, "totally.made.up")


def test_gradcam_heatmap_shape_and_range(cnn32):
    """Grad-CAM returns a (H, W) float32 array in [0, 1]."""
    img = torch.randn(1, 3, 32, 32)
    # build_cnn is a flat Sequential — pick its last conv-like module by name.
    last_conv_name = None
    for name, module in cnn32.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
    assert last_conv_name is not None, "build_cnn should expose at least one Conv2d"

    heatmap = gradcam_heatmap(cnn32, img, target_class=1, target_layer=last_conv_name)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (32, 32)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0


def test_gradcam_target_class_changes_heatmap(cnn32):
    """Different target_class should produce different heatmaps (model-dependent but not identical)."""
    torch.manual_seed(123)
    img = torch.randn(1, 3, 32, 32)
    last_conv_name = None
    for name, module in cnn32.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name

    h0 = gradcam_heatmap(cnn32, img, target_class=0, target_layer=last_conv_name)
    h1 = gradcam_heatmap(cnn32, img, target_class=1, target_layer=last_conv_name)
    # Random init may produce identical gradients in degenerate cases; allow small overlap
    # but the two should not be pixel-identical after independent runs.
    assert h0.shape == h1.shape
    diff = float(np.abs(h0 - h1).mean())
    assert diff > 1e-6, f"target_class had no effect (diff={diff})"


def test_gradcam_default_target_class_is_prediction(cnn32):
    """When target_class=None, Grad-CAM uses the argmax prediction."""
    torch.manual_seed(7)
    img = torch.randn(1, 3, 32, 32)
    last_conv_name = None
    for name, module in cnn32.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name

    with torch.no_grad():
        pred_class = int(cnn32(img).argmax(dim=1).item())
    heatmap_default = gradcam_heatmap(cnn32, img, target_layer=last_conv_name)
    heatmap_explicit = gradcam_heatmap(cnn32, img, target_class=pred_class, target_layer=last_conv_name)
    np.testing.assert_allclose(heatmap_default, heatmap_explicit, atol=1e-5)


def test_overlay_heatmap_shape_and_mode():
    """Overlay returns an RGB PIL image at the input's resolution."""
    img = Image.new("RGB", (96, 64), color=(120, 100, 80))
    heatmap = np.random.RandomState(42).rand(8, 8).astype(np.float32)
    overlay = overlay_heatmap(img, heatmap)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (96, 64)
    assert overlay.mode == "RGB"


def test_overlay_heatmap_alpha_zero_returns_original():
    """alpha=0 should yield an image visually identical to the input (no heatmap)."""
    img = Image.new("RGB", (48, 48), color=(200, 150, 100))
    heatmap = np.ones((8, 8), dtype=np.float32)
    overlay = overlay_heatmap(img, heatmap, alpha=0.0)
    arr_overlay = np.asarray(overlay).astype(np.int16)
    arr_input = np.asarray(img).astype(np.int16)
    np.testing.assert_array_equal(arr_overlay, arr_input)


def test_overlay_handles_non_square_heatmap():
    """Non-square heatmap should be upsampled to image size without errors."""
    img = Image.new("RGB", (100, 200), color=(50, 60, 70))
    heatmap = np.random.RandomState(1).rand(16, 32).astype(np.float32)
    overlay = overlay_heatmap(img, heatmap)
    assert overlay.size == (100, 200)
