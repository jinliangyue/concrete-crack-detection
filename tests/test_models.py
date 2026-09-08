"""
Smoke tests for the model architectures.

These do not train anything — they verify:
- Parameter counts match the paper claims (CNN ~5.3M, ResNet18 ~11M)
- Forward pass produces output of the expected shape (B, 2)
- build_model factory wires the right model
"""

import pytest
import torch

from src.models import build_cnn, build_model, build_resnet18, get_default_device


def test_cnn_param_count_matches_paper():
    """Paper claims ~5.3M parameters for the self-built CNN."""
    m = build_cnn(img_size=96)
    n = sum(p.numel() for p in m.parameters())
    # Allow a small tolerance for any subtle architectural differences
    assert 5_200_000 < n < 5_400_000, f"CNN has {n:,} params, expected ~5.3M"


def test_resnet18_param_count():
    """ResNet18 has ~11M parameters total."""
    m = build_resnet18()
    n = sum(p.numel() for p in m.parameters())
    assert 11_100_000 < n < 11_300_000, f"ResNet18 has {n:,} params, expected ~11M"


def test_cnn_forward_shape():
    m = build_cnn(img_size=96)
    m.eval()
    x = torch.randn(2, 3, 96, 96)
    out = m(x)
    assert out.shape == (2, 2)


def test_resnet18_forward_shape():
    m = build_resnet18()
    m.eval()
    x = torch.randn(2, 3, 160, 160)
    out = m(x)
    assert out.shape == (2, 2)


def test_resnet18_uses_imagenet_weights_by_default():
    """Default build_resnet18() should download ImageNet weights on first run."""
    m = build_resnet18()
    # First conv weights should not be all zeros (which would indicate untrained)
    w = m.conv1.weight.data
    assert w.abs().sum() > 0


def test_build_model_factory():
    """build_model('cnn') and build_model('resnet18') should return nn.Modules."""
    m_cnn = build_model("cnn", img_size=96)
    m_resnet = build_model("resnet18")
    assert isinstance(m_cnn, torch.nn.Module)
    assert isinstance(m_resnet, torch.nn.Module)


def test_build_model_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model("nonexistent")


def test_get_default_device_returns_valid_torch_device():
    d = get_default_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cpu", "mps", "cuda")
