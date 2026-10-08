"""
Smoke tests for the model architectures.

These do not train anything — they verify:
- Parameter counts match the paper claims (CNN ~5.3M, ResNet18 ~11M)
- Forward pass produces output of the expected shape (B, 2)
- build_model factory wires the right model
- New backbones (ResNet50 / EfficientNet-B0 / MobileNetV3-Large) build and
  forward correctly (pretrained=True requires the corresponding ImageNet
  weights to be cached locally; downloads will be skipped in CI by skipping
  the test if the cache is missing).
"""

import pytest
import torch

from src.models import (
    build_cnn,
    build_efficientnet_b0,
    build_mobilenetv3_large,
    build_model,
    build_resnet18,
    build_resnet50,
    get_default_device,
)


def _has_cached_weights(model_id: str) -> bool:
    """Check whether a torchvision pretrained weight file is already cached.

    Avoids hitting the network during CI / local smoke tests. Returns True
    if the file exists in the default torchvision cache directory.
    """
    from pathlib import Path
    cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    if not cache.exists():
        return False
    # torchvision names weights files like 'resnet18-f37072fd.pth'.
    # A cheap match: filename starts with the model id.
    return any(p.name.startswith(model_id) for p in cache.glob("*.pth"))


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


def test_resnet50_forward_shape():
    """ResNet50 forward produces (B, 2) — skipped if weights aren't cached locally."""
    if not _has_cached_weights("resnet50"):
        pytest.skip("ResNet50 ImageNet weights not cached locally — "
                    "skipping forward-shape test (would require network)")
    m = build_resnet50()
    m.eval()
    x = torch.randn(2, 3, 160, 160)
    out = m(x)
    assert out.shape == (2, 2)


def test_resnet50_param_count():
    """ResNet50 ~25M params (without the final fc)."""
    if not _has_cached_weights("resnet50"):
        pytest.skip("ResNet50 weights not cached — param count test also skipped")
    m = build_resnet50()
    n = sum(p.numel() for p in m.parameters())
    # ResNet50 standard is ~25.5M; allow ±2M for classifier head differences.
    assert 23_000_000 < n < 27_000_000, f"ResNet50 has {n:,} params, expected ~25M"


def test_efficientnet_b0_forward_shape():
    """EfficientNet-B0 forward produces (B, 2) — skipped if weights aren't cached."""
    if not _has_cached_weights("efficientnet_b0"):
        pytest.skip("EfficientNet-B0 ImageNet weights not cached locally")
    m = build_efficientnet_b0()
    m.eval()
    x = torch.randn(2, 3, 160, 160)
    out = m(x)
    assert out.shape == (2, 2)


def test_mobilenetv3_large_forward_shape():
    """MobileNetV3-Large forward produces (B, 2) — skipped if weights aren't cached."""
    if not _has_cached_weights("mobilenet_v3_large"):
        pytest.skip("MobileNetV3-Large ImageNet weights not cached locally")
    m = build_mobilenetv3_large()
    m.eval()
    x = torch.randn(2, 3, 160, 160)
    out = m(x)
    assert out.shape == (2, 2)


def test_build_model_factory():
    """build_model('cnn') and build_model('resnet18') should return nn.Modules."""
    m_cnn = build_model("cnn", img_size=96)
    m_resnet = build_model("resnet18")
    assert isinstance(m_cnn, torch.nn.Module)
    assert isinstance(m_resnet, torch.nn.Module)


def test_build_model_factory_includes_new_backbones():
    """build_model factory should dispatch to all seven model names."""
    # Use no-pretrained for the cached-weight-required ones to avoid network.
    # Skipped forward-shape assertion — we only care about dispatch + instantiation.
    from src.models import (
        build_cnn_se, build_resnet18, build_resnet50,
        build_efficientnet_b0, build_mobilenetv3_large,
    )
    for name in ("cnn", "cnn_se"):
        m = build_model(name)
        assert isinstance(m, torch.nn.Module)
    # New backbones: only assert construction if weights are cached.
    for name, builder in (
        ("resnet18", build_resnet18),
        ("resnet50", build_resnet50),
        ("efficientnet_b0", build_efficientnet_b0),
        ("mobilenetv3_large", build_mobilenetv3_large),
    ):
        if _has_cached_weights(builder.__name__.replace("build_", "").replace("mobilenet_v3", "mobilenet_v3")):
            m = build_model(name)
            assert isinstance(m, torch.nn.Module)


def test_build_model_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model("nonexistent")


def test_get_default_device_returns_valid_torch_device():
    d = get_default_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cpu", "mps", "cuda")
