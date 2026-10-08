"""
Model definitions for the four approaches compared in the project.

1. Random Forest (baseline, sklearn): flattened grayscale pixel features.
2. Self-built 4-layer CNN (PyTorch, ~5.3M params): trained from scratch.
3. Self-built 4-layer CNN + SE-Blocks (PyTorch, ~5.31M params): channel-attention variant.
4. ResNet18 with ImageNet pre-training (PyTorch): transfer learning.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — adaptive channel-wise recalibration.

    Hu et al., "Squeeze-and-Excitation Networks", TPAMI 2020.
    Adds ~2 * C^2 / reduction parameters per block (negligible vs the
    surrounding conv stack).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pooling → (B, C)
        z = x.mean(dim=(2, 3))
        # Excitation: 2-layer bottleneck + sigmoid → (B, C) in [0, 1]
        s = torch.relu(self.fc1(z))
        s = torch.sigmoid(self.fc2(s))
        # Recalibrate: scale feature map by per-channel weight
        return x * s.unsqueeze(-1).unsqueeze(-1)


def build_cnn(img_size: int = 96) -> nn.Module:
    """4-block convolutional network. Input: (B, 3, img_size, img_size).

    Architecture:
        Block 1: Conv32 x 2 -> MaxPool -> Dropout
        Block 2: Conv64 x 2 -> MaxPool -> Dropout
        Block 3: Conv128 x 2 -> MaxPool -> Dropout
        Block 4: Conv256    -> MaxPool -> Dropout
        Classifier: Linear(256 * (S/16)^2 -> 512) -> Linear(512 -> 2)
    """
    return nn.Sequential(
        # Block 1: 32 channels
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        # Block 2: 64 channels
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        # Block 3: 128 channels
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        # Block 4: 256 channels
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        # Classifier
        nn.Flatten(),
        nn.Linear(256 * (img_size // 16) * (img_size // 16), 512),
        nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 2),
    )


def build_cnn_se(img_size: int = 96) -> nn.Module:
    """CNN with Squeeze-and-Excitation channel attention on every block.

    Identical backbone to :func:`build_cnn` plus a single SEBlock at the
    end of each of the four conv blocks. SE adds only ~11K params (0.2%
    of the 5.3M backbone) but lets the network learn per-channel
    importance — useful when some channels capture crack texture while
    others capture background noise.
    """
    return nn.Sequential(
        # Block 1: 32 channels
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        SEBlock(32),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        # Block 2: 64 channels
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        SEBlock(64),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        # Block 3: 128 channels
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        SEBlock(128),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        # Block 4: 256 channels
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        SEBlock(256),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        # Classifier
        nn.Flatten(),
        nn.Linear(256 * (img_size // 16) * (img_size // 16), 512),
        nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 2),
    )


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ResNet18 with ImageNet pre-training and a 2-class classifier head.

    The default fc head (1000 classes) is replaced with Dropout(0.5) -> Linear(512, 2).
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2))
    return model


def _swap_classifier(model: nn.Module, in_features: int) -> nn.Module:
    """Replace ImageNet 1000-class head with 2-class classifier head.

    Works for any torchvision ResNet / EfficientNet / MobileNet whose final
    layer is a single `nn.Linear` taking in_features.
    """
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
    return model


def build_resnet50(pretrained: bool = True) -> nn.Module:
    """ResNet50 with ImageNet pre-training and a 2-class classifier head.

    ~25M params — a heavier cousin of ResNet18 used to test whether the crack
    detection task rewards more backbone capacity. Default fc head swapped
    out for Dropout(0.5) -> Linear(2048, 2).
    """
    model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    return _swap_classifier(model, in_features=2048)


def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with ImageNet pre-training and a 2-class classifier head.

    ~5M params — similar parameter budget to the self-built CNN but with
    a compound-scaled, MBConv-heavy backbone. Lets us test whether
    architectural innovation inside a small parameter budget (EfficientNet-B0)
    beats the hand-crafted 4-conv-block CNN.
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    return _swap_classifier(model, in_features=1280)


def build_mobilenetv3_large(pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Large with ImageNet pre-training and a 2-class classifier head.

    ~5.4M params — designed for mobile deployment (latency-aware NAS).
    Useful as a baseline for the "model size vs accuracy" tradeoff question
    and a future TFLite / CoreML deployment target.
    """
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
    return _swap_classifier(model, in_features=1280)


def build_model(name: str, img_size: int = 96) -> nn.Module:
    """Factory for the supported model architectures."""
    if name == "cnn":
        return build_cnn(img_size=img_size)
    elif name == "cnn_se":
        return build_cnn_se(img_size=img_size)
    elif name == "resnet18":
        return build_resnet18()
    elif name == "resnet50":
        return build_resnet50()
    elif name == "efficientnet_b0":
        return build_efficientnet_b0()
    elif name == "mobilenetv3_large":
        return build_mobilenetv3_large()
    else:
        raise ValueError(
            f"Unknown model name: {name!r} (expected one of "
            f"'cnn', 'cnn_se', 'resnet18', 'resnet50', 'efficientnet_b0', "
            f"'mobilenetv3_large')"
        )


def get_default_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available torch device: mps (Apple Silicon) > cuda > cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
