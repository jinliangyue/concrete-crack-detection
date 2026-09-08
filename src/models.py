"""
Model definitions for the three approaches compared in the project.

1. Random Forest (baseline, sklearn): flattened grayscale pixel features.
2. Self-built 4-layer CNN (PyTorch, ~5.3M params): trained from scratch.
3. ResNet18 with ImageNet pre-training (PyTorch): transfer learning.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


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


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ResNet18 with ImageNet pre-training and a 2-class classifier head.

    The default fc head (1000 classes) is replaced with Dropout(0.5) -> Linear(512, 2).
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2))
    return model


def build_model(name: str, img_size: int = 96) -> nn.Module:
    """Factory for the three supported model architectures."""
    if name == "cnn":
        return build_cnn(img_size=img_size)
    elif name == "resnet18":
        return build_resnet18()
    else:
        raise ValueError(f"Unknown model name: {name!r} (expected 'cnn' or 'resnet18')")


def get_default_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available torch device: mps (Apple Silicon) > cuda > cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
