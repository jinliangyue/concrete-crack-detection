"""
Single-image inference helper used by the Streamlit demo.

Loads a trained checkpoint from models/, runs forward pass on a single image,
returns (predicted_label, confidence, probability_for_each_class).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_ROOT,
    load_images,
    normalize_imagenet,
)
from src.models import build_cnn, build_resnet18

LABELS = ["NoCrack", "Crack"]
CHECKPOINTS = {
    "cnn":      PROJECT_ROOT / "models" / "crack_cnn_best.pt",
    "resnet18": PROJECT_ROOT / "models" / "crack_resnet18_best.pt",
}


def _load_image_array(image_path: str | Path, img_size: int) -> np.ndarray:
    """Load + resize a single image and return (1, 3, H, W) float32 array."""
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis, ...]


def _load_torch_model(name: str, ckpt_path: Path) -> Tuple[nn.Module, dict]:
    """Build a model and load its state_dict from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    img_size = cfg.get("img_size", 96)
    if name == "cnn":
        model = build_cnn(img_size=img_size)
    elif name == "resnet18":
        model = build_resnet18(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {name!r}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"img_size": img_size, "accuracy": ckpt.get("accuracy")}


def predict(
    image_path: str | Path,
    model_name: str = "resnet18",
    ckpt_path: Path | None = None,
) -> dict:
    """Run inference on a single image.

    Returns:
        {
          "label":    "Crack" | "NoCrack",
          "confidence": float in [0, 1],
          "probs":   {"NoCrack": float, "Crack": float},
          "model":   str,
          "img_size": int,
          "checkpoint_accuracy": float | None,
        }
    """
    if model_name not in ("cnn", "resnet18"):
        raise ValueError(f"model_name must be 'cnn' or 'resnet18', got {model_name!r}")

    ckpt_path = ckpt_path or CHECKPOINTS[model_name]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Train a model first: python -m src.train --model {model_name}"
        )

    model, meta = _load_torch_model(model_name, Path(ckpt_path))
    img_size = meta["img_size"]

    x = _load_image_array(image_path, img_size=img_size)
    if model_name == "resnet18":
        x = normalize_imagenet(x)
    x_t = torch.from_numpy(x).float()

    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "label": LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {label: float(prob) for label, prob in zip(LABELS, probs)},
        "model": model_name,
        "img_size": img_size,
        "checkpoint_accuracy": meta.get("accuracy"),
    }


__all__ = ["predict", "LABELS", "CHECKPOINTS"]
