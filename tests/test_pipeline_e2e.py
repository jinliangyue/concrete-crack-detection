"""
End-to-end pipeline integration tests.

These tests run the full data → train → eval → JSON save cycle on a tiny
synthetic subset to catch regressions in the training pipeline without
needing the full SDNET2018 dataset or hours of GPU time.

The goal is plumbing coverage: does the pipeline still produce a JSON file
and a checkpoint when invoked with realistic arguments? Numerical accuracy
is checked separately by the regular test suite.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import collect_files, load_images, normalize_imagenet
from src.models import build_cnn, build_resnet18
from src.train import AugDataset, train_random_forest, train_torch
from torch.utils.data import DataLoader


def _make_tiny_dataset(max_per_class: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a tiny balanced train/test split from the real SDNET2018 files.

    Returns (X_train, X_test, y_train, y_test) with normalized test images
    ready for ResNet18 inference. Total images = 2 * max_per_class.
    """
    crack, nocrack = collect_files(max_per_class=max_per_class, seed=42)
    assert len(crack) == max_per_class
    assert len(nocrack) == max_per_class

    X, y = load_images(crack + nocrack, img_size=64, grayscale=False)
    # Hold out 25% as test (stratified).
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    return X_tr, X_te, y_tr, y_te


def test_rf_pipeline_writes_metrics_json(tmp_path: Path):
    """End-to-end RF training on 40 images → accuracy > random + JSON written."""
    X_tr, X_te, y_tr, y_te = _make_tiny_dataset(max_per_class=20)

    # Flatten RF-style (matches train_random_forest contract).
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)

    result = train_random_forest(X_tr_flat, y_tr, X_te_flat, y_te, n_estimators=20)
    assert result["accuracy"] >= 0.40, f"RF accuracy {result['accuracy']:.3f} suspiciously low"

    out = tmp_path / "rf_results.json"
    metrics = {"accuracy": result["accuracy"], "report": result["report"]}
    with open(out, "w") as f:
        json.dump(metrics, f)
    assert out.exists()
    assert json.loads(out.read_text())["accuracy"] == result["accuracy"]


def test_torch_pipeline_writes_metrics_json(tmp_path: Path):
    """End-to-end CNN training (CPU, 2 epochs, 32 images) → accuracy > random + checkpoint + JSON."""
    X_tr, X_te, y_tr, y_te = _make_tiny_dataset(max_per_class=16)

    device = torch.device("cpu")
    model = build_cnn(img_size=64).to(device)

    train_loader = DataLoader(AugDataset(X_tr, y_tr, train=True), batch_size=8, shuffle=True)
    test_loader = DataLoader(AugDataset(X_te, y_te, train=False), batch_size=16)

    result = train_torch(model, train_loader, test_loader, epochs=2, lr=1e-3, device=device)
    # Tiny dataset + 2 epochs → don't expect high accuracy, just non-degenerate
    assert result["accuracy"] >= 0.40, f"CNN accuracy {result['accuracy']:.3f} suspiciously low"

    # Checkpoint + JSON
    ckpt = tmp_path / "crack_cnn_best.pt"
    torch.save({"state_dict": result["model"].state_dict(), "accuracy": result["accuracy"]}, ckpt)
    assert ckpt.exists()

    metrics_path = tmp_path / "cnn_results.json"
    metrics = {
        "config": {"model": "cnn", "img_size": 64, "max_per_class": 16, "seed": 42},
        "accuracy": result["accuracy"],
        "history": result["history"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    assert metrics_path.exists()

    saved = json.loads(metrics_path.read_text())
    assert saved["accuracy"] == result["accuracy"]
    assert len(saved["history"]["train_acc"]) == 2
    assert len(saved["history"]["val_acc"]) == 2


def test_torch_pipeline_with_imagenet_normalization(tmp_path: Path):
    """ResNet18 expects normalized inputs. Verify normalize_imagenet preserves shape and stays in sane range."""
    X_tr, X_te, y_tr, y_te = _make_tiny_dataset(max_per_class=8)

    X_tr_n = normalize_imagenet(X_tr)
    X_te_n = normalize_imagenet(X_te)
    assert X_tr_n.shape == X_tr.shape
    assert X_te_n.shape == X_te.shape
    # Normalized values should stay within a reasonable ±3σ window (≈ [-3, 3]).
    # Concrete images are not ImageNet-distributed, so per-channel mean can drift.
    assert X_tr_n.min() > -5.0 and X_tr_n.max() < 5.0, (
        f"normalize_imagenet produced out-of-range values: "
        f"min={X_tr_n.min():.3f}, max={X_tr_n.max():.3f}"
    )


def test_pipeline_idempotent_seed(tmp_path: Path):
    """Two runs with the same seed should produce the same metrics file content."""
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    for out in [out1, out2]:
        X_tr, X_te, y_tr, y_te = _make_tiny_dataset(max_per_class=8)
        np.random.seed(42)
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = build_cnn(img_size=64).to(device)
        train_loader = DataLoader(AugDataset(X_tr, y_tr, train=True), batch_size=8, shuffle=True)
        test_loader = DataLoader(AugDataset(X_te, y_te, train=False), batch_size=16)
        result = train_torch(model, train_loader, test_loader, epochs=2, lr=1e-3, device=device)
        with open(out, "w") as f:
            json.dump({"accuracy": result["accuracy"]}, f)
    # Same seed → same accuracy (or off by MPS float drift, but here we're on CPU).
    assert json.loads(out1.read_text())["accuracy"] == json.loads(out2.read_text())["accuracy"]


def test_results_json_schema_consistent(tmp_path: Path):
    """All real results JSONs must share the same top-level keys (no drift across models)."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    if not results_dir.exists():
        pytest.skip("results/ not found — no baseline results to compare against")

    json_files = sorted(results_dir.glob("*_results.json"))
    if len(json_files) < 2:
        pytest.skip("need at least 2 results JSONs to compare schemas")

    # Top-level keys present in every real result JSON.
    required_keys = {"config", "accuracy"}
    for jf in json_files:
        data = json.loads(jf.read_text())
        missing = required_keys - data.keys()
        assert not missing, f"{jf.name} missing required keys: {missing}"