"""
Tests for benchmark.external.run_cross_domain
=============================================

We test the script's machinery — model loading, surface-data loader,
inference path, JSON serialization — using a tiny SDNET2018 fixture
or, when the dataset is unavailable, mocking the model and loader.

The actual numerical baseline is produced by running the script with
`--save-json` on real checkpoints; that is too slow for a unit test.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.external import run_cross_domain


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def fake_X_y():
    """Tiny (N=20) random RGB batch and matching labels.

    Cracks vs uncracks are not actually distinguishable here — the
    goal is to exercise the pipeline, not the model.
    """
    rng = np.random.default_rng(42)
    X = rng.random((20, 3, 160, 160), dtype=np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    return X, y


@pytest.fixture
def stub_models(monkeypatch):
    """Replace _load_model with a deterministic stub.

    Avoids disk + torch.load in unit tests.
    """
    import torch
    import torch.nn as nn

    class _ConstantModel(nn.Module):
        def __init__(self, label: int = 1):
            super().__init__()
            self._label = label

        def forward(self, x):
            b = x.shape[0]
            # Always return logits that argmax to self._label.
            out = torch.zeros(b, 2)
            out[:, self._label] = 1.0
            return out

    def _fake_load(name):
        # CNN/SE predict crack; ResNet18 predicts no-crack.
        label = 0 if name == "resnet18" else 1
        return _ConstantModel(label=label), CHECKPOINTS_FOR_STUB[name]["img_size"], torch.device("cpu")

    monkeypatch.setattr(run_cross_domain, "_load_model", _fake_load)
    return _fake_load


CHECKPOINTS_FOR_STUB = {
    "cnn":     {"img_size": 96},
    "cnn_se":  {"img_size": 96},
    "resnet18": {"img_size": 160},
}


# --------------------------------------------------------------------- model zoo
def test_checkpoints_dict_has_expected_models():
    expected = {"cnn", "cnn_se", "resnet18"}
    assert set(run_cross_domain.CHECKPOINTS) == expected


def test_surface_labels_map_covers_all_surfaces():
    assert set(run_cross_domain.SURFACE_LABELS) == {"D", "P", "W"}


# --------------------------------------------------------------------- evaluate_one
def test_evaluate_one_runs_and_reports_metrics(stub_models, fake_X_y):
    X, y = fake_X_y
    r = run_cross_domain.evaluate_one(
        "resnet18", "D", X, y,
        max_per_class=10, seed=42,
    )
    # The stub ResNet18 always predicts class 0 (uncracked), so:
    #   accuracy = 10/20 (correctly predicted 10 uncracks, missed 10 cracks)
    #   recall on cracks = 0
    assert r["model"] == "resnet18"
    assert r["surface"] == "D"
    assert r["n_total"] == 20
    assert r["n_crack"] == 10
    assert r["n_uncracked"] == 10
    assert 0.0 <= r["accuracy"] <= 1.0
    assert r["precision"] == 0.0  # no TP (predicted class 0 throughout)
    assert r["recall"] == 0.0     # 0 of 10 cracks found
    assert "elapsed_ms" in r


def test_evaluate_one_perfect_classifier(stub_models, fake_X_y):
    """Stub that always predicts class 1 (crack) should give 100 % recall."""
    import torch
    import torch.nn as nn

    X, y = fake_X_y

    class _AlwaysCrack(nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 2)
            out[:, 1] = 1.0
            return out

    run_cross_domain._load_model = lambda n: (_AlwaysCrack(), 96, torch.device("cpu"))
    r = run_cross_domain.evaluate_one("cnn", "P", X, y, max_per_class=10, seed=0)
    # All 20 predicted as crack; 10 actually crack → 100 % recall
    assert r["recall"] == 1.0


# --------------------------------------------------------------------- argparse + json
def test_main_writes_results_json(tmp_path, stub_models, monkeypatch, capsys):
    """End-to-end with stubbed models and 5 images per class — fast."""
    out_json = tmp_path / "results.json"
    monkeypatch.setattr(run_cross_domain, "_load_surface",
                        lambda surface, max_per_class, seed: (
                            np.random.default_rng(0).random((4, 3, 160, 160), dtype=np.float32),
                            np.array([0, 0, 1, 1], dtype=np.int64),
                        ))
    rc = run_cross_domain.main([
        "--max-per-class", "2",
        "--surfaces", "D",
        "--models", "cnn",
        "--save-json", str(out_json),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["model"] == "cnn"
    assert r["surface"] == "D"
    assert r["n_total"] == 4


def test_main_skips_missing_checkpoint(tmp_path, monkeypatch, capsys):
    """If a checkpoint file is missing, that model is skipped with a clear message."""
    def _raise(name):
        raise FileNotFoundError(f"checkpoint for {name!r} not found")

    monkeypatch.setattr(run_cross_domain, "_load_model", _raise)
    monkeypatch.setattr(run_cross_domain, "_load_surface",
                        lambda surface, max_per_class, seed: (
                            np.zeros((2, 3, 160, 160), dtype=np.float32),
                            np.array([0, 1], dtype=np.int64),
                        ))
    rc = run_cross_domain.main([
        "--max-per-class", "1",
        "--surfaces", "D",
        "--models", "cnn",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[skip]" in captured.out


# --------------------------------------------------------------------- _predict resize
def test_predict_resizes_when_model_img_size_smaller(stub_models):
    """_predict should downsample 160x160 input to 96x96 when the model wants 96."""
    import torch
    rng = np.random.default_rng(1)
    X = rng.random((4, 3, 160, 160), dtype=np.float32)
    # Use the cnn stub (img_size 96) — input will be resized.
    model, img_size, device = stub_models("cnn")
    preds = run_cross_domain._predict(model, X, img_size, device)
    assert preds.shape == (4,)
    assert set(preds.tolist()) <= {0, 1}
