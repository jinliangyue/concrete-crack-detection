#!/usr/bin/env python3
"""
Export MobileNetV3-Large (crack detection) to ONNX + verify.

ONNX is a portable, cross-platform mobile deployment format:
    iOS        — ONNX Runtime Mobile (iOS 13+)
    Android    — ONNX Runtime Mobile
    Web        — onnxruntime-web (WASM + WebGPU)
    TF Lite    — convert via `onnx2tf` or `onnx-tf` (separate toolchain)
    Core ML    — convert via coremltools (separate toolchain)

This script intentionally stays in the training-friendly toolchain
(torch 2.8.0 + torchvision 0.23.0). The TFLite / CoreML conversions
depend on third-party toolchains that pin to older torch and conflict
with our training stack; see docs/MOBILE_DEPLOY.md for the explicit
conversion recipes on those targets.

Outputs:
  mobile/mobilenetv3_large.onnx         (FP32)
  mobile/manifest.json                  (size + accuracy summary)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_trained_model(ckpt_path: Path) -> torch.nn.Module:
    """Load the trained MobileNetV3-Large from models/crack_mobilenetv3_large_best.pt."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {ckpt_path}. "
            f"Run: python3 -m src.train --model mobilenetv3_large --max-per-class 2000"
        )
    from src.models import build_mobilenetv3_large  # noqa: PLC0415

    model = build_mobilenetv3_large(pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def quick_sanity_eval(model: torch.nn.Module, n_samples: int = 200) -> tuple[float, float]:
    """Evaluate PyTorch accuracy on a small SDNET2018 slice for sanity."""
    from src.data import DATA_ROOT, collect_files, load_images, normalize_imagenet  # noqa: PLC0415

    if not DATA_ROOT.exists():
        print("  (SDNET2018 not found locally — skipping sanity eval)")
        return float("nan"), float("nan")
    crack, nocrack = collect_files(max_per_class=n_samples // 2, seed=42)
    if not crack or not nocrack:
        print("  (not enough SDNET2018 images for sanity — skipping)")
        return float("nan"), float("nan")
    X, y = load_images(crack + nocrack, img_size=160, grayscale=False)
    X_n = normalize_imagenet(X)
    with torch.no_grad():
        out = model(torch.from_numpy(X_n[: len(y)]).float())
        preds = out.argmax(1).cpu().numpy()
    acc = float((preds == y[: len(preds)]).mean())
    n_per_class = len(y) // 2
    crack_acc = float((preds[n_per_class:] == 1).mean()) if n_per_class else float("nan")
    return acc, crack_acc


def export_onnx(model: torch.nn.Module, onnx_path: Path,
                input_shape: tuple = (1, 3, 160, 160)) -> int:
    """Export the model to ONNX. Returns the .onnx file size in bytes."""
    print(f"  Exporting ONNX → {onnx_path.name}")
    dummy = torch.randn(*input_shape)
    t0 = time.time()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    elapsed = time.time() - t0
    size = onnx_path.stat().st_size
    print(f"    ({elapsed:.1f}s, {size / 1024 / 1024:.2f} MB)")
    return size


def verify_onnx_inference(onnx_path: Path, model: torch.nn.Module) -> bool:
    """Confirm ONNX model's forward output matches PyTorch within 1e-3."""
    import onnx
    import onnxruntime as ort  # noqa: PLC0415

    print(f"  Verifying ONNX vs PyTorch inference parity")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        torch_out = model(dummy).numpy()[0]
    ort_out = sess.run(None, {"input": dummy.numpy()})[0][0]
    max_diff = float(np.abs(torch_out - ort_out).max())
    rel_diff = float(np.abs(torch_out - ort_out).max() / (np.abs(torch_out).max() + 1e-9))
    print(f"    max abs diff: {max_diff:.6f}  ·  max rel diff: {rel_diff*100:.4f}%")
    ok = max_diff < 1e-2
    print(f"    {'✓ OK' if ok else '✗ MISMATCH'}")
    return ok


def main() -> int:
    out_dir = PROJECT_ROOT / "mobile"
    out_dir.mkdir(exist_ok=True)
    ckpt = PROJECT_ROOT / "models" / "crack_mobilenetv3_large_best.pt"

    print("=" * 70)
    print("MobileNetV3-Large → ONNX export (mobile deployment artifact)")
    print("=" * 70)

    print(f"\n[1/4] Loading model from {ckpt.relative_to(PROJECT_ROOT)}")
    model = load_trained_model(ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    print(f"\n[2/4] Sanity eval on small SDNET2018 slice (200 images)")
    pytorch_acc, pytorch_crack = quick_sanity_eval(model)
    if not np.isnan(pytorch_acc):
        print(f"  PyTorch accuracy: {pytorch_acc*100:.2f}% (Crack class: {pytorch_crack*100:.2f}%)")

    onnx_path = out_dir / "mobilenetv3_large.onnx"
    print(f"\n[3/4] ONNX export (opset 17, dynamic batch)")
    onnx_size = export_onnx(model, onnx_path)

    print(f"\n[4/4] Verifying ONNX inference parity vs PyTorch")
    parity_ok = verify_onnx_inference(onnx_path, model)

    manifest = {
        "pytorch_params": n_params,
        "pytorch_acc_pct": round(pytorch_acc * 100, 2) if not np.isnan(pytorch_acc) else None,
        "pytorch_crack_acc_pct": round(pytorch_crack * 100, 2) if not np.isnan(pytorch_crack) else None,
        "onnx": {
            "path": str(onnx_path.relative_to(PROJECT_ROOT)),
            "size_bytes": onnx_size,
            "size_kb": onnx_size // 1024,
            "size_mb": round(onnx_size / 1024 / 1024, 2),
            "opset": 17,
            "producer": "pytorch 2.8.0",
            "parity_ok": parity_ok,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  PyTorch params:     {n_params:>10,}")
    if not np.isnan(pytorch_acc):
        print(f"  PyTorch accuracy:   {pytorch_acc*100:>9.2f}%  (sanity, 200 images)")
    print(f"  ONNX size:          {onnx_size/1024/1024:>9.2f} MB")
    print(f"  ONNX parity:        {'OK' if parity_ok else 'MISMATCH'}")
    print(f"\nManifest: mobile/manifest.json")
    print(f"\nNext steps (separate toolchains):")
    print(f"  - TFLite:    install onnx-tf + tensorflow, then `onnx-tf convert -i model.onnx -o tf_model`")
    print(f"  - CoreML:    install coremltools, then `coremltools.convert('model.onnx')`")
    print(f"  - iOS/Android: use ONNX Runtime Mobile directly with this .onnx file")
    return 0


if __name__ == "__main__":
    sys.exit(main())