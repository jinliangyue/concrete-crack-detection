#!/usr/bin/env python3
"""
Convert mobile/mobilenetv3_large.onnx → TFLite (FP32 + int8 PTQ).

This script lives outside the training-friendly toolchain by design — see
docs/MOBILE_DEPLOY.md for the why. It requires a TensorFlow install
that conflicts with the project's torch 2.8.0 + torchvision 0.23.0
stack on macOS Python 3.9. Run it in:

    - Linux + Python 3.10+
    - macOS + Python 3.10+ (Apple Silicon or Intel)
    - Windows + Python 3.10+
    - Inside the official tensorflow docker image

Usage:
    pip install "tensorflow>=2.13" onnx onnx-tf
    python3 scripts/convert_onnx_to_tflite.py
    # → mobile/mobilenetv3_large_fp32.tflite
    # → mobile/mobilenetv3_large_int8.tflite
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = PROJECT_ROOT / "mobile" / "mobilenetv3_large.onnx"
TF_DIR = PROJECT_ROOT / "mobile" / "tf_saved_model"
FP32_OUT = PROJECT_ROOT / "mobile" / "mobilenetv3_large_fp32.tflite"
INT8_OUT = PROJECT_ROOT / "mobile" / "mobilenetv3_large_int8.tflite"


def main() -> int:
    # Lazy imports — the user might be on a system where TF isn't installed.
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError as e:
        print("ERROR: missing one of: onnx, tensorflow, onnx-tf")
        print(f"  install with: pip install tensorflow onnx onnx-tf")
        print(f"  detail: {e}")
        return 2

    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH.relative_to(PROJECT_ROOT)} not found.")
        print("  Run: python3 scripts/export_onnx.py")
        return 1

    print(f"[1/4] Loading ONNX from {ONNX_PATH.relative_to(PROJECT_ROOT)}")
    onnx_model = onnx.load(str(ONNX_PATH))

    print(f"[2/4] Converting ONNX → TensorFlow SavedModel")
    TF_DIR.mkdir(parents=True, exist_ok=True)
    tf_rep = prepare(onnx_model, device="CPU")
    tf_rep.export_graph(str(TF_DIR))
    print(f"    SavedModel: {TF_DIR.relative_to(PROJECT_ROOT)}")

    print(f"[3/4] Converting SavedModel → TFLite FP32")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_DIR))
    fp32_model = converter.convert()
    FP32_OUT.write_bytes(fp32_model)
    print(f"    {FP32_OUT.relative_to(PROJECT_ROOT)}: {len(fp32_model)/1024/1024:.2f} MB")

    print(f"[4/4] Converting SavedModel → TFLite int8 (post-training quant)")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_model = converter.convert()
    INT8_OUT.write_bytes(int8_model)
    print(f"    {INT8_OUT.relative_to(PROJECT_ROOT)}: {len(int8_model)/1024/1024:.2f} MB")

    print()
    print("Done. Both TFLite artifacts written.")
    print("Expected accuracy delta vs PyTorch 86.67%: 0-1pp for FP32, 0-2pp for int8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())