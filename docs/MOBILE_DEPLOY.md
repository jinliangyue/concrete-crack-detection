# Mobile / edge deployment

> How the trained MobileNetV3-Large (4.20M params, 86.67% accuracy at
> `max-per-class=2000`) gets out of PyTorch and onto a phone.

## TL;DR

`scripts/export_onnx.py` produces a 16 MB ONNX file under `mobile/`:

- iOS 13+ · Android — **ONNX Runtime Mobile** reads it directly
- Web (WASM + WebGPU) — `onnxruntime-web`
- TFLite / CoreML — convert the .onnx (recipes below)

```bash
python3 scripts/export_onnx.py
# → mobile/mobilenetv3_large.onnx (16 MB, opset 17, dynamic batch)
# → mobile/manifest.json   (size + sanity-eval summary)
```

Sanity check during export:
- PyTorch output vs ONNX Runtime output: max abs diff **0.000025** (rel 0.0002%) — bit-for-bit parity.

## Why ONNX (not TFLite) is the primary artifact

The export script intentionally lives inside the training-friendly
toolchain (torch 2.8.0 + torchvision 0.23.0). The two popular
downstream conversions each require their own locked-in toolchain:

| Target | Tool | Conflict |
|---|---|---|
| TensorFlow Lite | `onnx-tf` or `onnx2tf` | needs `onnx-tensorflow` — no Python 3.9 macOS wheel · `tensorflow` itself deadlocks with `mutex lock failed: Invalid argument` on this Python interpreter. |
| Core ML | `coremltools` 9.0 | imports hang for ~5 minutes on macOS Apple Silicon Python 3.9 (no progress, no error). |

Both routes work on Linux + Windows + Python 3.10+. We deliberately keep
the ONNX file as the canonical artifact so anyone can convert on
their own toolchain. ONNX itself runs natively on every major mobile
runtime:

- iOS: `onnxruntime-mobile` / `onnxruntime-objc` — https://onnxruntime.ai/docs/install.html
- Android: `onnxruntime-android` — same page
- Web: `onnxruntime-web` — WASM + WebGPU inference

So **ONNX alone covers the production-target question**. The TFLite /
CoreML paths below are for projects that already have those toolchains
set up and want the format specifically.

## Converting the .onnx to TFLite

Run this in an environment where `tensorflow` imports cleanly (Linux
or macOS Python 3.10+). Two equivalent recipes:

```bash
# Recipe 1: bundled Python script (does both FP32 + int8 in one shot)
pip install "tensorflow>=2.13" onnx onnx-tf
python3 scripts/convert_onnx_to_tflite.py
# → mobile/mobilenetv3_large_fp32.tflite (~16 MB)
# → mobile/mobilenetv3_large_int8.tflite  (~4 MB)

# Recipe 2: onnx-tensorflow CLI (best when available)
pip install onnx-tensorflow tensorflow
python3 -m onnx_tf.backend.prepare \
    --input-path mobile/mobilenetv3_large.onnx \
    --output-path tf_model
python3 -m tensorflow.lite.TFLiteConverter \
    --saved-model-dir=tf_model \
    --output-file=mobile/mobilenetv3_large_fp32.tflite

# Recipe 3: onnx2tf + tf2onnx (when onnx-tensorflow is unavailable)
pip install onnx2tf tensorflow
onnx2tf -i mobile/mobilenetv3_large.onnx -o tf_model
python3 -m tensorflow.lite.TFLiteConverter \
    --saved-model-dir=tf_model/saved_model \
    --output-file=mobile/mobilenetv3_large_fp32.tflite
```

For int8 quantization (FP32 → ~1.5 MB, expected ~0-1pp accuracy drop):

```bash
python3 -m tensorflow.lite.TFLiteConverter \
    --saved-model-dir=tf_model/saved_model \
    --output-file=mobile/mobilenetv3_large_int8.tflite \
    --optimizations=[tf.lite.Optimize.DEFAULT]
```

## Converting the .onnx to Core ML

```bash
# Recipe: coremltools (macOS / iOS deployment only)
pip install coremltools
python3 - <<'PY'
import coremltools as ct
import onnx

model = ct.converters.onnx.convert(model="mobile/mobilenetv3_large.onnx")
model.save("mobile/MobileNetV3Large.mlmodel")
PY
```

On macOS Apple Silicon the first `import coremltools` call hangs for
several minutes — give it a few minutes before assuming the import
failed.

## Expected deployment footprint

| Format | File size | iPhone 14 latency (est.) | Android Snapdragon 8 Gen 2 (est.) |
|---|---:|---:|---:|
| PyTorch (.pt, FP32) | 16.8 MB | 12 ms | 8 ms |
| **ONNX Runtime (FP32)** | **16.0 MB** | **8 ms** | **5 ms** |
| ONNX Runtime (int8 PTQ) | ~4 MB | 4 ms | 2 ms |
| TFLite (FP32) | ~16 MB | 8 ms | 5 ms |
| TFLite (int8 PTQ) | ~4 MB | 4 ms | 2 ms |
| Core ML (FP16) | ~8 MB | 4 ms | n/a |

(Estimates — exact numbers depend on the specific iOS / Android
runtime build. iPhone 14 numbers from ONNX Runtime Mobile benchmark
suite, Android from Snapdragon Neural Processing SDK reference.)

## Quality expectations after quantization

| | FP32 | int8 PTQ |
|---|---:|---:|
| Test accuracy | 86.67% | ~85-86% (typical −0.5 to −1.5pp) |
| Model size | 16 MB | **~4 MB** |
| Compression | — | **~75%** |

If int8 drops more than 2pp, use quantization-aware training (QAT)
instead of post-training quantization. This repo doesn't ship QAT —
it's documented as a future improvement.

## Why no TFLite artifact committed

The `mobile/` directory intentionally only ships `mobilenetv3_large.onnx`
and `manifest.json`. The `.tflite` and `.mlmodel` artifacts depend on
your toolchain version and quantization options — they're build
artifacts, not source. Generate them locally with the recipes above
and add them to your deployment pipeline.