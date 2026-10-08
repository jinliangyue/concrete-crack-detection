#!/usr/bin/env bash
#
# Quick K-fold cross-validation demo (~2 minutes).
#
# Runs Stratified 5-fold CV on a small subset (1000 images / 500 per class) so
# you can sanity-check the K-fold pipeline without waiting for the full
# 4000-per-class training. The numbers you get from this demo are NOT the
# README headline numbers — the headline is the single 85/15 stratified split
# at max-per-class=4000 (see README.md "Results" table).
#
# Usage:
#     bash scripts/run_kfold_demo.sh
#
# Output:
#     results/rf_kfold.json
#     results/cnn_se_kfold.json   (if --all flag)
#     results/resnet18_kfold.json (if --all flag)
#
# This script intentionally uses --out-dir so it doesn't clobber the canonical
# results/*.json files.

set -euo pipefail

cd "$(dirname "$0")/.."

PY="python3 -u -m src.train"

echo "=== RF 5-fold (max-per-class=500, ~30s) ==="
$PY --model rf --max-per-class 500 --folds 5 --out-dir results_kfold_demo

if [[ "${1:-}" == "--all" ]]; then
    echo
    echo "=== CNN+SE 5-fold (max-per-class=500, ~5 min on MPS) ==="
    $PY --model cnn_se --max-per-class 500 --folds 5 --epochs 10 \
        --out-dir results_kfold_demo

    echo
    echo "=== ResNet18 5-fold (max-per-class=500, ~10 min on MPS) ==="
    $PY --model resnet18 --max-per-class 500 --folds 5 --epochs 8 \
        --out-dir results_kfold_demo
fi

echo
echo "Done. Aggregated metrics are in results_kfold_demo/results/*_kfold.json."