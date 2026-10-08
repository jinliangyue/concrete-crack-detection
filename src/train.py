"""
Train one of four crack detection models on SDNET2018.

Usage:
    python -m src.train --model rf       # Random Forest baseline  (~59%)
    python -m src.train --model cnn      # Self-built 4-layer CNN  (~76%)
    python -m src.train --model cnn_se   # CNN + Squeeze-and-Excitation  (~?)
    python -m src.train --model resnet18 # ResNet18 transfer learning (~86%)

Outputs:
    models/crack_<model>_best.pt   — checkpoint (state_dict + config + history)
    results/<model>_results.json    — metrics + classification report + history
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data import (
    collect_files,
    load_images,
    normalize_imagenet,
)
from src.kfold import aggregate_folds, make_kfold_indices, per_fold_metrics
from src.models import (
    build_cnn,
    build_cnn_se,
    build_efficientnet_b0,
    build_mobilenetv3_large,
    build_resnet18,
    build_resnet50,
    get_default_device,
)


# Per-model defaults that match the original experiments
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "rf":                 {"img_size": 64,  "epochs": None, "lr": None,  "batch_size": None},
    "cnn":                {"img_size": 96,  "epochs": 20,   "lr": 1e-3,  "batch_size": 32},
    "cnn_se":             {"img_size": 96,  "epochs": 20,   "lr": 1e-3,  "batch_size": 32},
    "resnet18":           {"img_size": 160, "epochs": 12,   "lr": 3e-4,  "batch_size": 16},
    "resnet50":           {"img_size": 160, "epochs": 12,   "lr": 3e-4,  "batch_size": 16},
    "efficientnet_b0":    {"img_size": 160, "epochs": 12,   "lr": 3e-4,  "batch_size": 16},
    "mobilenetv3_large":  {"img_size": 160, "epochs": 12,   "lr": 3e-4,  "batch_size": 16},
}


class AugDataset(Dataset):
    """Tensor dataset with optional horizontal/vertical flip augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = False):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = self.X[i].copy()
        if self.train:
            if np.random.random() > 0.5:
                x = np.flip(x, 2).copy()  # horizontal flip
            if np.random.random() > 0.5:
                x = np.flip(x, 1).copy()  # vertical flip
        return torch.from_numpy(x).float(), torch.tensor(int(self.y[i]))


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
) -> Dict[str, Any]:
    """Train sklearn RandomForestClassifier on flattened pixel features."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, target_names=["NoCrack", "Crack"], output_dict=True,
    )
    print(f"\nFinal Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(classification_report(y_test, preds, target_names=["NoCrack", "Crack"]))
    return {
        "model": rf,
        "accuracy": float(acc),
        "report": report,
        "history": None,
        "predictions": [int(p) for p in preds],
        "labels": [int(l) for l in y_test],
    }


def train_torch(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a torch model with AdamW + CosineAnnealingLR + best-checkpoint tracking."""
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    history: Dict[str, list] = {"train_acc": [], "val_acc": [], "loss": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            t_correct += (out.argmax(1) == by).sum().item()
            t_total += len(by)
        train_acc = t_correct / t_total
        train_loss = t_loss / len(train_loader)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                v_correct += (model(bx.to(device)).argmax(1).cpu() == by).sum().item()
                v_total += len(by)
        val_acc = v_correct / v_total

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["loss"].append(train_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

        print(
            f"E{epoch + 1:2d}/{epochs} | Loss {train_loss:.4f} | "
            f"Train {train_acc:.4f} | Val {val_acc:.4f} | Best {best_acc:.4f} | "
            f"LR {opt.param_groups[0]['lr']:.6f}"
        )

    # Reload best checkpoint for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test-set evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            all_preds.extend(model(bx.to(device)).argmax(1).cpu().tolist())
            all_labels.extend(by.tolist())
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(classification_report(all_labels, all_preds, target_names=["NoCrack", "Crack"]))

    return {
        "model": model,
        "accuracy": float(acc),
        "best_val_acc": float(best_acc),
        "history": history,
        "report": classification_report(
            all_labels, all_preds, target_names=["NoCrack", "Crack"], output_dict=True,
        ),
        "predictions": [int(p) for p in all_preds],
        "labels": [int(l) for l in all_labels],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a crack detection model on SDNET2018.",
    )
    p.add_argument(
        "--model", required=True,
        choices=["rf", "cnn", "cnn_se", "resnet18", "resnet50",
                 "efficientnet_b0", "mobilenetv3_large"],
        help="rf = random forest baseline; cnn = self-built 4-layer CNN; "
             "cnn_se = CNN + SE-Blocks; resnet18/50 = ImageNet transfer; "
             "efficientnet_b0 / mobilenetv3_large = compound-scaled / mobile backbones.",
    )
    p.add_argument(
        "--max-per-class", type=int, default=4000,
        help="Max images per class (crack / nocrack) for balance. Default 4000 (=8000 total).",
    )
    p.add_argument("--img-size", type=int, default=None,
                   help="Override default image size for the chosen model.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override default epoch count (torch models only).")
    p.add_argument("--lr", type=float, default=None,
                   help="Override default learning rate (torch models only).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override default batch size (torch models only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="Force a specific torch device (mps/cuda/cpu).")
    p.add_argument("--surfaces", default="D,P,W",
                   help="Comma-separated surfaces to load (D, P, W).")
    p.add_argument("--out-dir", default=None,
                   help="Override the results/models output directories.")
    p.add_argument("--folds", type=int, default=1,
                   help="Stratified K-fold cross-validation. 1 = single train/test split "
                        "(default, backward compatible). 5 = 5-fold CV (saves per-fold "
                        "metrics + aggregated mean ± std; folds stays in_results/ JSON only, "
                        "no per-fold checkpoint).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    defaults = DEFAULTS[args.model]
    img_size = args.img_size or defaults["img_size"]
    epochs = args.epochs or defaults["epochs"]
    lr = args.lr or defaults["lr"]
    batch_size = args.batch_size or defaults["batch_size"]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from src.data import PROJECT_ROOT
    base_dir = Path(args.out_dir).resolve() if args.out_dir else PROJECT_ROOT
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training: {args.model} ===")
    print(
        f"img_size={img_size}, max_per_class={args.max_per_class}, "
        f"surfaces={args.surfaces}"
    )
    if epochs is not None:
        print(f"epochs={epochs}, lr={lr}, batch_size={batch_size}")

    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    files_crack, files_nocrack = collect_files(
        max_per_class=args.max_per_class, surfaces=surfaces, seed=args.seed,
    )
    print(f"Crack: {len(files_crack)}, NoCrack: {len(files_nocrack)}")

    grayscale = (args.model == "rf")
    print("Loading images...")
    t0 = time.time()
    X, y = load_images(files_crack + files_nocrack, img_size=img_size, grayscale=grayscale)
    print(f"  loaded {len(X)} images in {time.time() - t0:.1f}s, shape={X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    config = {
        "model": args.model,
        "img_size": img_size,
        "max_per_class": args.max_per_class,
        "surfaces": surfaces,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": args.seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    metrics: Dict[str, Any] = {"config": config}

    if args.model == "rf":
        result = train_random_forest(X_train, y_train, X_test, y_test)
        metrics["accuracy"] = result["accuracy"]
        metrics["report"] = result["report"]
    else:
        device = get_default_device(args.device)
        print(f"Device: {device}")

        X_train_n = normalize_imagenet(X_train) if args.model == "resnet18" else X_train
        X_test_n = normalize_imagenet(X_test) if args.model == "resnet18" else X_test

        train_loader = DataLoader(
            AugDataset(X_train_n, y_train, train=True),
            batch_size=batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            AugDataset(X_test_n, y_test, train=False),
            batch_size=batch_size * 2,
        )

        if args.model == "cnn":
            model = build_cnn(img_size=img_size).to(device)
        elif args.model == "cnn_se":
            model = build_cnn_se(img_size=img_size).to(device)
        elif args.model == "resnet18":
            model = build_resnet18(pretrained=True).to(device)
        elif args.model == "resnet50":
            model = build_resnet50(pretrained=True).to(device)
        elif args.model == "efficientnet_b0":
            model = build_efficientnet_b0(pretrained=True).to(device)
        elif args.model == "mobilenetv3_large":
            model = build_mobilenetv3_large(pretrained=True).to(device)
        else:
            raise ValueError(f"Unknown torch model: {args.model}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")
        config["n_params"] = int(n_params)

        result = train_torch(
            model, train_loader, test_loader, epochs=epochs, lr=lr, device=device,
        )

        # Save best checkpoint
        ckpt_path = models_dir / f"crack_{args.model}_best.pt"
        torch.save({
            "model_name": args.model,
            "state_dict": result["model"].state_dict(),
            "config": config,
            "accuracy": result["accuracy"],
            "best_val_acc": result["best_val_acc"],
            "history": result["history"],
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        metrics["accuracy"] = result["accuracy"]
        metrics["best_val_acc"] = result["best_val_acc"]
        metrics["history"] = result["history"]
        metrics["report"] = result["report"]
        metrics["checkpoint"] = str(ckpt_path.relative_to(base_dir))

    # Save predictions + labels for confusion matrix / ROC analysis (v8.3+, all models)
    if "predictions" in result and "labels" in result:
        metrics["predictions"] = result["predictions"]
        metrics["labels"] = result["labels"]

    # Persist JSON metrics
    out_path = results_dir / f"{args.model}_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics: {out_path}")

    # K-fold branch — only runs when --folds > 1. Reuses the loaded X, y so
    # the image-loading cost is paid once. Each fold re-builds the model from
    # scratch so different folds see different initializations (MPS
    # non-determinism), giving an honest mean ± std. Writes a separate
    # results/<model>_kfold.json so the existing single-split JSON stays
    # untouched (the 87.92% number that the standard rebuttal cites must not
    # quietly change).
    if args.folds > 1:
        print(f"\n=== K-fold CV: {args.folds} folds on {args.model} ===")
        fold_indices = make_kfold_indices(y, n_folds=args.folds, seed=args.seed)
        fold_metrics_list = []
        last_ckpt_state = None
        last_ckpt_acc = 0.0

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices, start=1):
            print(f"\n--- Fold {fold_idx}/{args.folds} "
                  f"(train={len(train_idx)}, test={len(test_idx)}) ---")
            np.random.seed(args.seed + fold_idx)
            torch.manual_seed(args.seed + fold_idx)

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if args.model == "rf":
                X_tr_f = X_tr.reshape(len(X_tr), -1)
                X_te_f = X_te.reshape(len(X_te), -1)
                fold_result = train_random_forest(X_tr_f, y_tr, X_te_f, y_te, n_estimators=200)
            else:
                device = get_default_device(args.device)
                X_tr_n = normalize_imagenet(X_tr) if args.model == "resnet18" else X_tr
                X_te_n = normalize_imagenet(X_te) if args.model == "resnet18" else X_te
                train_loader = DataLoader(
                    AugDataset(X_tr_n, y_tr, train=True),
                    batch_size=batch_size, shuffle=True,
                )
                test_loader = DataLoader(
                    AugDataset(X_te_n, y_te, train=False),
                    batch_size=batch_size * 2,
                )
                if args.model == "cnn":
                    model = build_cnn(img_size=img_size).to(device)
                elif args.model == "cnn_se":
                    model = build_cnn_se(img_size=img_size).to(device)
                elif args.model == "resnet18":
                    model = build_resnet18(pretrained=True).to(device)
                elif args.model == "resnet50":
                    model = build_resnet50(pretrained=True).to(device)
                elif args.model == "efficientnet_b0":
                    model = build_efficientnet_b0(pretrained=True).to(device)
                elif args.model == "mobilenetv3_large":
                    model = build_mobilenetv3_large(pretrained=True).to(device)
                else:
                    raise ValueError(f"Unknown torch model: {args.model}")
                fold_result = train_torch(
                    model, train_loader, test_loader, epochs=epochs, lr=lr, device=device,
                )
                # Keep the best fold's state_dict (last fold is fine for
                # a small portfolio demo; not the canonical "best across folds").
                if fold_result["accuracy"] > last_ckpt_acc:
                    last_ckpt_acc = fold_result["accuracy"]
                    last_ckpt_state = {
                        k: v.clone() for k, v in fold_result["model"].state_dict().items()
                    }

            fold_metrics_list.append(per_fold_metrics(
                fold_result["report"],
                predictions=fold_result.get("predictions"),
                labels=fold_result.get("labels"),
            ))
            print(f"  fold {fold_idx} accuracy = {fold_metrics_list[-1]['accuracy']:.4f}")

        aggregated = aggregate_folds(fold_metrics_list)
        kfold_metrics = {
            "config": {**config, "folds": args.folds, "kfold_seed": args.seed},
            "folds": fold_metrics_list,
            "aggregated": aggregated,
        }
        # Save best-fold checkpoint under a separate name so the canonical
        # crack_<model>_best.pt (used by Streamlit demo) stays as-is.
        if last_ckpt_state is not None:
            kfold_ckpt = models_dir / f"crack_{args.model}_best_kfold.pt"
            torch.save({
                "model_name": args.model,
                "state_dict": last_ckpt_state,
                "config": kfold_metrics["config"],
                "best_fold_acc": last_ckpt_acc,
            }, kfold_ckpt)
            kfold_metrics["checkpoint"] = str(kfold_ckpt.relative_to(base_dir))
            print(f"Saved best-fold checkpoint: {kfold_ckpt}")
        kfold_out = results_dir / f"{args.model}_kfold.json"
        with open(kfold_out, "w") as f:
            json.dump(kfold_metrics, f, indent=2, ensure_ascii=False)
        print(f"K-fold metrics: {kfold_out}")
        print(f"  aggregated accuracy = "
              f"{aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
