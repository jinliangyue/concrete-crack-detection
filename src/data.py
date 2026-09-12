"""
Shared data loading for SDNET2018 crack detection.

SDNET2018 layout:
    data/DATA_Maguire_20180517_ALL/SDNET2018/
        D/{CD,UD}/   — bridge deck (cracked / uncracked)
        P/{CP,UP}/   — pavement
        W/{CW,UW}/   — wall
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "DATA_Maguire_20180517_ALL" / "SDNET2018"

# ImageNet channel statistics (used by CNN/ResNet18 normalization)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Canonical split: surface → (cracked_subdir, uncracked_subdir)
SUBDIRS = {
    "D": ("CD", "UD"),
    "P": ("CP", "UP"),
    "W": ("CW", "UW"),
}


def collect_files(
    max_per_class: Optional[int] = None,
    surfaces: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Walk SDNET2018 subdirectories and return balanced file lists.

    Returns (crack_files, nocrack_files). Each list contains Paths to images.
    Sampling is random with the given seed for reproducibility.
    """
    surfaces = surfaces or list(SUBDIRS.keys())
    files_crack: List[Path] = []
    files_nocrack: List[Path] = []

    for surface in surfaces:
        crack_sub, nocrack_sub = SUBDIRS[surface]
        for sub, bucket in [(crack_sub, files_crack), (nocrack_sub, files_nocrack)]:
            folder = DATA_ROOT / surface / sub
            if not folder.exists():
                continue
            bucket.extend(sorted(folder.glob("*.jpg")))

    if max_per_class is not None:
        # Legacy np.random.choice matches the original paper's sampling semantics
        # and gives reproducible results as long as the caller has seeded np.random
        # (src.train.main does this before invoking collect_files).
        np.random.seed(seed)
        if len(files_crack) > max_per_class:
            files_crack = list(np.random.choice(files_crack, max_per_class, replace=False))
        if len(files_nocrack) > max_per_class:
            files_nocrack = list(np.random.choice(files_nocrack, max_per_class, replace=False))

    return files_crack, files_nocrack


def load_images(
    file_paths: List[Path],
    img_size: int,
    grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images into numpy arrays.

    For RGB (grayscale=False): shape (N, 3, H, W) float32 in [0, 1].
    For grayscale (grayscale=True): shape (N, H*W) float32 in [0, 1].

    Skips any file that fails to open.
    """
    X_list, y_list = [], []
    skipped = 0
    for fp in file_paths:
        try:
            mode = "L" if grayscale else "RGB"
            img = Image.open(fp).convert(mode).resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if grayscale:
                X_list.append(arr.flatten())
            else:
                X_list.append(arr.transpose(2, 0, 1))
            y_list.append(1 if fp.parent.name.startswith("C") else 0)
        except Exception:
            skipped += 1

    if skipped:
        print(f"  skipped {skipped} unreadable images")

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64)


def normalize_imagenet(X: np.ndarray) -> np.ndarray:
    """Apply ImageNet channel-wise normalization to a (N, 3, H, W) batch."""
    return (X - IMAGENET_MEAN) / IMAGENET_STD


def count_per_class(files_crack: List[Path], files_nocrack: List[Path]) -> dict:
    """Return a breakdown of file counts per surface / class — useful for reporting."""
    breakdown: dict = {}
    for fp in files_crack + files_nocrack:
        surface = fp.parent.parent.name
        sub = fp.parent.name
        key = (surface, sub)
        breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown
