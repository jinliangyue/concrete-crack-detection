"""
Smoke tests for the SDNET2018 dataset contract.

The training pipeline depends on a specific directory layout under
data/DATA_Maguire_20180517_ALL/SDNET2018/. These tests verify that
layout exists and the file counts match what the README documents.
"""

import numpy as np
import pytest

from src.data import DATA_ROOT, SUBDIRS, collect_files, load_images, normalize_imagenet

EXPECTED_TOTAL_IMAGES = 56_092
EXPECTED_PER_SUBDIR = {
    ("D", "CD"): 2_025,
    ("D", "UD"): 11_595,
    ("P", "CP"): 2_608,
    ("P", "UP"): 21_726,
    ("W", "CW"): 3_851,
    ("W", "UW"): 14_287,
}


def test_data_root_exists():
    assert DATA_ROOT.exists(), (
        f"SDNET2018 dataset not found at {DATA_ROOT}. "
        f"Download from https://digitalcommons.usu.edu/all_datasets/48/ "
        f"and unzip into data/DATA_Maguire_20180517_ALL/."
    )


@pytest.mark.parametrize("surface,sub", list(EXPECTED_PER_SUBDIR.keys()))
def test_subdir_file_count(surface, sub):
    folder = DATA_ROOT / surface / sub
    assert folder.exists(), f"Missing folder: {folder}"
    files = list(folder.glob("*.jpg"))
    assert len(files) == EXPECTED_PER_SUBDIR[(surface, sub)], (
        f"{surface}/{sub}: expected {EXPECTED_PER_SUBDIR[(surface, sub)]} images, "
        f"got {len(files)}"
    )


def test_total_image_count_matches_paper():
    """Total images across all 6 subdirs should equal 56,092 (paper-claimed)."""
    total = 0
    for surface, sub in EXPECTED_PER_SUBDIR:
        total += len(list((DATA_ROOT / surface / sub).glob("*.jpg")))
    assert total == EXPECTED_TOTAL_IMAGES


def test_collect_files_balanced_with_seed():
    """When max_per_class is set, both classes should have exactly that many files."""
    n = 100
    crack, nocrack = collect_files(max_per_class=n, seed=42)
    assert len(crack) == n
    assert len(nocrack) == n


def test_collect_files_seed_reproducibility():
    a_crack, a_nocrack = collect_files(max_per_class=50, seed=42)
    b_crack, b_nocrack = collect_files(max_per_class=50, seed=42)
    assert a_crack == b_crack
    assert a_nocrack == b_nocrack


def test_load_images_grayscale_shape():
    crack, _ = collect_files(max_per_class=2)
    X, y = load_images(crack[:2], img_size=64, grayscale=True)
    assert X.shape == (2, 64 * 64)
    assert X.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0
    # crack label = 1
    assert (y == 1).all()


def test_load_images_rgb_shape():
    crack, nocrack = collect_files(max_per_class=1)
    X, y = load_images(crack + nocrack, img_size=96, grayscale=False)
    assert X.shape == (2, 3, 96, 96)
    assert X.dtype == np.float32
    assert y.tolist() == [1, 0]


def test_normalize_imagenet_output_shape():
    """normalize_imagenet should accept a (N, 3, H, W) batch and return same shape."""
    X = np.random.rand(4, 3, 32, 32).astype(np.float32)
    out = normalize_imagenet(X)
    assert out.shape == X.shape
    # Output should be roughly centered around 0 (mean shifted)
    assert abs(out.mean()) < 1.0
