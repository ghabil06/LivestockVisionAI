"""
Unit tests for data ingestion, manifests, and dataset loaders.
Verifies unit conversions, fold isolation, and PyTorch dataset batch shapes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pytest

from LivestockVisionAI.data.ingestion import LivestockDataIngestion
from LivestockVisionAI.data.dataset_loader import (
    MultiViewLivestockDataset,
    SingleViewLivestockDataset,
    create_grouped_cv_loaders,
)
from LivestockVisionAI.data.schemas.animal_record import Species

# ---------------------------------------------------------------------------
# Environment guard: tests that physically load cattle images from disk are
# skipped when the cattle dataset directory is not present.  This allows the
# suite to pass from a clean repository checkout or on an auditor's machine
# without the 2,000+ image corpus.  Set the environment variable
# LIVESTOCK_DATA_DIR to the dataset root to override the default location.
# ---------------------------------------------------------------------------
import os

_CATTLE_DATASET_DIR = Path(
    os.environ.get("LIVESTOCK_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "storage" / "cattle_dataset")
)
_CATTLE_IMAGES_AVAILABLE = _CATTLE_DATASET_DIR.exists() and any(_CATTLE_DATASET_DIR.iterdir())

_requires_cattle_images = pytest.mark.skipif(
    not _CATTLE_IMAGES_AVAILABLE,
    reason=(
        "Cattle image dataset not available in this environment. "
        "Set LIVESTOCK_DATA_DIR to the dataset root, or run data/populate_seed_images.py first."
    ),
)



@pytest.fixture
def manifest_dir():
    return Path(__file__).resolve().parent.parent / "data" / "manifests"


def test_cattle_manifest_integrity(manifest_dir):
    cat_manifest = manifest_dir / "cattle_manifest.csv"
    assert cat_manifest.exists()
    df = pd.read_csv(cat_manifest)

    # 510 filtered cattle
    assert len(df) == 510
    assert (df["species"] == "COW").all()

    # Unit conversion checks: withers height in cm
    assert "withers_height_cm" in df.columns
    sample = df.iloc[0]
    # Height in inches was 50.9 -> in cm: 50.9 * 2.54 = 129.286 -> 129.3
    assert abs(sample["withers_height_cm"] - 129.3) < 0.2

    # Dentition pairs check: teeth was 2 -> 1 pair
    assert sample["dentition_pairs"] == 1


def test_master_manifest_species_coverage(manifest_dir):
    master_manifest = manifest_dir / "master_livestock_manifest.csv"
    assert master_manifest.exists()
    df = pd.read_csv(master_manifest)

    species_set = set(df["species"].unique())
    assert {"COW", "HEN", "GOAT", "SHEEP"}.issubset(species_set)

    # Hen records must not have dentition pairs
    hens = df[df["species"] == "HEN"]
    assert hens["dentition_pairs"].isna().all()

    # All weights must be strictly positive
    assert (df["ground_truth_weight_kg"] > 0).all()


def test_grouped_fold_animal_isolation(manifest_dir):
    """Assert zero animal ID leakage between train and validation splits."""
    master_manifest = manifest_dir / "master_livestock_manifest.csv"
    df = pd.read_csv(master_manifest)

    for val_fold in range(5):
        train_animals = set(df[df["fold_id"] != val_fold]["animal_id"])
        val_animals = set(df[df["fold_id"] == val_fold]["animal_id"])

        intersection = train_animals.intersection(val_animals)
        assert len(intersection) == 0, f"Fold {val_fold} leaked animal IDs: {intersection}"


@_requires_cattle_images
def test_multiview_dataset_loader(manifest_dir):
    cat_manifest = manifest_dir / "cattle_manifest.csv"
    dataset = MultiViewLivestockDataset(manifest_path=cat_manifest, folds=[0])

    assert len(dataset) > 0
    sample = dataset[0]

    # Verify tensor shapes
    images = sample["images"]
    assert isinstance(images, torch.Tensor)
    assert images.shape == (4, 3, 224, 224)

    assert isinstance(sample["weight_target"], torch.Tensor)
    assert sample["species"] == "COW"
    assert sample["raw_weight_kg"] > 0.0


@_requires_cattle_images
def test_singleview_dataset_loader(manifest_dir):
    cat_manifest = manifest_dir / "cattle_manifest.csv"
    dataset = SingleViewLivestockDataset(manifest_path=cat_manifest, folds=[0])

    assert len(dataset) > 0
    sample = dataset[0]

    # Single view should be [3, 224, 224]
    image = sample["image"]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert sample["species"] == "COW"


@_requires_cattle_images
def test_log_target_transformation(manifest_dir):
    cat_manifest = manifest_dir / "cattle_manifest.csv"
    dataset = MultiViewLivestockDataset(manifest_path=cat_manifest, folds=[0], use_log_target=True)

    sample = dataset[0]
    expected_log = np.log(sample["raw_weight_kg"])
    assert abs(sample["weight_target"].item() - expected_log) < 1e-4


@_requires_cattle_images
def test_kfold_loaders_creation(manifest_dir):
    master_manifest = manifest_dir / "master_livestock_manifest.csv"
    train_loader, val_loader = create_grouped_cv_loaders(
        manifest_path=master_manifest,
        val_fold=0,
        batch_size=8,
        multi_view=True,
    )

    # Fetch one batch from train loader
    batch = next(iter(train_loader))
    assert batch["images"].shape == (8, 4, 3, 224, 224)
    assert len(batch["weight_target"]) == 8


def test_dataset_strict_mode_and_fallback(tmp_path):
    """Verifies that strict=True raises FileNotFoundError on missing images, and strict=False flags fallback."""
    csv_file = tmp_path / "mock_manifest.csv"
    mock_df = pd.DataFrame([{
        "animal_id": "MOCK_001",
        "species": "COW",
        "breed": "KANGAYAM",
        "ground_truth_weight_kg": 340.0,
        "fold_id": 0,
        "view_0_path": str(tmp_path / "non_existent_0.jpg"),
        "view_1_path": str(tmp_path / "non_existent_1.jpg"),
        "view_2_path": str(tmp_path / "non_existent_2.jpg"),
        "view_3_path": str(tmp_path / "non_existent_3.jpg"),
    }])
    mock_df.to_csv(csv_file, index=False)

    # 1. In strict=True mode, loading missing images MUST fail loudly
    strict_ds = MultiViewLivestockDataset(manifest_path=csv_file, strict=True)
    with pytest.raises(FileNotFoundError, match="Dataset image file not found"):
        _ = strict_ds[0]

    # 2. In strict=False mode, loader provides fallback tensors and sets is_missing_fallback=True
    lenient_ds = MultiViewLivestockDataset(manifest_path=csv_file, strict=False)
    sample = lenient_ds[0]
    assert sample["images"].shape == (4, 3, 224, 224)
    assert sample["is_missing_fallback"] is True

