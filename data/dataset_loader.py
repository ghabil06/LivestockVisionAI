"""
Multi-Species PyTorch Dataset Loaders.
Loads multi-view and single-view captures with animal-isolated fold splitting.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from LivestockVisionAI.data.dataset_path_resolver import resolve_image_path

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class MultiViewLivestockDataset(Dataset):
    """
    Loads 4 views (Lateral Left, Lateral Right, Cranial, Caudal) per animal.
    Enforces strict group isolation by animal_id.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        species: Optional[str] = None,
        folds: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        use_log_target: bool = False,
        strict: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)

        # Filter by species if requested
        if species:
            spec_clean = species.upper()
            df = df[df["species"] == spec_clean]

        # Filter by cross-validation folds
        if folds is not None:
            df = df[df["fold_id"].isin(folds)]

        self.df = df.reset_index(drop=True)
        self.transform = transform or DEFAULT_TRANSFORM
        self.use_log_target = use_log_target
        self.strict = strict
        self.species_filter = species

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path_str: str, species: Optional[str] = None) -> Tuple[torch.Tensor, bool]:
        """
        Safely load and transform an image from disk with automatic path resolution.
        When strict=True, raises FileNotFoundError or IOError if missing or unreadable.
        When strict=False, returns a zero-tensor and sets is_fallback=True.
        """
        resolved_path = resolve_image_path(path_str, species=species)
        if not resolved_path or not resolved_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Dataset image file not found on disk: {path_str} (resolved as: {resolved_path})")
            logger.warning("Missing image file on disk: %s (using neutral placeholder)", path_str)
            return torch.zeros(3, 224, 224), True
        try:
            img = Image.open(resolved_path).convert("RGB")
            return self.transform(img), False
        except Exception as e:
            if self.strict:
                raise IOError(f"Failed to load image file {resolved_path}: {e}") from e
            logger.warning("Unreadable image file %s: %s (using neutral placeholder)", resolved_path, e)
            return torch.zeros(3, 224, 224), True

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        spec = str(row["species"])

        # Load all 4 views with path resolution
        v0, fb0 = self._load_image(str(row["view_0_path"]), species=spec)
        v1, fb1 = self._load_image(str(row["view_1_path"]), species=spec)
        v2, fb2 = self._load_image(str(row["view_2_path"]), species=spec)
        v3, fb3 = self._load_image(str(row["view_3_path"]), species=spec)

        # Stack into [4, 3, 224, 224]
        stacked_images = torch.stack([v0, v1, v2, v3], dim=0)

        weight_kg = float(row["ground_truth_weight_kg"])
        target = np.log(weight_kg) if self.use_log_target else weight_kg

        return {
            "images": stacked_images,  # Shape [4, 3, 224, 224]
            "weight_target": torch.tensor(target, dtype=torch.float32),
            "raw_weight_kg": weight_kg,
            "animal_id": str(row["animal_id"]),
            "species": str(row["species"]),
            "breed": str(row.get("breed", "UNKNOWN")),
            "fold_id": int(row["fold_id"]),
            "withers_height_cm": float(row["withers_height_cm"]) if pd.notna(row.get("withers_height_cm")) else None,
            "dentition_pairs": int(row["dentition_pairs"]) if pd.notna(row.get("dentition_pairs")) else None,
            "is_missing_fallback": (fb0 or fb1 or fb2 or fb3),
        }


class SingleViewLivestockDataset(Dataset):
    """Loads primary lateral profile view (View 0) for single-image estimation."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        species: Optional[str] = None,
        folds: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        use_log_target: bool = False,
        strict: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        if species:
            df = df[df["species"] == species.upper()]
        if folds is not None:
            df = df[df["fold_id"].isin(folds)]

        self.df = df.reset_index(drop=True)
        self.transform = transform or DEFAULT_TRANSFORM
        self.use_log_target = use_log_target
        self.strict = strict

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path_str: str, species: Optional[str] = None) -> Tuple[torch.Tensor, bool]:
        resolved_path = resolve_image_path(path_str, species=species)
        if not resolved_path or not resolved_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Dataset image file not found on disk: {path_str} (resolved as: {resolved_path})")
            return torch.zeros(3, 224, 224), True
        try:
            img = Image.open(resolved_path).convert("RGB")
            return self.transform(img), False
        except Exception as e:
            if self.strict:
                raise IOError(f"Failed to load image file {resolved_path}: {e}") from e
            return torch.zeros(3, 224, 224), True

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        spec = str(row["species"])
        img, fb = self._load_image(str(row["view_0_path"]), species=spec)
        weight_kg = float(row["ground_truth_weight_kg"])
        target = np.log(weight_kg) if self.use_log_target else weight_kg

        return {
            "image": img,
            "weight_target": torch.tensor(target, dtype=torch.float32),
            "raw_weight_kg": weight_kg,
            "animal_id": str(row["animal_id"]),
            "species": spec,
            "breed": str(row.get("breed", "UNKNOWN")),
            "fold_id": int(row["fold_id"]),
            "withers_height_cm": float(row["withers_height_cm"]) if pd.notna(row.get("withers_height_cm")) else None,
            "dentition_pairs": int(row["dentition_pairs"]) if pd.notna(row.get("dentition_pairs")) else None,
            "is_missing_fallback": fb,
        }


def livestock_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator handling heterogeneous multi-species metadata with optional None values."""
    is_multiview = "images" in batch[0]
    img_key = "images" if is_multiview else "image"
    images = torch.stack([item[img_key] for item in batch], dim=0)
    weight_target = torch.stack([item["weight_target"] for item in batch], dim=0)

    return {
        img_key: images,
        "weight_target": weight_target,
        "raw_weight_kg": [item["raw_weight_kg"] for item in batch],
        "animal_id": [item["animal_id"] for item in batch],
        "species": [item["species"] for item in batch],
        "breed": [item["breed"] for item in batch],
        "fold_id": torch.tensor([item["fold_id"] for item in batch], dtype=torch.long),
        "withers_height_cm": [item.get("withers_height_cm") for item in batch],
        "dentition_pairs": [item.get("dentition_pairs") for item in batch],
        "is_missing_fallback": [item.get("is_missing_fallback", False) for item in batch],
    }


def create_grouped_cv_loaders(
    manifest_path: Union[str, Path],
    val_fold: int,
    species: Optional[str] = None,
    batch_size: int = 16,
    multi_view: bool = True,
    use_log_target: bool = False,
    num_folds: int = 5,
    strict: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders ensuring no animal leaks between folds.
    """
    all_folds = list(range(num_folds))
    train_folds = [f for f in all_folds if f != val_fold]
    val_folds = [val_fold]

    ds_class = MultiViewLivestockDataset if multi_view else SingleViewLivestockDataset

    train_ds = ds_class(
        manifest_path=manifest_path,
        species=species,
        folds=train_folds,
        use_log_target=use_log_target,
        strict=strict,
    )

    val_ds = ds_class(
        manifest_path=manifest_path,
        species=species,
        folds=val_folds,
        use_log_target=use_log_target,
        strict=strict,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=livestock_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=livestock_collate_fn,
    )

    return train_loader, val_loader
