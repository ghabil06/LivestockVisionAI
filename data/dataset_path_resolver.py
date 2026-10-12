"""
Dataset Path Resolution Utility.
Provides environment-portable resolution for livestock image paths,
eliminating machine-specific hardcoded paths across Windows, Linux, and CI/CD.
"""

import os
from pathlib import Path
from typing import Union, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_dataset_root() -> Path:
    """Returns dataset root directory, prioritizing environment variable LIVESTOCK_DATA_DIR."""
    env_dir = os.environ.get("LIVESTOCK_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p
    return PROJECT_ROOT / "data" / "storage"


def resolve_image_path(raw_path: Union[str, Path], species: Optional[str] = None) -> Path:
    """
    Resolves an image path into an existing absolute Path.
    Checks:
    1. Direct existence of raw_path
    2. Path relative to PROJECT_ROOT
    3. Path relative to dataset root
    4. Path inside species subfolder in dataset root
    5. Stripping legacy absolute prefixes (e.g. Downloads/images) to find animal subfolder
    """
    if isinstance(raw_path, Path):
        p = raw_path
    else:
        p = Path(str(raw_path).replace("\\", "/"))

    # Direct check
    if p.exists():
        return p

    # Relative to project root
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate

    # Relative to dataset root
    data_root = get_dataset_root()
    candidate = data_root / p
    if candidate.exists():
        return candidate

    # If species specified, check data_root / species / p
    if species:
        candidate = data_root / species.lower() / p
        if candidate.exists():
            return candidate

    # Extract animal folder and filename if legacy path provided
    parts = p.parts
    if len(parts) >= 2:
        rel_tail = Path(parts[-2]) / parts[-1]
        for sub in ([species.lower()] if species else ["cattle", "cattle_dataset", "chicken", "goat", "sheep", ""]):
            candidate = data_root / sub / rel_tail
            if candidate.exists():
                return candidate
            candidate = PROJECT_ROOT / "data" / "storage" / sub / rel_tail
            if candidate.exists():
                return candidate
            if sub == "cattle":
                candidate = PROJECT_ROOT / "data" / "storage" / "cattle_dataset" / rel_tail
                if candidate.exists():
                    return candidate

    return p
