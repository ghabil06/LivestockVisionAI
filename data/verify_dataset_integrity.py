"""
Comprehensive Dataset Integrity & Manifest Verification.
Ensures 100% of referenced image files exist on disk, are valid non-corrupt images,
and load successfully with strict=True (raising hard errors on any missing files).
"""

from pathlib import Path
import sys
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARENT_DIR = PROJECT_ROOT.parent
for p in [str(PARENT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from LivestockVisionAI.data.dataset_path_resolver import resolve_image_path
from LivestockVisionAI.data.dataset_loader import MultiViewLivestockDataset


def verify_manifest(manifest_path: Path, species_name: str) -> bool:
    print(f"--- Auditing Manifest: {manifest_path.name} ({species_name}) ---")
    if not manifest_path.exists():
        print(f"  [FAIL] Manifest does not exist: {manifest_path}")
        return False

    df = pd.read_csv(manifest_path)
    total_animals = len(df)
    print(f"  Total animal records: {total_animals}")

    missing_files = 0
    corrupt_files = 0
    checked_files = 0

    for idx, row in df.iterrows():
        spec = str(row.get("species", species_name))
        for v in range(4):
            col = f"view_{v}_path"
            if col in row and pd.notna(row[col]):
                raw_p = str(row[col])
                resolved = resolve_image_path(raw_p, species=spec)
                checked_files += 1
                if not resolved.exists():
                    missing_files += 1
                    if missing_files <= 3:
                        print(f"  [MISSING] Row {idx} View {v}: {raw_p} -> {resolved}")
                else:
                    try:
                        with Image.open(resolved) as im:
                            im.verify()
                    except Exception as e:
                        corrupt_files += 1
                        print(f"  [CORRUPT] Row {idx} View {v}: {resolved} ({e})")

    print(f"  Checked {checked_files} images: {missing_files} missing, {corrupt_files} corrupt.")

    # Test loading with MultiViewLivestockDataset in strict mode
    try:
        ds = MultiViewLivestockDataset(manifest_path=manifest_path, strict=True)
        sample = ds[0]
        assert sample["images"].shape == (4, 3, 224, 224)
        print(f"  [PASS] MultiViewLivestockDataset strict=True loaded sample successfully: shape {sample['images'].shape}")
    except Exception as e:
        print(f"  [FAIL] MultiViewLivestockDataset strict mode failed: {e}")
        return False

    return missing_files == 0 and corrupt_files == 0


def main():
    manifests = [
        (PROJECT_ROOT / "data" / "manifests" / "cattle_manifest.csv", "COW"),
        (PROJECT_ROOT / "data" / "manifests" / "hen_seed_manifest.csv", "HEN"),
        (PROJECT_ROOT / "data" / "manifests" / "goat_seed_manifest.csv", "GOAT"),
        (PROJECT_ROOT / "data" / "manifests" / "sheep_seed_manifest.csv", "SHEEP"),
        (PROJECT_ROOT / "data" / "manifests" / "master_livestock_manifest.csv", "ALL"),
    ]

    all_passed = True
    for mpath, spec in manifests:
        ok = verify_manifest(mpath, spec)
        if not ok:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL MANIFESTS AND REAL IMAGE DATASETS VERIFIED STRICTLY (100% INTEGRITY)")
    else:
        print("DATASET INTEGRITY FAILED: SOME IMAGES ARE MISSING OR UNREADABLE")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
