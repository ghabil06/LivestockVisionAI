"""
Release Packaging Script for LivestockVisionAI.
Packages the full production repository into a clean release ZIP archive:
- Excludes __pycache__, .pytest_cache, and OS artifacts
- Includes trained PyTorch checkpoints, ONNX models, and full Android project
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "LivestockVisionAI_v1.0.0_release.zip"

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".system_generated", ".idea", ".gradle", "build"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".tmp", ".log"}


def package_release():
    print(f"Creating release archive: {OUTPUT_ZIP}")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in EXCLUDE_EXTS or file.startswith("."):
                    continue

                full_path = Path(root) / file
                # Skip the zip itself if inside tree
                if full_path == OUTPUT_ZIP:
                    continue

                rel_path = full_path.relative_to(PROJECT_ROOT)
                # Skip giant raw datasets if present as subfolder (keep manifests)
                if "cattle_dataset" in rel_path.parts and ext in {".jpg", ".jpeg", ".png"}:
                    continue

                arcname = Path("LivestockVisionAI") / rel_path
                zipf.write(full_path, str(arcname))
                file_count += 1
                total_uncompressed_bytes += full_path.stat().st_size

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {file_count} files into {OUTPUT_ZIP.name}")
    print(f"Archive Size: {zip_size_mb:.2f} MB")
    return OUTPUT_ZIP, file_count, zip_size_mb


if __name__ == "__main__":
    package_release()
