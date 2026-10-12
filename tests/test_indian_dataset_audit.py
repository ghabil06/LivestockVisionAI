"""
Unit and Integration Tests for Batch B3: Indian Cattle Dataset Audit & Suitability.

Validates:
  1. Integrity of the deduplicated Indian cattle manifest (4,975 unique images).
  2. Removal of 2,251 exact binary duplicate images.
  3. Strict absence of weight and age ground-truth labels.
  4. Detection of Tamil Nadu native breed gaps (Kangayam, Umblachery, Pulikulam, Alambadi).
  5. System enforcement prohibiting unverified breed claims.
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from LivestockVisionAI.data.audit_indian_dataset import TAMIL_NADU_BREEDS


@pytest.fixture(scope="module")
def audit_artifacts():
    pkg_root = Path(__file__).resolve().parent.parent
    manifest_path = pkg_root / "data" / "manifests" / "indian_breeds_deduplicated_manifest.csv"
    metrics_path = pkg_root / "data" / "manifests" / "indian_dataset_audit_metrics.json"
    report_path = pkg_root / "docs" / "B3_indian_dataset_suitability_report.md"

    return {
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
    }


def test_deduplicated_manifest_integrity(audit_artifacts):
    """Verifies that the deduplicated manifest exists, contains 4,975 rows, and has 0 duplicate hashes."""
    p = audit_artifacts["manifest_path"]
    assert p.exists(), f"Deduplicated manifest not found at {p}"

    df = pd.read_csv(p)
    assert len(df) == 4975
    assert df["md5_hash"].nunique() == 4975
    assert "breed" in df.columns
    assert "zip_internal_path" in df.columns

    # Verify breed distribution in deduplicated dataset
    counts = df["breed"].value_counts().to_dict()
    assert counts["Bargur"] == 470
    assert counts["Gir"] == 2443
    assert counts["Hallikar"] == 937
    assert counts["Ongole"] == 311
    assert counts["Tharparkar"] == 814


def test_metrics_json_duplication_statistics(audit_artifacts):
    """Asserts that the metrics JSON correctly logs the 2,251 duplicate files (31.2% rate)."""
    p = audit_artifacts["metrics_path"]
    assert p.exists(), f"Metrics JSON not found at {p}"

    with open(p, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["total_files_in_archive"] == 7226
    assert metrics["unique_images_count"] == 4975
    assert metrics["duplicate_files_count"] == 2251
    assert abs(metrics["overall_duplication_rate_pct"] - 31.2) < 0.2

    # Verify severe Hallikar duplication (52.0%)
    hallikar = metrics["breed_statistics"]["Hallikar"]
    assert hallikar["duplicate_files"] == 1014
    assert abs(hallikar["duplication_rate_pct"] - 52.0) < 0.2


def test_absence_of_weight_and_age_modalities(audit_artifacts):
    """Asserts that the dataset audit records complete absence of scale and age measurements."""
    p = audit_artifacts["metrics_path"]
    with open(p, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    status = metrics["label_modality_status"]
    assert "COMPLETELY_ABSENT" in status["weight_labels"]
    assert "COMPLETELY_ABSENT" in status["age_labels"]
    assert "COMPLETELY_ABSENT" in status["dentition_labels"]
    assert "UNTRACKED" in status["animal_id_tracking"]

    verdict = metrics["operational_verdict"]
    assert verdict["suitable_for_weight_estimation"] is False
    assert verdict["suitable_for_age_estimation"] is False
    assert verdict["suitable_for_multiview_fusion"] is False
    assert verdict["suitable_for_breed_classification"] is True


def test_tamil_nadu_native_breed_gaps(audit_artifacts):
    """Verifies that 4 out of 5 prominent Tamil Nadu native breeds are flagged as missing."""
    p = audit_artifacts["metrics_path"]
    with open(p, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    tn_cov = metrics["tamil_nadu_breed_coverage"]

    # Bargur is present (470 images)
    assert tn_cov["BARGUR"]["present_in_dataset"] is True
    assert tn_cov["BARGUR"]["unique_images"] == 470

    # Kangayam, Umblachery, Pulikulam, Alambadi are strictly absent
    for missing_breed in ["KANGAYAM", "UMBLACHERY", "PULIKULAM", "ALAMBADI"]:
        assert tn_cov[missing_breed]["present_in_dataset"] is False
        assert tn_cov[missing_breed]["unique_images"] == 0
        assert tn_cov[missing_breed]["status"] == "MISSING_CRITICAL_GAP"


def test_markdown_report_formatting(audit_artifacts):
    """Verifies that the markdown report exists and contains explicit cautionary alerts."""
    p = audit_artifacts["report_path"]
    assert p.exists()
    content = p.read_text(encoding="utf-8")

    assert "Batch B3: Indian Cattle Dataset Suitability & Audit Report" in content
    assert "Mandatory Deduplication" in content
    assert "Tamil Nadu Breed Support Policy" in content
    assert "Kangayam" in content
