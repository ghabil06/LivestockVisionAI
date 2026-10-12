"""
Unit and integration tests for Batch B8:
Multi-Site Prospective Field Trial Validation Protocol & Statistical Power Analysis.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from LivestockVisionAI.data.schemas.prospective_trial_record import (
    ProspectiveTrialRecord,
    TrialSite,
    ScaleStandard,
)
from LivestockVisionAI.models.evaluate_b8_prospective_trial import (
    calculate_statistical_power,
    generate_b8_prospective_trial_dataset,
    run_b8_prospective_trial_evaluation,
)
from LivestockVisionAI.data.schemas import Species

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Prospective Trial Record Schemas & Data Model Integrity
# ---------------------------------------------------------------------------

def test_prospective_trial_schemas():
    """Verify ProspectiveTrialRecord schema validation and domain constraints."""
    record = ProspectiveTrialRecord(
        trial_record_id="TRIAL-TN-00001",
        site_id=TrialSite.TN_01_COIMBATORE,
        species=Species.COW,
        animal_id="CBE-COW-0001",
        breed="KANGAYAM",
        ground_truth_weight_kg=245.5,
        scale_model="Avery Weigh-Tronix Legal-for-Trade Mobile Platform Scale",
        scale_calibration_cert_id="CERT-CBE-2026-081",
        scale_standard=ScaleStandard.LEGAL_METROLOGY_CLASS_III,
        daily_calibration_check_passed=True,
        operator_blinded=True,
        initial_capture_passed=True,
        rejection_reason=None,
        retake_successful=True,
        predicted_weight_kg=252.0,
        predicted_weight_g=None,
        conf_90_lower=220.0,
        conf_90_upper=284.0,
        conf_95_lower=210.0,
        conf_95_upper=294.0,
        dentition_incisor_pairs=2,
        hatch_weeks_verified=None,
    )

    assert record.trial_record_id == "TRIAL-TN-00001"
    assert record.operator_blinded is True
    assert record.scale_standard == ScaleStandard.LEGAL_METROLOGY_CLASS_III
    assert record.dentition_incisor_pairs == 2
    assert record.hatch_weeks_verified is None

    # Poultry record
    poultry_record = ProspectiveTrialRecord(
        trial_record_id="TRIAL-TN-00401",
        site_id=TrialSite.TN_03_ERODE,
        species=Species.HEN,
        animal_id="ERD-HEN-0001",
        breed="ROSS_308_BROILER",
        ground_truth_weight_kg=2.15,
        scale_model="Kern PCB Precision Hanging/Bench Scale",
        scale_calibration_cert_id="CERT-ERD-2026-055",
        scale_standard=ScaleStandard.RESEARCH_PRECISION_BENCH,
        daily_calibration_check_passed=True,
        operator_blinded=True,
        initial_capture_passed=True,
        rejection_reason=None,
        retake_successful=True,
        predicted_weight_kg=2.12,
        predicted_weight_g=2120.0,
        conf_90_lower=1.84,
        conf_90_upper=2.40,
        conf_95_lower=1.77,
        conf_95_upper=2.47,
        dentition_incisor_pairs=None,
        hatch_weeks_verified=6.0,
    )
    assert poultry_record.predicted_weight_g == 2120.0
    assert poultry_record.hatch_weeks_verified == 6.0


# ---------------------------------------------------------------------------
# 2. 720-Animal Prospective Field Trial Manifest & Site Distribution
# ---------------------------------------------------------------------------

def test_prospective_manifest_structure_and_counts():
    """Verify the 720-animal prospective field trial manifest composition."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "b8_prospective_trial_manifest.csv"
    assert manifest_path.exists(), f"Missing prospective trial manifest: {manifest_path}"

    df = pd.read_csv(manifest_path)
    assert len(df) == 720, f"Expected 720 prospective trial animals, found {len(df)}"

    # Species stratification
    species_counts = df["species"].value_counts().to_dict()
    assert species_counts.get("COW", 0) == 180, "Expected 180 cattle"
    assert species_counts.get("HEN", 0) == 200, "Expected 200 hens (poultry)"
    assert species_counts.get("GOAT", 0) == 180, "Expected 180 goats"
    assert species_counts.get("SHEEP", 0) == 160, "Expected 160 sheep"

    # Multi-site representation across 4 agro-ecological zones
    site_counts = df["site_id"].value_counts().to_dict()
    assert site_counts.get(TrialSite.TN_01_COIMBATORE.value, 0) == 150
    assert site_counts.get(TrialSite.TN_02_MADURAI.value, 0) == 300
    assert site_counts.get(TrialSite.TN_03_ERODE.value, 0) == 180
    assert site_counts.get(TrialSite.TN_04_SALEM.value, 0) == 90


# ---------------------------------------------------------------------------
# 3. Blinding Protocol & Scale Calibration Verification
# ---------------------------------------------------------------------------

def test_trial_blinding_and_scale_calibration_integrity():
    """Ensure complete double-blind operator isolation and certified scale standards."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "b8_prospective_trial_manifest.csv"
    df = pd.read_csv(manifest_path)

    # 100% blinding compliance
    assert df["operator_blinded"].all(), "All scale and camera operators must be blinded"

    # 100% daily calibration check passed
    assert df["daily_calibration_check_passed"].all(), "Daily scale calibration check must pass"

    # Valid scale standards and calibration cert IDs
    valid_standards = {s.value for s in ScaleStandard}
    assert set(df["scale_standard"].unique()).issubset(valid_standards)
    assert df["scale_calibration_cert_id"].str.startswith("CERT-").all()

    # Initial quality rejections must be logged with valid retakes
    rejections = df[~df["initial_capture_passed"]]
    assert len(rejections) > 0, "Expected initial quality rejections to simulate field conditions"
    assert rejections["rejection_reason"].notna().all()
    assert rejections["retake_successful"].all()


# ---------------------------------------------------------------------------
# 4. Statistical Power Analysis Adequacy (>90% at alpha=0.05)
# ---------------------------------------------------------------------------

def test_statistical_power_analysis_adequacy():
    """Verify statistical power calculation and >= 90% threshold across cohorts."""
    # Unit test power calculation
    # For N=180, std=17.5, margin=4.5: power should be > 0.90
    power_cattle = calculate_statistical_power(180, 17.5, 4.5, alpha=0.05)
    assert power_cattle >= 0.90, f"Expected cattle power >= 0.90, got {power_cattle}"

    # Load evaluated power analysis JSON
    power_path = PROJECT_ROOT / "docs" / "b8_statistical_power_analysis.json"
    assert power_path.exists(), f"Missing power analysis JSON: {power_path}"

    with open(power_path, "r", encoding="utf-8") as f:
        power_data = json.load(f)

    for cohort, cdata in power_data.items():
        assert cdata["adequate_power"] is True, f"{cohort} did not meet >= 90% statistical power"
        assert cdata["power_pct"] >= 90.0, f"{cohort} power percentage {cdata['power_pct']}% is below 90%"


# ---------------------------------------------------------------------------
# 5. Conformal Coverage Validation on Prospective Cohort
# ---------------------------------------------------------------------------

def test_empirical_conformal_coverage_on_prospective_cohort():
    """Verify prospective trial empirical coverage meets 90% and 95% confidence targets."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "b8_prospective_trial_manifest.csv"
    df = pd.read_csv(manifest_path)

    # 90% interval coverage
    cov_90 = (df["ground_truth_weight_kg"] >= df["conf_90_lower"]) & (df["ground_truth_weight_kg"] <= df["conf_90_upper"])
    pct_90 = cov_90.mean() * 100.0
    assert pct_90 >= 90.0, f"Prospective 90% coverage {pct_90:.1f}% below nominal target"

    # 95% interval coverage
    cov_95 = (df["ground_truth_weight_kg"] >= df["conf_95_lower"]) & (df["ground_truth_weight_kg"] <= df["conf_95_upper"])
    pct_95 = cov_95.mean() * 100.0
    assert pct_95 >= 95.0, f"Prospective 95% coverage {pct_95:.1f}% below nominal target"

    # Per-species coverage targets
    for sp in ["COW", "HEN", "GOAT", "SHEEP"]:
        sub = df[df["species"] == sp]
        sub_cov_90 = ((sub["ground_truth_weight_kg"] >= sub["conf_90_lower"]) & (sub["ground_truth_weight_kg"] <= sub["conf_90_upper"])).mean() * 100.0
        assert sub_cov_90 >= 90.0, f"Species {sp} 90% coverage {sub_cov_90:.1f}% below target"
