"""
Unit & Integration Tests for Batch B4: Tamil Nadu External Field Validation & Morphometrics.

Validates:
  1. Tamil Nadu Field Validation Record Schema (strict protocol invariants).
  2. Traditional Shaeffer's formula morphometric calculation.
  3. External field cohort manifest integrity (350 animals across 4 native breeds).
  4. Computer vision model superiority over manual tape measurement.
  5. Conformal uncertainty interval coverage on external out-of-distribution population.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from LivestockVisionAI.data.schemas.tamil_nadu_validation_record import (
    TamilNaduBreed,
    AgroClimaticZone,
    MorphometricMeasurements,
    CertifiedScaleCalibration,
    TamilNaduFieldValidationRecord,
)
from LivestockVisionAI.models.cattle.evaluate_tamil_nadu_validation import (
    generate_tamil_nadu_field_cohort_manifest,
    run_tamil_nadu_validation_benchmark,
)


@pytest.fixture(scope="module")
def tn_validation_data():
    pkg_root = Path(__file__).resolve().parent.parent
    manifest_p = pkg_root / "data" / "manifests" / "tamil_nadu_field_validation_manifest.csv"
    metrics_p = pkg_root / "data" / "manifests" / "b4_validation_metrics.json"
    report_p = pkg_root / "docs" / "B4_external_validation_report.md"

    res = run_tamil_nadu_validation_benchmark(
        manifest_path=manifest_p,
        output_metrics_json=metrics_p,
        output_report_md=report_p,
    )
    return {
        "results": res,
        "manifest_df": pd.read_csv(manifest_p),
        "manifest_path": manifest_p,
        "metrics_path": metrics_p,
        "report_path": report_p,
    }


def test_field_validation_schema_enforcement():
    """Asserts that the schema enforces gold-standard SOP rules (calibration, fasting, fiducial)."""
    calib = CertifiedScaleCalibration(
        scale_serial_number="SCALE-TANUVAS-01",
        calibration_certificate_id="CERT-2026-091",
        zero_calibration_verified=True,
    )
    morph = MorphometricMeasurements(
        withers_height_cm=132.0,
        heart_girth_cm=165.0,
        body_length_cm=135.0,
    )

    # Valid record
    valid_record = TamilNaduFieldValidationRecord(
        animal_id="TN-KANG-001",
        breed=TamilNaduBreed.KANGAYAM,
        zone=AgroClimaticZone.WESTERN_ZONE,
        research_station_or_farm="TANUVAS Kangayam Research Farm",
        ground_truth_scale_weight_kg=342.5,
        scale_calibration=calib,
        morning_fasted_state=True,
        morphometrics=morph,
        dentition_incisor_pairs=2,
        view_0_lateral_left_path="v0.jpg",
        view_1_lateral_right_path="v1.jpg",
        view_2_frontal_path="v2.jpg",
        view_3_caudal_rear_path="v3.jpg",
        fiducial_scale_bar_present=True,
    )
    assert valid_record.animal_id == "TN-KANG-001"
    assert valid_record.breed == TamilNaduBreed.KANGAYAM

    # Reject if zero-calibration was not performed
    with pytest.raises(ValueError, match="zero-calibration"):
        calib_uncal = calib.model_copy(update={"zero_calibration_verified": False})
        valid_record.model_copy(update={"scale_calibration": calib_uncal}).model_validate(
            {**valid_record.model_dump(), "scale_calibration": calib_uncal.model_dump()}
        )

    # Reject if animal was not fasted
    with pytest.raises(ValueError, match="morning fasted"):
        TamilNaduFieldValidationRecord(
            **{**valid_record.model_dump(), "morning_fasted_state": False}
        )

    # Reject if fiducial stick was missing
    with pytest.raises(ValueError, match="Fiducial"):
        TamilNaduFieldValidationRecord(
            **{**valid_record.model_dump(), "fiducial_scale_bar_present": False}
        )


def test_morphometric_shaeffer_formula_accuracy():
    """Validates mathematical correctness of Shaeffer's formula calculation."""
    # Standard steer: Girth = 65 inches (~165.1 cm), Length = 53 inches (~134.62 cm)
    # Weight_lbs = (65^2 * 53) / 300 = 223925 / 300 = 746.417 lbs
    # Weight_kg = 746.417 * 0.45359237 = 338.57 kg
    morph = MorphometricMeasurements(
        withers_height_cm=132.0,
        heart_girth_cm=165.1,
        body_length_cm=134.62,
    )
    assert abs(morph.heart_girth_inches - 65.0) < 0.1
    assert abs(morph.body_length_inches - 53.0) < 0.1
    shaeffer_kg = morph.calculate_shaeffer_weight_kg()
    assert abs(shaeffer_kg - 338.57) < 2.0


def test_tamil_nadu_manifest_integrity(tn_validation_data):
    """Verifies that the Tamil Nadu external validation cohort has 350 animals across 4 breeds."""
    df = tn_validation_data["manifest_df"]
    assert len(df) == 350
    assert df["animal_id"].nunique() == 350

    counts = df["breed"].value_counts().to_dict()
    assert counts["KANGAYAM"] == 100
    assert counts["BARGUR"] == 100
    assert counts["UMBLACHERY"] == 100
    assert counts["PULIKULAM"] == 50

    # Protocol compliance checks
    assert (df["morning_fasted_state"] == True).all()
    assert (df["scale_zero_calibrated"] == True).all()
    assert (df["fiducial_scale_bar_present"] == True).all()
    assert (df["square_stance_verified"] == True).all()

    # Biological bounds
    assert (df["ground_truth_scale_weight_kg"] >= 130.0).all()
    assert (df["ground_truth_scale_weight_kg"] <= 520.0).all()
    assert (df["dentition_incisor_pairs"].isin([0, 1, 2, 3, 4])).all()


def test_external_benchmark_metrics_and_shaeffer_comparison(tn_validation_data):
    """Asserts that CV model outperforms Shaeffer's formula and transfers conformal intervals."""
    res = tn_validation_data["results"]
    perf = res["overall_performance"]

    cv_mae = perf["cv_model"]["mae_kg"]
    shaeffer_mae = perf["shaeffer_formula_baseline"]["mae_kg"]

    # CV model achieves lower MAE than physical tape measurement
    assert cv_mae < shaeffer_mae
    assert cv_mae <= 20.0  # Acceptance threshold defined in SOP protocol
    assert perf["error_reduction_vs_shaeffer_pct"] > 0.0

    # Out-of-distribution Conformal Coverage transfer check
    conf = res["conformal_uncertainty_performance"]
    assert conf["nominal_90_coverage_pct"] >= 88.0
    assert conf["nominal_95_coverage_pct"] >= 93.0

    # Breed-stratified check
    breeds = res["breed_stratified_performance"]
    assert len(breeds) == 4
    for b in breeds:
        assert b["sample_size"] in [50, 100]
        assert b["cv_mae_kg"] < 25.0
        assert b["conformal_cov90_pct"] >= 85.0


def test_b4_report_and_metrics_artifacts(tn_validation_data):
    """Asserts that the markdown report and metrics JSON exist and are fully populated."""
    rep_p = tn_validation_data["report_path"]
    met_p = tn_validation_data["metrics_path"]

    assert rep_p.exists()
    assert met_p.exists()

    content = rep_p.read_text(encoding="utf-8")
    assert "Batch B4: External On-Field Validation Report" in content
    assert "Kangayam" in content or "KANGAYAM" in content
    assert "Shaeffer's Formula" in content
