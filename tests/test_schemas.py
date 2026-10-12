"""
Unit tests for Pydantic data schemas.
Verifies biological consistency rules, bounds, and required validations.
"""

import pytest
from pydantic import ValidationError

from LivestockVisionAI.data.schemas.animal_record import (
    AnimalRecord,
    Species,
    Sex,
    HealthStatus,
)
from LivestockVisionAI.data.schemas.measurement_record import (
    MeasurementRecord,
    MeasurementType,
    MeasurementUnit,
)
from LivestockVisionAI.data.schemas.capture_metadata import (
    CaptureMetadata,
    ViewAngle,
    QualityStatus,
    QualityCheckResult,
)


def test_valid_cow_record():
    cow = AnimalRecord(
        animal_id="blf2025",
        species=Species.COW,
        farm_id="FARM_TAMILNADU_01",
        breed="KANGAYAM",
        sex=Sex.MALE,
        scale_weight_kg=345.5,
        dentition_incisor_pairs=2,
    )
    assert cow.animal_id == "BLF2025"  # Should be normalized to uppercase
    assert cow.species == Species.COW
    assert cow.scale_weight_kg == 345.5
    assert cow.dentition_incisor_pairs == 2


def test_hen_cannot_have_teeth():
    """Hens do not have incisors; schema must reject positive incisor pairs."""
    with pytest.raises(ValidationError) as exc_info:
        AnimalRecord(
            animal_id="HEN_001",
            species=Species.HEN,
            farm_id="POULTRY_FARM_01",
            dentition_incisor_pairs=2,
        )
    assert "Poultry (HEN) cannot possess mammalian incisor pairs" in str(exc_info.value)


def test_cattle_minimum_weight_invariant():
    """Cattle cannot be enrolled with sub-viable scale weight (< 20 kg)."""
    with pytest.raises(ValidationError) as exc_info:
        AnimalRecord(
            animal_id="COW_TINY",
            species=Species.COW,
            farm_id="FARM_01",
            scale_weight_kg=12.0,
        )
    assert "below minimum viable weight for cattle" in str(exc_info.value)


def test_poultry_maximum_weight_invariant():
    """Hens cannot be enrolled with impossible mass (> 10 kg)."""
    with pytest.raises(ValidationError) as exc_info:
        AnimalRecord(
            animal_id="HEN_HUGE",
            species=Species.HEN,
            farm_id="FARM_01",
            scale_weight_kg=18.5,
        )
    assert "exceeds maximum physiological limit for poultry" in str(exc_info.value)


def test_blank_animal_id_rejected():
    with pytest.raises(ValidationError):
        AnimalRecord(
            animal_id="   ",
            species=Species.GOAT,
            farm_id="FARM_01",
        )


def test_measurement_record_validation():
    # Valid measurement
    rec = MeasurementRecord(
        measurement_id="M_101",
        animal_id="BLF2025",
        measurement_type=MeasurementType.WEIGHT_SCALE,
        value=320.0,
        unit=MeasurementUnit.KG,
        calibration_certificate_id="CAL-2026-X8",
        timestamp="2026-10-12T10:00:00Z",
    )
    assert rec.value == 320.0
    assert rec.unit == MeasurementUnit.KG

    # Negative measurement rejected
    with pytest.raises(ValidationError):
        MeasurementRecord(
            measurement_id="M_102",
            animal_id="BLF2025",
            measurement_type=MeasurementType.WEIGHT_SCALE,
            value=-5.0,
            unit=MeasurementUnit.KG,
            timestamp="2026-10-12T10:00:00Z",
        )


def test_quality_check_result_schema():
    q = QualityCheckResult(
        passed=False,
        status=QualityStatus.REJECTED_BLUR,
        laplacian_variance=45.2,
        brightness_mean=110.0,
        rejection_reason="Unable to estimate reliably: Image blur detected",
        recommendation="Hold camera steady",
    )
    assert not q.passed
    assert q.status == QualityStatus.REJECTED_BLUR
    assert "Unable to estimate reliably" in q.rejection_reason
