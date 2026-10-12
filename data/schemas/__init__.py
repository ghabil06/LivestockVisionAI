"""Data schemas package for LivestockVisionAI."""

from .animal_record import AnimalRecord, Species, Sex, HealthStatus
from .measurement_record import MeasurementRecord, MeasurementType, MeasurementUnit
from .capture_metadata import (
    CaptureMetadata,
    ViewAngle,
    LightingCondition,
    QualityStatus,
    QualityCheckResult,
)
from .tamil_nadu_validation_record import TamilNaduFieldValidationRecord, MorphometricMeasurements
from .prospective_trial_record import ProspectiveTrialRecord, TrialSite, ScaleStandard

__all__ = [
    "AnimalRecord",
    "Species",
    "Sex",
    "HealthStatus",
    "MeasurementRecord",
    "MeasurementType",
    "MeasurementUnit",
    "CaptureMetadata",
    "ViewAngle",
    "LightingCondition",
    "QualityStatus",
    "QualityCheckResult",
    "TamilNaduFieldValidationRecord",
    "MorphometricMeasurements",
    "ProspectiveTrialRecord",
    "TrialSite",
    "ScaleStandard",
]
