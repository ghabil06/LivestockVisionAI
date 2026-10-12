"""
Prospective Field Trial Record Schema.
Defines canonical data structures for multi-site prospective validation trials,
certified scale ground truth, blinding protocols, and quality screening.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator

from LivestockVisionAI.data.schemas.animal_record import Species


class TrialSite(str, Enum):
    TN_01_COIMBATORE = "TN-01-COIMBATORE"
    TN_02_MADURAI = "TN-02-MADURAI"
    TN_03_ERODE = "TN-03-ERODE"
    TN_04_SALEM = "TN-04-SALEM"


class ScaleStandard(str, Enum):
    LEGAL_METROLOGY_CLASS_III = "LEGAL_METROLOGY_CLASS_III"
    ISO_IEC_17025_CERTIFIED = "ISO_IEC_17025_CERTIFIED"
    RESEARCH_PRECISION_BENCH = "RESEARCH_PRECISION_BENCH"


class ProspectiveTrialRecord(BaseModel):
    """Canonical record representing a single animal evaluation in a prospective trial."""
    trial_record_id: str = Field(..., description="Unique prospective trial record ID (e.g. TRIAL-TN-00001)")
    site_id: TrialSite = Field(..., description="Participating research station or farm cluster")
    species: Species = Field(..., description="Target livestock species")
    animal_id: str = Field(..., description="Individual animal ear-tag or identification code")
    breed: str = Field(..., description="Breed or genetic line classification")

    # Ground Truth Scale Measurement
    ground_truth_weight_kg: float = Field(..., gt=0.05, lt=1500.0, description="Certified scale weight in kg")
    scale_model: str = Field(..., description="Make and model of digital platform scale")
    scale_calibration_cert_id: str = Field(..., description="Calibration certificate identifier")
    scale_standard: ScaleStandard = Field(default=ScaleStandard.LEGAL_METROLOGY_CLASS_III)
    daily_calibration_check_passed: bool = Field(True, description="Whether 50kg standard weight check passed prior to session")
    operator_blinded: bool = Field(True, description="Whether scale operator was blinded to AI predictions")

    # Pre-Inference Quality Gate
    initial_capture_passed: bool = Field(True, description="Whether primary camera capture passed quality screening")
    rejection_reason: Optional[str] = Field(None, description="Quality rejection reason if initial attempt failed")
    retake_successful: bool = Field(True, description="Whether operator successfully retook capture after guidance")

    # AI Model Prediction & Uncertainty Bounds
    predicted_weight_kg: float = Field(..., gt=0.05, description="Model point estimate in kg")
    predicted_weight_g: Optional[float] = Field(None, description="Point estimate in grams (poultry)")
    conf_90_lower: float = Field(..., ge=0.0, description="90% conformal lower bound")
    conf_90_upper: float = Field(..., ge=0.0, description="90% conformal upper bound")
    conf_95_lower: float = Field(..., ge=0.0, description="95% conformal lower bound")
    conf_95_upper: float = Field(..., ge=0.0, description="95% conformal upper bound")

    # Biological Age Ground Truth
    dentition_incisor_pairs: Optional[int] = Field(None, ge=0, le=4, description="Verified incisor pairs (ruminants)")
    hatch_weeks_verified: Optional[float] = Field(None, ge=0.0, description="Verified hatch weeks (poultry)")

    @field_validator("trial_record_id")
    @classmethod
    def validate_trial_id(cls, v: str) -> str:
        clean = v.strip().upper()
        if not clean.startswith("TRIAL-"):
            raise ValueError(f"trial_record_id must begin with 'TRIAL-', got '{v}'")
        return clean

    @model_validator(mode="after")
    def validate_conformal_bounds_order(self) -> "ProspectiveTrialRecord":
        if self.conf_90_lower > self.conf_90_upper:
            raise ValueError(f"90% lower bound ({self.conf_90_lower}) exceeds upper bound ({self.conf_90_upper})")
        if self.conf_95_lower > self.conf_95_upper:
            raise ValueError(f"95% lower bound ({self.conf_95_lower}) exceeds upper bound ({self.conf_95_upper})")
        if self.conf_95_lower > self.conf_90_lower or self.conf_95_upper < self.conf_90_upper:
            raise ValueError("95% conformal interval must contain or equal the 90% conformal interval")
        return self
