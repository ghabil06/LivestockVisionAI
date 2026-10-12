"""
Tamil Nadu Indigenous Cattle Field Validation Record Schema.
Standard Operating Procedure (SOP) schema for external on-field validation cohorts.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator


class TamilNaduBreed(str, Enum):
    KANGAYAM = "KANGAYAM"
    BARGUR = "BARGUR"
    UMBLACHERY = "UMBLACHERY"
    PULIKULAM = "PULIKULAM"
    ALAMBADI = "ALAMBADI"


class AgroClimaticZone(str, Enum):
    WESTERN_ZONE = "WESTERN_ZONE"          # Kangayam (Tiruppur, Erode, Karur)
    HILLY_FOREST_ZONE = "HILLY_FOREST_ZONE" # Bargur (Anthiyur, Erode hills)
    CAUVERY_DELTA_ZONE = "CAUVERY_DELTA_ZONE" # Umblachery (Korukkai, Thiruvarur, Nagapattinam)
    SOUTHERN_ZONE = "SOUTHERN_ZONE"        # Pulikulam (Sivagangai, Madurai)
    NORTH_WESTERN_ZONE = "NORTH_WESTERN_ZONE" # Alambadi (Dharmapuri, Salem)


class MorphometricMeasurements(BaseModel):
    """Veterinary physical tape measurements for allometric analysis."""
    withers_height_cm: float = Field(..., ge=60.0, le=180.0, description="Withers height at highest point of shoulder in cm")
    heart_girth_cm: float = Field(..., ge=80.0, le=250.0, description="Circumference directly behind shoulder blades in cm")
    body_length_cm: float = Field(..., ge=70.0, le=220.0, description="Point of shoulder to pin bone distance in cm")

    @property
    def heart_girth_inches(self) -> float:
        return round(self.heart_girth_cm / 2.54, 2)

    @property
    def body_length_inches(self) -> float:
        return round(self.body_length_cm / 2.54, 2)

    @property
    def withers_height_inches(self) -> float:
        return round(self.withers_height_cm / 2.54, 2)

    def calculate_shaeffer_weight_kg(self) -> float:
        """
        Calculates traditional veterinary Shaeffer's formula weight:
        Weight (lbs) = (Girth_in^2 * Length_in) / 300
        Converts lbs to kg (* 0.45359237).
        """
        g_in = self.heart_girth_inches
        l_in = self.body_length_inches
        weight_lbs = (g_in ** 2 * l_in) / 300.0
        weight_kg = weight_lbs * 0.45359237
        return round(weight_kg, 2)


class CertifiedScaleCalibration(BaseModel):
    """Metadata verifying physical platform scale calibration."""
    scale_serial_number: str = Field(..., min_length=3)
    scale_capacity_kg: float = Field(default=1000.0, ge=500.0)
    scale_resolution_kg: float = Field(default=0.5, le=1.0)
    zero_calibration_verified: bool = Field(default=True)
    calibration_certificate_id: str = Field(..., min_length=3)


class TamilNaduFieldValidationRecord(BaseModel):
    """
    Gold-standard external validation record for Tamil Nadu native cattle.
    Enforces strict protocol compliance for allometric vision validation.
    """
    animal_id: str = Field(..., min_length=3, description="Unique ear-tag identifier e.g. TN-KANG-0101")
    breed: TamilNaduBreed = Field(..., description="Certified indigenous Tamil Nadu cattle breed")
    zone: AgroClimaticZone = Field(..., description="Agro-climatic collection zone")
    research_station_or_farm: str = Field(..., min_length=3, description="TANUVAS research station or district livestock farm")

    # Certified Physical Scale Ground Truth
    ground_truth_scale_weight_kg: float = Field(..., ge=35.0, le=1200.0, description="Physical digital scale reading in kg")
    scale_calibration: CertifiedScaleCalibration = Field(..., description="Scale certification & calibration metadata")
    morning_fasted_state: bool = Field(True, description="Pre-feeding morning weighing to minimize gut-fill variance")

    # Auxiliary Morphometrics
    morphometrics: MorphometricMeasurements = Field(..., description="Rigid cane height and heart-girth measurements")
    dentition_incisor_pairs: int = Field(..., ge=0, le=4, description="Veterinary anterior dental incisor pairs (0 to 4)")

    # Standardized 4-View Capture Protocol
    view_0_lateral_left_path: str = Field(..., min_length=1)
    view_1_lateral_right_path: str = Field(..., min_length=1)
    view_2_frontal_path: str = Field(..., min_length=1)
    view_3_caudal_rear_path: str = Field(..., min_length=1)

    camera_distance_meters: float = Field(default=3.2, ge=2.8, le=4.0, description="Standard 3.0m - 3.5m shooting distance")
    fiducial_scale_bar_present: bool = Field(True, description="1.0m rigid fiducial ground stick present in frame")
    square_stance_verified: bool = Field(True, description="Animal stood square with 4 hooves flat and weight distributed")

    @model_validator(mode="after")
    def verify_protocol_integrity(self) -> "TamilNaduFieldValidationRecord":
        if not self.scale_calibration.zero_calibration_verified:
            raise ValueError(f"Scale zero-calibration was not verified for animal {self.animal_id}.")
        if not self.morning_fasted_state:
            raise ValueError(f"Animal {self.animal_id} was not measured in pre-feeding morning fasted state (gut fill variance risk).")
        if not self.fiducial_scale_bar_present:
            raise ValueError(f"Fiducial metric scale reference missing from capture plane for animal {self.animal_id}.")
        return self
