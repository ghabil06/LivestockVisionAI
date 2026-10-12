"""
Measurement Record Data Schema.
Defines ground-truth sensor and physical measurement data entries.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class MeasurementType(str, Enum):
    WEIGHT_SCALE = "WEIGHT_SCALE"
    WITHERS_HEIGHT = "WITHERS_HEIGHT"
    HEART_GIRTH = "HEART_GIRTH"
    BODY_LENGTH = "BODY_LENGTH"
    CHEST_CIRCUMFERENCE = "CHEST_CIRCUMFERENCE"
    FLEECE_DEPTH = "FLEECE_DEPTH"
    DENTAL_EXAM = "DENTAL_EXAM"


class MeasurementUnit(str, Enum):
    KG = "KG"
    G = "G"
    CM = "CM"
    INCH = "INCH"
    COUNT = "COUNT"


class MeasurementRecord(BaseModel):
    """Certified measurement observation for an animal."""
    measurement_id: str = Field(..., description="Unique ID for this measurement event")
    animal_id: str = Field(..., description="Target animal ID")
    measurement_type: MeasurementType
    value: float = Field(..., description="Numeric measurement value")
    unit: MeasurementUnit
    
    # Traceability & Verification
    device_model: Optional[str] = Field(None, description="Physical device or scale make/model")
    calibration_certificate_id: Optional[str] = Field(None, description="Calibration certificate code")
    operator_name: Optional[str] = Field(None, description="Technician / veterinarian name")
    timestamp: str = Field(..., description="ISO 8601 measurement timestamp")
    notes: Optional[str] = Field(None, description="Operational notes")

    @field_validator("value")
    @classmethod
    def validate_positive_value(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("Measurement value must be strictly positive")
        return v
