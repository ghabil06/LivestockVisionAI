"""
Animal Record Data Schema.
Defines canonical data structures for multi-species animal registry and biological ground truth.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class Species(str, Enum):
    COW = "COW"
    HEN = "HEN"
    GOAT = "GOAT"
    SHEEP = "SHEEP"
    UNKNOWN = "UNKNOWN"


class Sex(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    CASTRATED = "CASTRATED"
    UNKNOWN = "UNKNOWN"


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    SICK = "SICK"
    PREGNANT = "PREGNANT"
    INJURED = "INJURED"
    UNKNOWN = "UNKNOWN"


class AnimalRecord(BaseModel):
    """Canonical record representing an individual animal in the system."""
    animal_id: str = Field(..., description="Unique animal identification tag or SKU", min_length=2)
    species: Species = Field(..., description="Species classification")
    farm_id: str = Field(..., description="Identifier of the farm, facility or research station")
    breed: Optional[str] = Field(None, description="Breed name or crossbreed label")
    sex: Sex = Field(default=Sex.UNKNOWN, description="Biological sex")
    health_status: HealthStatus = Field(default=HealthStatus.HEALTHY, description="Current health status")
    
    # Ground Truth Biological Attributes
    verified_dob: Optional[str] = Field(None, description="Verified date of birth (YYYY-MM-DD)")
    verified_age_months: Optional[float] = Field(None, ge=0.0, description="Chronological age in months")
    dentition_incisor_pairs: Optional[int] = Field(
        None, ge=0, le=4, description="Number of permanent incisor pairs erupted (0 to 4)"
    )
    
    # Scale Ground Truth (if recorded at enrollment)
    scale_weight_kg: Optional[float] = Field(
        None, ge=0.05, le=1500.0, description="Certified ground-truth scale weight in kg"
    )
    scale_calibration_id: Optional[str] = Field(None, description="Calibration certificate ID of the weighing scale")
    scale_measurement_timestamp: Optional[str] = Field(None, description="ISO8601 timestamp of scale measurement")

    notes: Optional[str] = Field(None, description="Field notes or veterinary observations")

    @field_validator("animal_id")
    @classmethod
    def validate_animal_id_clean(cls, v: str) -> str:
        clean = v.strip().upper()
        if not clean:
            raise ValueError("animal_id cannot be blank")
        return clean

    @model_validator(mode="after")
    def validate_biological_consistency(self) -> "AnimalRecord":
        # Hens do not possess teeth
        if self.species == Species.HEN and self.dentition_incisor_pairs is not None:
            if self.dentition_incisor_pairs > 0:
                raise ValueError("Poultry (HEN) cannot possess mammalian incisor pairs")

        # Species-specific weight bounds validation
        if self.scale_weight_kg is not None:
            if self.species == Species.HEN and self.scale_weight_kg > 10.0:
                raise ValueError(f"Weight {self.scale_weight_kg} kg exceeds maximum physiological limit for poultry")
            if self.species in (Species.GOAT, Species.SHEEP) and self.scale_weight_kg > 200.0:
                raise ValueError(f"Weight {self.scale_weight_kg} kg exceeds physiological limit for small ruminants")
            if self.species == Species.COW and self.scale_weight_kg < 20.0:
                raise ValueError(f"Weight {self.scale_weight_kg} kg is below minimum viable weight for cattle")

        return self
