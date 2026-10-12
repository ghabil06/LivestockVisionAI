"""
Capture Metadata & Quality Schemas.
Defines image capture specifications, orientation angles, and quality evaluation results.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ViewAngle(str, Enum):
    LATERAL_LEFT = "LATERAL_LEFT"
    LATERAL_RIGHT = "LATERAL_RIGHT"
    CRANIAL_FRONT = "CRANIAL_FRONT"
    CAUDAL_REAR = "CAUDAL_REAR"
    DORSAL_TOP = "DORSAL_TOP"
    DENTAL_ANTERIOR = "DENTAL_ANTERIOR"
    UNKNOWN = "UNKNOWN"


class LightingCondition(str, Enum):
    DIRECT_SUN = "DIRECT_SUN"
    DIFFUSE_DAYLIGHT = "DIFFUSE_DAYLIGHT"
    INDOOR_SHED = "INDOOR_SHED"
    LOW_LIGHT = "LOW_LIGHT"
    HARSH_SHADOW = "HARSH_SHADOW"


class QualityStatus(str, Enum):
    PASSED = "PASSED"
    REJECTED_BLUR = "REJECTED_BLUR"
    REJECTED_UNDEREXPOSED = "REJECTED_UNDEREXPOSED"
    REJECTED_OVEREXPOSED = "REJECTED_OVEREXPOSED"
    REJECTED_OCCLUDED = "REJECTED_OCCLUDED"
    REJECTED_TOO_DISTANT = "REJECTED_TOO_DISTANT"
    REJECTED_TOO_CLOSE = "REJECTED_TOO_CLOSE"
    REJECTED_UNSUPPORTED_POSE = "REJECTED_UNSUPPORTED_POSE"
    REJECTED_UNKNOWN_SPECIES = "REJECTED_UNKNOWN_SPECIES"


class QualityCheckResult(BaseModel):
    """Result of automated image quality and suitability screening."""
    passed: bool = Field(..., description="Whether the capture passed all quality gates")
    status: QualityStatus = Field(..., description="Quality status classification")
    laplacian_variance: float = Field(..., description="Laplacian blur score (higher = sharper)")
    brightness_mean: float = Field(..., description="Mean grayscale brightness (0-255)")
    bbox_area_ratio: Optional[float] = Field(None, description="Animal bounding box area / image area")
    rejection_reason: Optional[str] = Field(None, description="Detailed explanation if rejected")
    recommendation: Optional[str] = Field(None, description="User guidance to capture an acceptable image")


class CaptureMetadata(BaseModel):
    """Metadata recorded at the moment of image acquisition."""
    capture_id: str = Field(..., description="Unique UUID or hash for this capture")
    animal_id: Optional[str] = Field(None, description="Associated animal ID if known")
    view_angle: ViewAngle = Field(default=ViewAngle.UNKNOWN)
    lighting: Optional[LightingCondition] = None
    camera_device: Optional[str] = Field(None, description="Smartphone or camera model name")
    focal_length_mm: Optional[float] = None
    estimated_distance_m: Optional[float] = None
    resolution_w: int = Field(..., ge=64)
    resolution_h: int = Field(..., ge=64)
    timestamp: str = Field(..., description="ISO 8601 acquisition timestamp")
    quality_result: Optional[QualityCheckResult] = None
