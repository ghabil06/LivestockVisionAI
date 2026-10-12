"""
Multi-Species Age Estimation Pipeline.
Enforces biological grounding:
- Ruminants (Cattle, Goats, Sheep): Requires oral/dental incisor inspection.
- Poultry: Requires hatch cohort records.
- Refuses to predict chronological age from casual full-body photographs.
"""

from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

from LivestockVisionAI.data.schemas.animal_record import Species
from LivestockVisionAI.models.cattle.age_estimator import CattleAgeEstimator
from LivestockVisionAI.models.chicken.age_estimator import ChickenAgeEstimator
from LivestockVisionAI.models.goat.age_estimator import GoatAgeEstimator
from LivestockVisionAI.models.sheep.age_estimator import SheepAgeEstimator


from pydantic import BaseModel, model_validator


class AgeEstimationResult(BaseModel):
    status: str
    species: str
    method: Optional[str] = None
    dental_stage: Optional[str] = None
    life_stage: Optional[str] = None
    estimated_age_range_months: Optional[list] = None
    estimated_age_weeks_range: Optional[list] = None
    weeks_since_hatch: Optional[float] = None
    detected_incisor_pairs: Optional[int] = None
    confidence: Optional[float] = None
    landmarks_detected: Optional[Dict[str, Any]] = None
    visual_biomarkers: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    reliability: Optional[str] = None
    rejection_reason: Optional[str] = None
    reason: Optional[str] = None
    recommendation: Optional[str] = None

    @model_validator(mode="after")
    def sync_rejection_reasons(self) -> "AgeEstimationResult":
        if self.rejection_reason and not self.reason:
            self.reason = self.rejection_reason
        elif self.reason and not self.rejection_reason:
            self.rejection_reason = self.reason
        return self


class UnifiedAgePipeline:
    """Orchestrates multi-species age estimation and domain-honesty rejections."""

    def __init__(self):
        self.cattle_age = CattleAgeEstimator()
        self.chicken_age = ChickenAgeEstimator()
        self.goat_age = GoatAgeEstimator()
        self.sheep_age = SheepAgeEstimator()

    def estimate_age(
        self,
        species: Species,
        dentition_incisor_pairs: Optional[int] = None,
        weeks_since_hatch: Optional[float] = None,
        is_oral_dental_capture: bool = False,
        image_input: Optional[Any] = None,
    ) -> AgeEstimationResult:
        """
        Evaluate age based on species, verified biomarkers, or computer vision photographic analysis.
        """
        if species == Species.UNKNOWN:
            return AgeEstimationResult(
                status="REJECTED",
                species="UNKNOWN",
                rejection_reason="Unable to estimate age: Species is UNKNOWN or unsupported.",
                recommendation="Identify animal species before requesting age estimation.",
            )

        # 1. Poultry (Hen)
        if species == Species.HEN:
            # Mode A: Explicit verified hatch date
            if weeks_since_hatch is not None:
                res = self.chicken_age.estimate_from_hatch_date(weeks_since_hatch)
                return AgeEstimationResult(
                    status="SUCCESS",
                    species="HEN",
                    method=res["method"],
                    life_stage=res["life_stage"],
                    weeks_since_hatch=res["weeks_since_hatch"],
                    reliability=res["reliability"],
                )
            # Mode B: Image provided -> Computer vision poultry morphological aging
            if image_input is not None:
                v_res = self.chicken_age.estimate_from_bird_photo(image_input)
                if v_res.get("status") == "SUCCESS":
                    return AgeEstimationResult(
                        status="SUCCESS",
                        species="HEN",
                        method=v_res["method"],
                        life_stage=v_res["life_stage"],
                        estimated_age_weeks_range=v_res.get("estimated_age_weeks_range"),
                        estimated_age_range_months=v_res.get("estimated_age_range_months"),
                        confidence=v_res.get("confidence"),
                        visual_biomarkers=v_res.get("visual_biomarkers"),
                        description=v_res.get("description"),
                        reliability=v_res.get("reliability"),
                    )
                return AgeEstimationResult(
                    status="REJECTED",
                    species="HEN",
                    rejection_reason=v_res.get("reason", "Unable to analyze poultry photograph."),
                    recommendation=v_res.get("recommendation", "Provide flock hatch date or a clearer bird photo."),
                )
            # Mode C: Casual request without hatch record or image
            res = self.chicken_age.estimate_from_casual_photo()
            return AgeEstimationResult(
                status="REJECTED",
                species="HEN",
                rejection_reason=res["reason"],
                recommendation=res["recommendation"],
            )

        # 2. Ruminants (Cattle, Goat, Sheep)
        estimator_map = {
            Species.COW: self.cattle_age,
            Species.GOAT: self.goat_age,
            Species.SHEEP: self.sheep_age,
        }
        est = estimator_map[species]

        # Mode A: Explicit incisor pair count provided
        if dentition_incisor_pairs is not None:
            res = est.estimate_from_dentition_pairs(dentition_incisor_pairs)
            return AgeEstimationResult(
                status="SUCCESS",
                species=species.value,
                method=res["method"],
                dental_stage=res.get("dental_stage") or f"PAIRS_{dentition_incisor_pairs}",
                detected_incisor_pairs=dentition_incisor_pairs,
                estimated_age_range_months=res["estimated_age_range_months"],
                description=res.get("description") or res.get("desc"),
                reliability=res.get("reliability", "HIGH_VETERINARY_GROUNDED"),
            )

        # Mode B: Image provided -> Computer vision dental incisor detection
        if image_input is not None:
            v_res = est.estimate_from_dental_image(image_input)
            if v_res.get("status") == "SUCCESS":
                return AgeEstimationResult(
                    status="SUCCESS",
                    species=species.value,
                    method=v_res["method"],
                    dental_stage=v_res.get("dental_stage"),
                    detected_incisor_pairs=v_res.get("detected_incisor_pairs"),
                    estimated_age_range_months=v_res.get("estimated_age_range_months"),
                    confidence=v_res.get("confidence"),
                    landmarks_detected=v_res.get("landmarks_detected"),
                    description=v_res.get("description"),
                    reliability=v_res.get("reliability"),
                )
            return AgeEstimationResult(
                status="REJECTED",
                species=species.value,
                rejection_reason=v_res.get("reason", "Unable to analyze dental photograph."),
                recommendation=v_res.get("recommendation", "Capture a focused close-up photo of the lower front incisors."),
            )

        # Mode C: Direct casual request without oral/dental exam or image
        res = est.estimate_from_casual_photo()
        return AgeEstimationResult(
            status="REJECTED",
            species=species.value,
            rejection_reason=res["reason"],
            recommendation=res["recommendation"],
        )
