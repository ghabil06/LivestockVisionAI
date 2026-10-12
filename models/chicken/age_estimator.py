"""
Poultry Age Estimator.
Implements flock cohort / hatch-date life stage staging.
Poultry has no teeth; dentition estimation is strictly rejected.
"""

from typing import Dict, Any
from enum import Enum


class PoultryLifeStage(str, Enum):
    CHICK = "CHICK"               # 0 - 3 weeks (brooding)
    GROWER = "GROWER"             # 4 - 8 weeks
    DEVELOPER = "DEVELOPER"       # 9 - 18 weeks (pullet / cockerel)
    POINT_OF_LAY = "POINT_OF_LAY" # 18 - 22 weeks
    ADULT_LAYER = "ADULT_LAYER"   # 22+ weeks
    UNKNOWN = "UNKNOWN"


class ChickenAgeEstimator:
    """Poultry life-stage estimator based on verified hatch cohort records."""

    def estimate_from_hatch_date(self, weeks_since_hatch: float) -> Dict[str, Any]:
        """Classify life stage from verified hatch-date record."""
        w = float(weeks_since_hatch)
        if w < 0:
            raise ValueError("Hatch age cannot be negative")

        if w < 4:
            stage = PoultryLifeStage.CHICK
        elif w < 9:
            stage = PoultryLifeStage.GROWER
        elif w < 18:
            stage = PoultryLifeStage.DEVELOPER
        elif w <= 22:
            stage = PoultryLifeStage.POINT_OF_LAY
        else:
            stage = PoultryLifeStage.ADULT_LAYER

        return {
            "species": "HEN",
            "method": "VERIFIED_HATCH_COHORT",
            "weeks_since_hatch": w,
            "life_stage": stage.value,
            "reliability": "HIGH_IF_HATCH_RECORDED",
        }

    def estimate_from_bird_photo(self, image_input: Any) -> Dict[str, Any]:
        """Estimate life stage and age range from bird photograph using computer vision."""
        from LivestockVisionAI.models.chicken.poultry_vision import PoultryVisionAnalyzer
        analyzer = PoultryVisionAnalyzer()
        res = analyzer.analyze_poultry_morphology(image_input)
        if res.get("status") == "SUCCESS":
            return {
                "species": "HEN",
                "status": "SUCCESS",
                "method": res["method"],
                "life_stage": res["life_stage"],
                "estimated_age_weeks_range": res["estimated_age_weeks_range"],
                "estimated_age_range_months": res["estimated_age_range_months"],
                "confidence": res["confidence"],
                "description": res["description"],
                "visual_biomarkers": res.get("visual_biomarkers"),
                "reliability": "COMPUTER_VISION_MORPHOLOGY_ESTIMATED",
            }
        return {
            "species": "HEN",
            "status": "REJECTED",
            "reason": res.get("reason", "Unable to analyze poultry photograph."),
            "recommendation": "Provide a clear, unobstructed photograph of the bird or provide flock hatch records.",
        }

    def estimate_from_casual_photo(self) -> Dict[str, Any]:
        """Refuse to predict exact age from an uncalibrated chicken photo."""
        return {
            "species": "HEN",
            "status": "REJECTED",
            "reason": (
                "Unable to estimate exact poultry age reliably from photo alone. "
                "Individual bird feathering varies with nutrition, molt, and ambient temperature."
            ),
            "recommendation": "Provide flock hatch date or batch placement manifest.",
        }
