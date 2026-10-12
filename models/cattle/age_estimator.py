"""
Cattle Age Estimator.
Implements veterinary dental incisor eruption staging and age bracket estimation.
Explicitly rejects predicting chronological age from single casual body photographs.
"""

from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel


class DentalStage(str, Enum):
    MILK_TEETH = "MILK_TEETH"               # 0 pairs permanent incisors (< 18-24 months)
    ONE_PAIR = "ONE_PAIR"                   # 1 pair permanent incisors (18-24 months)
    TWO_PAIRS = "TWO_PAIRS"                 # 2 pairs permanent incisors (24-30 months)
    THREE_PAIRS = "THREE_PAIRS"             # 3 pairs permanent incisors (30-36 months)
    FULL_MOUTH = "FULL_MOUTH"               # 4 pairs permanent incisors (36-48+ months)
    BROKEN_MOUTH = "BROKEN_MOUTH"           # Senior / worn / missing incisors (> 8+ years)
    UNKNOWN = "UNKNOWN"


class CattleAgeEstimator:
    """Veterinary dentition-based cattle age staging."""

    STAGE_AGE_MAP = {
        DentalStage.MILK_TEETH: {"min_months": 0, "max_months": 18, "description": "Calf / young cattle (< 1.5 years)"},
        DentalStage.ONE_PAIR: {"min_months": 18, "max_months": 24, "description": "First permanent pair erupted (~1.5 - 2 years)"},
        DentalStage.TWO_PAIRS: {"min_months": 24, "max_months": 30, "description": "Second permanent pair erupted (~2 - 2.5 years)"},
        DentalStage.THREE_PAIRS: {"min_months": 30, "max_months": 36, "description": "Third permanent pair erupted (~2.5 - 3 years)"},
        DentalStage.FULL_MOUTH: {"min_months": 36, "max_months": 96, "description": "Full mouth permanent incisors (~3 - 8 years)"},
        DentalStage.BROKEN_MOUTH: {"min_months": 96, "max_months": 180, "description": "Senior cattle with worn/lost teeth (> 8 years)"},
    }

    def estimate_from_dentition_pairs(self, num_pairs: int) -> Dict[str, Any]:
        """Classify age bracket from count of permanent incisor pairs."""
        if not (0 <= num_pairs <= 4):
            raise ValueError(f"Incisor pairs must be between 0 and 4, got {num_pairs}")

        mapping = {
            0: DentalStage.MILK_TEETH,
            1: DentalStage.ONE_PAIR,
            2: DentalStage.TWO_PAIRS,
            3: DentalStage.THREE_PAIRS,
            4: DentalStage.FULL_MOUTH,
        }
        stage = mapping[num_pairs]
        bounds = self.STAGE_AGE_MAP[stage]

        return {
            "species": "COW",
            "method": "VETERINARY_DENTITION_EXAMINATION",
            "permanent_incisor_pairs": num_pairs,
            "dental_stage": stage.value,
            "estimated_age_range_months": [bounds["min_months"], bounds["max_months"]],
            "description": bounds["description"],
            "reliability": "HIGH_IF_DIRECT_DENTAL_INSPECTION",
        }

    def estimate_from_dental_image(self, image_input: Any) -> Dict[str, Any]:
        """Derive permanent incisor pairs and veterinary age bracket from oral photograph."""
        from LivestockVisionAI.models.cattle.dental_vision import DentalVisionAnalyzer
        analyzer = DentalVisionAnalyzer()
        res = analyzer.detect_incisor_pairs(image_input)

        if not res.get("is_valid_dental_view"):
            return {
                "species": "COW",
                "status": "REJECTED",
                "reason": res.get("reason", "Unable to analyze dental photograph."),
                "recommendation": "Capture a close-up photo of the lower front incisors with the lower lip gently parted.",
                "metrics": res.get("metrics"),
            }

        pairs = res["detected_incisor_pairs"]
        base_res = self.estimate_from_dentition_pairs(pairs)
        base_res["status"] = "SUCCESS"
        base_res["method"] = "COMPUTER_VISION_DENTAL_INCISOR_DETECTION"
        base_res["detected_incisor_pairs"] = pairs
        base_res["confidence"] = res["confidence"]
        base_res["landmarks_detected"] = res.get("landmarks_detected")
        base_res["reliability"] = "COMPUTER_VISION_DENTAL_ESTIMATED"
        return base_res

    def estimate_from_casual_photo(self) -> Dict[str, Any]:
        """
        Enforce product safety requirement:
        Refuse to predict precise age from an ordinary casual body photograph.
        """
        return {
            "species": "COW",
            "status": "REJECTED",
            "reason": (
                "Unable to estimate age reliably from full-body 2D photograph. "
                "Cattle age cannot be accurately determined without anterior dental examination "
                "or verified birth registration records."
            ),
            "recommendation": (
                "Please capture a focused close-up photo of the animal's lower front incisors "
                "or provide the ear-tag birth record."
            ),
        }
