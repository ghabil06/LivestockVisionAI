"""
Sheep Age Estimator.
Implements ovine dentition eruption staging.
"""

from typing import Dict, Any


class SheepAgeEstimator:
    """Ovine age estimation via dental incisor pairs."""

    STAGE_AGE_MAP = {
        0: {"min_months": 0, "max_months": 12, "desc": "Lamb (< 1 year), milk teeth only"},
        1: {"min_months": 12, "max_months": 18, "desc": "Two-tooth (1 pair central permanent) (~1 - 1.5 years)"},
        2: {"min_months": 18, "max_months": 24, "desc": "Four-tooth (2 pairs permanent) (~1.5 - 2 years)"},
        3: {"min_months": 24, "max_months": 36, "desc": "Six-tooth (3 pairs permanent) (~2 - 3 years)"},
        4: {"min_months": 36, "max_months": 84, "desc": "Full mouth (4 pairs permanent) (~3 - 7 years)"},
    }

    def estimate_from_dentition_pairs(self, num_pairs: int) -> Dict[str, Any]:
        """Classify sheep age bracket from count of permanent incisor pairs."""
        if not (0 <= num_pairs <= 4):
            raise ValueError(f"Incisor pairs must be between 0 and 4, got {num_pairs}")

        info = self.STAGE_AGE_MAP[num_pairs]
        return {
            "species": "SHEEP",
            "method": "VETERINARY_DENTITION_EXAMINATION",
            "permanent_incisor_pairs": num_pairs,
            "estimated_age_range_months": [info["min_months"], info["max_months"]],
            "description": info["desc"],
        }

    def estimate_from_dental_image(self, image_input: Any) -> Dict[str, Any]:
        """Derive permanent incisor pairs and veterinary age bracket from oral photograph."""
        from LivestockVisionAI.models.cattle.dental_vision import DentalVisionAnalyzer
        analyzer = DentalVisionAnalyzer()
        res = analyzer.detect_incisor_pairs(image_input)

        if not res.get("is_valid_dental_view"):
            return {
                "species": "SHEEP",
                "status": "REJECTED",
                "reason": res.get("reason", "Unable to analyze sheep dental photograph."),
                "recommendation": "Capture a close-up photo of the lower front incisors with the lower lip parted.",
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
        """Refuse casual body photo age estimation."""
        return {
            "species": "SHEEP",
            "status": "REJECTED",
            "reason": (
                "Unable to estimate sheep age reliably from body photo alone. "
                "Fleece volume masks skeletal frame and maturity markers."
            ),
            "recommendation": "Inspect dental arcade or consult flock ear tag records.",
        }
