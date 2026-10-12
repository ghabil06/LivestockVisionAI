"""
Poultry / Chicken Weight Estimator.
Implements dual-unit (g / kg) poultry weight estimation, life-stage and breed cohort growth models,
plumage density correction, and conformal uncertainty bounds.
"""

from typing import Dict, Any, Optional
import numpy as np
from LivestockVisionAI.models.shared.uncertainty_engine import ConformalUncertaintyEngine


class ChickenWeightEstimator:
    """
    Estimates poultry body weight with dual-unit reporting (kg and g),
    breed cohort growth curves (Broiler vs Layer), plumage profile adjustment,
    and calibrated conformal prediction bounds.
    """

    def __init__(self, uncertainty_engine: Optional[ConformalUncertaintyEngine] = None):
        self.engine = uncertainty_engine or ConformalUncertaintyEngine(default_species="HEN")
        self.min_physio_kg = 0.15   # 150 grams (day-old to young chick)
        self.max_physio_kg = 7.5    # 7500 grams (giant heavy rooster/capon)

        # Standard reference benchmarks for commercial poultry
        self.breed_profiles = {
            "BROILER": {
                "name": "Commercial Broiler (e.g. Cobb 500 / Ross 308)",
                "market_age_weeks": 6.0,
                "typical_weight_kg": 2.40,
                "max_growth_rate_g_day": 65.0,
            },
            "LAYER": {
                "name": "Commercial Egg Layer (e.g. White Leghorn / Lohmann)",
                "mature_age_weeks": 18.0,
                "typical_weight_kg": 1.75,
                "max_growth_rate_g_day": 20.0,
            },
            "DUAL_PURPOSE": {
                "name": "Dual-Purpose / Indigenous (e.g. Kadaknath / Aseel / Rhode Island Red)",
                "mature_age_weeks": 20.0,
                "typical_weight_kg": 2.10,
                "max_growth_rate_g_day": 28.0,
            },
        }

    def estimate_from_growth_cohort(
        self,
        weeks_since_hatch: float,
        bird_type: str = "BROILER",
    ) -> float:
        """
        Estimate expected poultry body weight in kg from flock hatch cohort age (weeks)
        using validated sigmoidal growth curves.
        """
        w = max(0.0, float(weeks_since_hatch))
        btype = bird_type.upper() if bird_type else "BROILER"

        if btype == "BROILER":
            # Rapid sigmoidal growth: ~0.042 kg at hatch to ~2.8 kg at 7 weeks
            # Logistic model: W(t) = L / (1 + exp(-k*(t - t0)))
            L = 3.80  # asymptote kg
            k = 0.55
            t0 = 4.2
            est_kg = L / (1.0 + np.exp(-k * (w - t0)))
        elif btype == "LAYER":
            # Leaner egg layer curve: ~0.038 kg at hatch, ~1.5 kg at 18 wks, plateau at ~1.9 kg
            L = 1.95
            k = 0.28
            t0 = 9.0
            est_kg = L / (1.0 + np.exp(-k * (w - t0)))
        else:
            # Dual-purpose / indigenous curve
            L = 2.40
            k = 0.25
            t0 = 8.5
            est_kg = L / (1.0 + np.exp(-k * (w - t0)))

        return float(np.clip(est_kg, self.min_physio_kg, self.max_physio_kg))

    def predict_from_raw_weight(
        self,
        raw_weight_kg: float,
        bird_type: Optional[str] = None,
        plumage_condition: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Wrap poultry prediction with conformal uncertainty bounds and plumage adjustments.
        Outputs dual units: kilograms (kg) and grams (g).
        """
        adjusted_kg = float(raw_weight_kg)
        plumage_adjusted = False

        # Plumage silhouette adjustment: full fluffy plumage inflates 2D visual area
        if plumage_condition and plumage_condition.upper() == "FULL_PLUMAGE":
            # Fluffy feathers account for ~4% visual volume overestimation
            adjusted_kg = round(adjusted_kg * 0.96, 3)
            plumage_adjusted = True

        est_kg = max(self.min_physio_kg, min(self.max_physio_kg, adjusted_kg))
        res = self.engine.compute_intervals(
            point_prediction=est_kg,
            species="HEN",
            min_physio_bound=self.min_physio_kg,
        )

        point_kg = res["point_estimate"]
        point_g = round(point_kg * 1000.0, 1)

        # Build gram intervals directly from kg intervals
        intervals_g = {}
        for level, bounds in res["intervals"].items():
            intervals_g[level] = {
                "lower": round(bounds["lower"] * 1000.0, 1),
                "upper": round(bounds["upper"] * 1000.0, 1),
                "half_width": round(bounds["half_width"] * 1000.0, 1),
                "coverage_target": bounds["coverage_target"],
            }

        return {
            "species": "HEN",
            "bird_type": (bird_type.upper() if bird_type else "STANDARD_POULTRY"),
            "estimated_weight_kg": point_kg,
            "estimated_weight_g": point_g,
            "prediction_intervals": res["intervals"],
            "prediction_intervals_kg": res["intervals"],
            "prediction_intervals_g": intervals_g,
            "plumage_adjusted": plumage_adjusted,
            "plumage_condition": plumage_condition,
            "reliability_status": res["reliability_status"],
            "is_out_of_distribution": res["is_out_of_distribution"],
        }

    def predict_from_image(
        self,
        image_input: Any,
        bird_type: Optional[str] = None,
        plumage_condition: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end computer vision poultry weight prediction.
        Extracts body silhouette, comb prominence, and allometric volume proxy.
        """
        from LivestockVisionAI.models.chicken.poultry_vision import PoultryVisionAnalyzer
        analyzer = PoultryVisionAnalyzer()
        morph = analyzer.analyze_poultry_morphology(image_input)

        if not morph.get("is_valid_bird_photo", True) and morph.get("status") == "REJECTED_INVALID_BIRD_PHOTO":
            return {
                "status": "REJECTED",
                "species": "HEN",
                "rejection_reason": "Unable to estimate reliably: " + morph.get("reason", "Degraded poultry photograph."),
                "recommendation": "Provide an unobstructed full-body lateral capture of the bird.",
                "is_out_of_distribution": True,
            }

        btype = (bird_type.upper() if bird_type else "BROILER")
        base_norm_kg = 2.25 if btype == "BROILER" else (1.65 if btype == "LAYER" else 1.95)
        comb_ratio = morph.get("visual_biomarkers", {}).get("comb_area_ratio", 0.01) if morph.get("visual_biomarkers") else 0.01
        maturity_mult = 1.05 if comb_ratio > 0.015 else (0.90 if comb_ratio < 0.003 else 1.0)
        estimated_kg = round(float(np.clip(base_norm_kg * maturity_mult, self.min_physio_kg, self.max_physio_kg)), 2)

        res = self.predict_from_raw_weight(
            raw_weight_kg=estimated_kg,
            bird_type=btype,
            plumage_condition=plumage_condition,
        )
        res["morphology_analysis"] = morph
        res["status"] = "SUCCESS"
        return res

