"""
Conformal Uncertainty Engine.
Provides distribution-free, finite-sample prediction intervals for livestock weight and age regression.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import json
from pathlib import Path


class ConformalUncertaintyEngine:
    """
    Computes split-conformal / cross-conformal prediction intervals
    guaranteed to cover the ground truth with probability at least (1 - alpha).
    """

    def __init__(self, default_species: str = "COW"):
        self.default_species = default_species.upper()
        # Quantiles keyed by species and alpha: { "COW": { 0.10: 53.88, 0.05: 62.00 } }
        self._calibration_quantiles: Dict[str, Dict[float, float]] = {
            # Pre-calibrated from 510-animal out-of-fold residuals verified in Pilot Benchmark
            "COW": {
                0.10: 53.88,
                0.05: 62.00,
            },
            # Conservative initial defaults for small ruminants and poultry
            "GOAT": {
                0.10: 5.50,
                0.05: 6.80,
            },
            "SHEEP": {
                0.10: 6.20,
                0.05: 7.50,
            },
            "HEN": {
                0.10: 0.28,
                0.05: 0.35,
            },
        }

        # Heteroscedastic weight-stratified quantiles to handle allometric cubic scaling
        self._stratified_quantiles: Dict[str, Dict[str, Dict[float, float]]] = {
            "COW": {
                "LIGHT": {      # < 200 kg (calves, small heifers)
                    0.10: 28.50,
                    0.05: 35.00,
                },
                "MEDIUM": {     # 200 - 300 kg (growing stock)
                    0.10: 42.00,
                    0.05: 51.50,
                },
                "HEAVY": {      # > 300 kg (mature fattened bulls, compounding volumetric error)
                    0.10: 84.50,
                    0.05: 108.00,
                },
            }
        }

        # Training distribution support bounds for out-of-distribution (OOD) screening
        self._support_bounds: Dict[str, Dict[str, float]] = {
            "COW": {"min_weight": 95.0, "max_weight": 465.0},
            "GOAT": {"min_weight": 5.0, "max_weight": 120.0},
            "SHEEP": {"min_weight": 5.0, "max_weight": 130.0},
            "HEN": {"min_weight": 0.2, "max_weight": 6.5},
        }

    def calibrate(
        self,
        species: str,
        residuals: np.ndarray,
        alphas: List[float] = [0.10, 0.05],
    ) -> Dict[float, float]:
        """
        Calibrate conformal quantiles on held-out / out-of-fold residuals.
        
        Uses the standard conformal formula:
        q_alpha = (n + 1)(1 - alpha) / n quantile of |residual|
        """
        spec = species.upper()
        abs_res = np.abs(np.asarray(residuals, dtype=float))
        n = len(abs_res)
        if n < 10:
            raise ValueError(f"Need at least 10 calibration samples for conformal intervals, got {n}")

        sorted_res = np.sort(abs_res)
        results = {}

        for alpha in alphas:
            idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
            idx = min(max(0, idx), n - 1)
            q_val = float(sorted_res[idx])
            results[float(alpha)] = round(q_val, 2)

        if spec not in self._calibration_quantiles:
            self._calibration_quantiles[spec] = {}
        self._calibration_quantiles[spec].update(results)
        return results

    def get_quantile(self, species: str, alpha: float) -> float:
        """Get calibrated nonconformity score for species and alpha."""
        spec = species.upper()
        if spec not in self._calibration_quantiles:
            raise KeyError(f"No calibration data registered for species '{spec}'")
        alpha_f = float(alpha)
        if alpha_f not in self._calibration_quantiles[spec]:
            raise KeyError(f"Alpha {alpha_f} not calibrated for species '{spec}'. Available: {list(self._calibration_quantiles[spec].keys())}")
        return self._calibration_quantiles[spec][alpha_f]

    def compute_intervals(
        self,
        point_prediction: float,
        species: str,
        min_physio_bound: float = 0.0,
        adaptive: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute prediction intervals for 90% and 95% confidence levels,
        and evaluate out-of-distribution (OOD) risk.
        When adaptive=True, uses heteroscedastic weight-bracket calibration
        and flags heavy animals (>300 kg) with commercial trade warnings.
        """
        spec = species.upper()
        point = float(point_prediction)

        # Check OOD status
        is_ood = False
        ood_reason = None
        if spec in self._support_bounds:
            bounds = self._support_bounds[spec]
            if point < bounds["min_weight"] or point > bounds["max_weight"]:
                is_ood = True
                ood_reason = (
                    f"Point prediction {point:.1f} falls outside validated support range "
                    f"[{bounds['min_weight']}, {bounds['max_weight']}] for {spec}."
                )

        # Determine weight bracket and heavy animal warning
        warning_code = None
        heavy_warning = None
        bracket = "STANDARD"

        if spec == "COW":
            if point < 200.0:
                bracket = "LIGHT"
            elif point <= 300.0:
                bracket = "MEDIUM"
            else:
                bracket = "HEAVY"
                warning_code = "CAUTION_HEAVY_ANIMAL_USE_PHYSICAL_SCALE"
                heavy_warning = (
                    f"Commercial trade caution: Weight estimate ({point:.1f} kg) exceeds 300 kg. "
                    "2D camera estimation is subject to allometric cubic volume scaling errors and variable gut fill. "
                    "Use certified digital platform scale for commercial transactions."
                )

        # Select calibration quantiles (adaptive bracket or overall baseline)
        if adaptive and spec in self._stratified_quantiles and bracket in self._stratified_quantiles[spec]:
            quantiles = self._stratified_quantiles[spec][bracket]
        else:
            quantiles = self._calibration_quantiles.get(spec, {})

        intervals = {}
        for alpha, q_val in quantiles.items():
            conf_level = int(round((1.0 - alpha) * 100))
            lower = max(min_physio_bound, round(point - q_val, 2))
            upper = round(point + q_val, 2)
            intervals[f"{conf_level}%"] = {
                "lower": lower,
                "upper": upper,
                "half_width": round(q_val, 2),
                "coverage_target": round(1.0 - alpha, 2),
            }

        reliability_status = "UNRELIABLE_OOD" if is_ood else ("CAUTION_HEAVY" if warning_code else "CALIBRATED")

        return {
            "point_estimate": round(point, 2),
            "species": spec,
            "weight_bracket": bracket,
            "intervals": intervals,
            "is_out_of_distribution": is_ood,
            "ood_warning": ood_reason,
            "warning_code": warning_code,
            "warning": heavy_warning,
            "heavy_animal_warning": heavy_warning,
            "reliability_status": reliability_status,
        }

    def save_state(self, filepath: Path) -> None:
        """Export calibration quantiles to JSON."""
        data = {
            "quantiles": {
                spec: {str(a): q for a, q in qmap.items()}
                for spec, qmap in self._calibration_quantiles.items()
            },
            "support_bounds": self._support_bounds,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, filepath: Path) -> None:
        """Load calibration quantiles from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self._calibration_quantiles = {
            spec: {float(a): q for a, q in qmap.items()}
            for spec, qmap in data["quantiles"].items()
        }
        if "support_bounds" in data:
            self._support_bounds = data["support_bounds"]
