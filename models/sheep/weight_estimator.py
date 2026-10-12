"""
Ovine / Sheep Weight Estimator.
Estimates sheep body weight with fleece-depth subtraction, ovine allometrics,
and conformal uncertainty bounds.
"""

from typing import Dict, Any, Optional
import numpy as np
from LivestockVisionAI.models.shared.uncertainty_engine import ConformalUncertaintyEngine


class SheepWeightEstimator:
    """
    Estimates sheep body weight with fleece-depth adjustment,
    morphometric allometrics, and conformal prediction bounds.
    """

    def __init__(self, uncertainty_engine: Optional[ConformalUncertaintyEngine] = None):
        self.engine = uncertainty_engine or ConformalUncertaintyEngine(default_species="SHEEP")
        self.min_physio_kg = 4.0
        self.max_physio_kg = 160.0
        self.allometric_constant = 10500.0  # Metric ovine divisor (cm^3 to kg)

    def calculate_morphometric_weight(
        self,
        body_length_cm: float,
        heart_girth_cm: float,
    ) -> float:
        """
        Calculate raw body weight from morphometrics before fleece adjustment:
        Weight (kg) = (Heart Girth [cm]^2 * Body Length [cm]) / 10500
        """
        hg = float(heart_girth_cm)
        bl = float(body_length_cm)
        if hg <= 0 or bl <= 0:
            raise ValueError(f"Heart girth and body length must be positive, got HG={hg}, BL={bl}")

        raw_kg = (hg ** 2 * bl) / self.allometric_constant
        return float(np.clip(raw_kg, self.min_physio_kg, self.max_physio_kg))

    def predict_from_morphometrics(
        self,
        body_length_cm: float,
        heart_girth_cm: float,
        fleece_depth_cm: Optional[float] = None,
        is_shorn: bool = False,
    ) -> Dict[str, Any]:
        """Estimate weight from taped or vision-derived morphometrics with fleece adjustment."""
        raw_kg = self.calculate_morphometric_weight(body_length_cm, heart_girth_cm)
        res = self.predict_from_raw_weight(
            raw_weight_kg=raw_kg,
            fleece_depth_cm=fleece_depth_cm,
            is_shorn=is_shorn,
        )
        res["morphometrics"] = {
            "body_length_cm": round(float(body_length_cm), 1),
            "heart_girth_cm": round(float(heart_girth_cm), 1),
            "allometric_formula": "Weight = (HG^2 * BL) / 10500",
        }
        return res

    def predict_from_raw_weight(
        self,
        raw_weight_kg: float,
        fleece_depth_cm: Optional[float] = None,
        is_shorn: bool = False,
    ) -> Dict[str, Any]:
        """
        Wrap sheep prediction with fleece correction and conformal bounds.
        
        If unshorn with heavy fleece (> 4 cm), body silhouette overestimates mass by ~3-7%.
        """
        adjusted_weight = float(raw_weight_kg)
        fleece_adjusted = False
        deducted_fleece_kg = 0.0

        if not is_shorn and fleece_depth_cm is not None and fleece_depth_cm > 4.0:
            # Empirical adjustment factor for heavy wool volume bias
            correction_factor = max(0.90, 1.0 - (fleece_depth_cm - 4.0) * 0.015)
            adjusted_weight = round(adjusted_weight * correction_factor, 2)
            deducted_fleece_kg = round(float(raw_weight_kg) - adjusted_weight, 2)
            fleece_adjusted = True

        est_kg = max(self.min_physio_kg, min(self.max_physio_kg, adjusted_weight))
        res = self.engine.compute_intervals(
            point_prediction=est_kg,
            species="SHEEP",
            min_physio_bound=self.min_physio_kg,
        )

        return {
            "species": "SHEEP",
            "estimated_weight_kg": res["point_estimate"],
            "unit": "KG",
            "fleece_adjusted": fleece_adjusted,
            "fleece_depth_cm": fleece_depth_cm,
            "is_shorn": is_shorn,
            "deducted_fleece_kg": deducted_fleece_kg,
            "prediction_intervals": res["intervals"],
            "reliability_status": res["reliability_status"],
            "is_out_of_distribution": res["is_out_of_distribution"],
        }

    def predict_from_image(
        self,
        image_input: Any,
        body_length_cm: Optional[float] = None,
        heart_girth_cm: Optional[float] = None,
        fleece_depth_cm: Optional[float] = None,
        is_shorn: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict sheep weight directly from camera image with fleece-depth subtraction.
        """
        import io
        from pathlib import Path
        from PIL import Image
        import cv2

        if body_length_cm is not None and heart_girth_cm is not None:
            res = self.predict_from_morphometrics(body_length_cm, heart_girth_cm, fleece_depth_cm, is_shorn)
            res["status"] = "SUCCESS"
            return res

        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        elif isinstance(image_input, Image.Image):
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        if img is None:
            return {
                "status": "REJECTED",
                "species": "SHEEP",
                "rejection_reason": "Unable to estimate reliably: Image is corrupt or unreadable.",
                "is_out_of_distribution": True,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        nz = cv2.findNonZero(edges)

        if nz is None or len(nz) < 100:
            return {
                "status": "REJECTED",
                "species": "SHEEP",
                "rejection_reason": "Unable to estimate reliably: Ovine contours ambiguous or degraded.",
                "is_out_of_distribution": True,
            }

        x, y, w, h = cv2.boundingRect(nz)
        est_bl_cm = float(np.clip(w * 0.44, 48.0, 115.0))
        est_hg_cm = float(np.clip(h * 0.53, 52.0, 110.0))

        edge_density = float(len(nz)) / float(w * h)
        detected_fleece_cm = fleece_depth_cm if fleece_depth_cm is not None else float(np.clip(edge_density * 35.0, 1.5, 9.0))

        res = self.predict_from_morphometrics(
            body_length_cm=est_bl_cm,
            heart_girth_cm=est_hg_cm,
            fleece_depth_cm=detected_fleece_cm,
            is_shorn=is_shorn,
        )
        res["status"] = "SUCCESS"
        res["vision_derived_morphometrics"] = True
        return res

