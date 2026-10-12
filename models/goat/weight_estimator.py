"""
Caprine / Goat Weight Estimator.
Estimates goat body weight with morphometric allometry (heart girth & body length),
small-ruminant conformal bounds, and physiological bounds.
"""

from typing import Dict, Any, Optional
import numpy as np
from LivestockVisionAI.models.shared.uncertainty_engine import ConformalUncertaintyEngine


class GoatWeightEstimator:
    """
    Estimates goat body weight with morphometric allometrics and conformal bounds.
    Implements caprine metric Shaeffer allometry: Weight (kg) = (HG^2 * BL) / 10760
    """

    def __init__(self, uncertainty_engine: Optional[ConformalUncertaintyEngine] = None):
        self.engine = uncertainty_engine or ConformalUncertaintyEngine(default_species="GOAT")
        self.min_physio_kg = 4.0    # 4 kg (newborn kid)
        self.max_physio_kg = 140.0  # 140 kg (heavy mature buck, Boer/Jamnapari)
        self.allometric_constant = 10760.0  # Metric caprine divisor (cm^3 to kg)

    def calculate_morphometric_weight(
        self,
        body_length_cm: float,
        heart_girth_cm: float,
    ) -> float:
        """
        Calculate body weight in kg using caprine metric allometric equation:
        Weight (kg) = (Heart Girth [cm]^2 * Body Length [cm]) / 10760
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
    ) -> Dict[str, Any]:
        """Estimate weight from taped or vision-derived morphometric measurements."""
        calculated_kg = self.calculate_morphometric_weight(body_length_cm, heart_girth_cm)
        res = self.predict_from_raw_weight(raw_weight_kg=calculated_kg)
        res["morphometrics"] = {
            "body_length_cm": round(float(body_length_cm), 1),
            "heart_girth_cm": round(float(heart_girth_cm), 1),
            "allometric_formula": "Weight = (HG^2 * BL) / 10760",
        }
        return res

    def predict_from_raw_weight(self, raw_weight_kg: float) -> Dict[str, Any]:
        """Wrap caprine prediction with conformal uncertainty bounds."""
        est_kg = max(self.min_physio_kg, min(self.max_physio_kg, float(raw_weight_kg)))
        res = self.engine.compute_intervals(
            point_prediction=est_kg,
            species="GOAT",
            min_physio_bound=self.min_physio_kg,
        )

        return {
            "species": "GOAT",
            "estimated_weight_kg": res["point_estimate"],
            "unit": "KG",
            "prediction_intervals": res["intervals"],
            "reliability_status": res["reliability_status"],
            "is_out_of_distribution": res["is_out_of_distribution"],
        }

    def predict_from_image(
        self,
        image_input: Any,
        body_length_cm: Optional[float] = None,
        heart_girth_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict goat weight directly from camera image using contour morphometrics
        and Shaeffer caprine allometry.
        """
        import io
        from pathlib import Path
        from PIL import Image
        import cv2

        if body_length_cm is not None and heart_girth_cm is not None:
            res = self.predict_from_morphometrics(body_length_cm, heart_girth_cm)
            res["status"] = "SUCCESS"
            return res

        # Vision extraction of caprine dimensions
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
                "species": "GOAT",
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
                "species": "GOAT",
                "rejection_reason": "Unable to estimate reliably: Caprine contours ambiguous or blurry.",
                "is_out_of_distribution": True,
            }

        x, y, w, h = cv2.boundingRect(nz)
        est_bl_cm = float(np.clip(w * 0.45, 45.0, 110.0))
        est_hg_cm = float(np.clip(h * 0.52, 50.0, 105.0))

        res = self.predict_from_morphometrics(est_bl_cm, est_hg_cm)
        res["status"] = "SUCCESS"
        res["vision_derived_morphometrics"] = True
        return res

