"""
Unit and integration tests for species-specific weight estimators and prediction API.
Verifies multi-species estimation, conformal bounds, fleece correction, and physiological clipping.
"""

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from LivestockVisionAI.models.cattle.weight_estimator import CattleWeightEstimator
from LivestockVisionAI.models.chicken.weight_estimator import ChickenWeightEstimator
from LivestockVisionAI.models.goat.weight_estimator import GoatWeightEstimator
from LivestockVisionAI.models.sheep.weight_estimator import SheepWeightEstimator
from LivestockVisionAI.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def create_test_image_bytes() -> bytes:
    arr = (((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_cattle_weight_pipeline():
    estimator = CattleWeightEstimator(target_mode="linear")
    res = estimator.predict_from_features_or_value(raw_prediction_kg=260.0)

    assert res["species"] == "COW"
    assert res["estimated_weight_kg"] == 260.0
    assert "90%" in res["prediction_intervals"]
    assert "95%" in res["prediction_intervals"]
    assert res["reliability_status"] == "CALIBRATED"

    # Physiological lower bound check: must be >= 35 kg
    res_small = estimator.predict_from_features_or_value(raw_prediction_kg=15.0)
    assert res_small["estimated_weight_kg"] == 35.0


def test_chicken_weight_pipeline():
    estimator = ChickenWeightEstimator()
    res = estimator.predict_from_raw_weight(raw_weight_kg=2.4)

    assert res["species"] == "HEN"
    assert res["estimated_weight_kg"] == 2.4
    assert res["estimated_weight_g"] == 2400.0
    assert "90%" in res["prediction_intervals_kg"]
    assert "95%" in res["prediction_intervals_kg"]


def test_goat_weight_pipeline():
    estimator = GoatWeightEstimator()
    res = estimator.predict_from_raw_weight(raw_weight_kg=38.5)

    assert res["species"] == "GOAT"
    assert res["estimated_weight_kg"] == 38.5
    assert "90%" in res["prediction_intervals"]
    assert res["reliability_status"] == "CALIBRATED"


def test_sheep_fleece_adjustment():
    estimator = SheepWeightEstimator()

    # 1. Unshorn sheep with heavy fleece (6 cm) should have mass adjusted downwards
    res_fleece = estimator.predict_from_raw_weight(
        raw_weight_kg=45.0,
        fleece_depth_cm=6.0,
        is_shorn=False,
    )
    assert res_fleece["fleece_adjusted"] is True
    assert res_fleece["estimated_weight_kg"] < 45.0  # Silhouette adjusted down

    # 2. Shorn sheep should not be adjusted
    res_shorn = estimator.predict_from_raw_weight(
        raw_weight_kg=45.0,
        fleece_depth_cm=6.0,
        is_shorn=True,
    )
    assert res_shorn["fleece_adjusted"] is False
    assert res_shorn["estimated_weight_kg"] == 45.0


def test_api_predict_weight_all_species(client):
    img_bytes = create_test_image_bytes()

    for spec in ["COW", "HEN", "GOAT", "SHEEP"]:
        res = client.post(
            "/api/v1/predict/weight",
            files={"file": (f"test_{spec}.png", img_bytes, "image/png")},
            data={"species": spec},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert data["species"] == spec
        assert data["estimated_weight_kg"] > 0.0
        assert "90%" in data["prediction_intervals"]
        assert "95%" in data["prediction_intervals"]


def test_api_heavy_cattle_warning(client):
    img_bytes = create_test_image_bytes()
    # Test cow with reference weight 350 kg (>300 kg threshold)
    res = client.post(
        "/api/v1/predict/weight",
        files={"file": ("heavy_bull.png", img_bytes, "image/png")},
        data={"species": "COW", "reference_weight_kg": 350.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["species"] == "COW"
    assert data["estimated_weight_kg"] == 350.0
    assert data["warning_code"] == "CAUTION_HEAVY_ANIMAL_USE_PHYSICAL_SCALE"
    assert "exceeds 300 kg" in data["warning_message"]
    # Check adaptive interval half-width is 84.50 kg (expanded from baseline 53.88)
    assert data["prediction_intervals"]["90%"]["half_width"] == 84.50
    assert data["prediction_intervals"]["95%"]["half_width"] == 108.00
