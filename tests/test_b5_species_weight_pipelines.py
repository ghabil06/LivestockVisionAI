"""
Unit and integration tests for Batch B5:
Multi-Species Weight Estimation Pipelines, Multi-View Pooling, and Conformal Uncertainty Bounds.
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from LivestockVisionAI.models.cattle.weight_estimator import (
    CattleWeightEstimator,
    MultiViewWeightRegressionModel,
)
from LivestockVisionAI.models.chicken.weight_estimator import ChickenWeightEstimator
from LivestockVisionAI.models.goat.weight_estimator import GoatWeightEstimator
from LivestockVisionAI.models.sheep.weight_estimator import SheepWeightEstimator
from LivestockVisionAI.models.evaluate_b5_multi_species_weight import run_b5_multi_species_benchmark
from LivestockVisionAI.api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    return TestClient(app)


def create_mock_image(color=(128, 128, 128)) -> Image.Image:
    arr = (((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def create_mock_image_bytes() -> bytes:
    img = create_mock_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Cattle Multi-View Pooling Tests
# ---------------------------------------------------------------------------

def test_cattle_multiview_pooling_architecture():
    estimator = CattleWeightEstimator(target_mode="linear")
    img1 = create_mock_image()
    img2 = create_mock_image()
    img3 = create_mock_image()
    img4 = create_mock_image()

    # Feature extraction dimension test
    pooled_feat = estimator.extract_multiview_features([img1, img2, img3, img4])
    assert pooled_feat.shape == (512,)

    # Multi-view 4-perspective inference
    res_4v = estimator.predict_from_images([img1, img2, img3, img4], withers_height_cm=125.0)
    assert res_4v["species"] == "COW"
    assert res_4v["views_processed"] == 4
    assert res_4v["multi_view_mode"] == "MULTI_VIEW_4"
    assert res_4v["estimated_weight_kg"] >= 35.0
    assert "90%" in res_4v["prediction_intervals"]
    assert "95%" in res_4v["prediction_intervals"]

    # Single-view graceful degradation
    res_1v = estimator.predict_from_images([img1], withers_height_cm=125.0)
    assert res_1v["views_processed"] == 1
    assert res_1v["multi_view_mode"] == "SINGLE_VIEW"


# ---------------------------------------------------------------------------
# 2. Poultry Dual-Unit & Cohort Trajectory Tests
# ---------------------------------------------------------------------------

def test_chicken_dual_units_and_plumage():
    estimator = ChickenWeightEstimator()

    # Dual units check
    res = estimator.predict_from_raw_weight(
        raw_weight_kg=2.35,
        bird_type="BROILER",
        plumage_condition="FULL_PLUMAGE",
    )
    assert res["species"] == "HEN"
    assert res["bird_type"] == "BROILER"
    assert res["estimated_weight_kg"] < 2.35  # Plumage adjustment applied
    assert res["plumage_adjusted"] is True
    assert res["estimated_weight_g"] == round(res["estimated_weight_kg"] * 1000.0, 1)

    # Conformal interval conversion check
    int_kg = res["prediction_intervals_kg"]
    int_g = res["prediction_intervals_g"]
    assert round(int_kg["90%"]["lower"] * 1000.0, 1) == int_g["90%"]["lower"]
    assert round(int_kg["90%"]["upper"] * 1000.0, 1) == int_g["90%"]["upper"]


def test_chicken_growth_cohort_trajectories():
    estimator = ChickenWeightEstimator()

    # Broiler growth vs layer growth at 6 weeks
    w_broiler = estimator.estimate_from_growth_cohort(weeks_since_hatch=6.0, bird_type="BROILER")
    w_layer = estimator.estimate_from_growth_cohort(weeks_since_hatch=6.0, bird_type="LAYER")

    assert w_broiler > w_layer
    assert 2.0 <= w_broiler <= 3.2
    assert 0.4 <= w_layer <= 1.2


# ---------------------------------------------------------------------------
# 3. Caprine Morphometric Allometry Tests
# ---------------------------------------------------------------------------

def test_goat_morphometric_allometry():
    estimator = GoatWeightEstimator()

    # Standard formula: (65^2 * 60) / 10760 = 23.559 kg
    calc_wt = estimator.calculate_morphometric_weight(body_length_cm=60.0, heart_girth_cm=65.0)
    assert pytest.approx(calc_wt, rel=1e-2) == 23.56

    res = estimator.predict_from_morphometrics(body_length_cm=60.0, heart_girth_cm=65.0)
    assert res["species"] == "GOAT"
    assert pytest.approx(res["estimated_weight_kg"], rel=1e-2) == 23.56
    assert "morphometrics" in res
    assert "90%" in res["prediction_intervals"]


# ---------------------------------------------------------------------------
# 4. Ovine Fleece-Depth Subtraction Tests
# ---------------------------------------------------------------------------

def test_sheep_fleece_deduction_and_morphometrics():
    estimator = SheepWeightEstimator()

    # Morphometrics: (70^2 * 65) / 10500 = 30.333 kg
    calc_wt = estimator.calculate_morphometric_weight(body_length_cm=65.0, heart_girth_cm=70.0)
    assert pytest.approx(calc_wt, rel=1e-2) == 30.33

    # Heavy fleece (7.5 cm) should deduct fleece weight
    res_fleece = estimator.predict_from_morphometrics(
        body_length_cm=65.0,
        heart_girth_cm=70.0,
        fleece_depth_cm=7.5,
        is_shorn=False,
    )
    assert res_fleece["species"] == "SHEEP"
    assert res_fleece["fleece_adjusted"] is True
    assert res_fleece["deducted_fleece_kg"] > 0.0
    assert res_fleece["estimated_weight_kg"] < 30.33


# ---------------------------------------------------------------------------
# 5. Multi-Species Benchmark & Empirical Conformal Coverage Integrity
# ---------------------------------------------------------------------------

def test_b5_multi_species_benchmark_dataset_integrity():
    pred_path = PROJECT_ROOT / "data" / "manifests" / "b5_multi_species_predictions.csv"
    assert pred_path.exists(), "B5 predictions CSV must exist"

    df = pd.read_csv(pred_path)
    assert len(df) == 660, "Must evaluate all 660 animals in the master dataset"

    # All species present with proper sample counts
    counts = df["species"].value_counts().to_dict()
    assert counts["COW"] == 510
    assert counts["HEN"] == 50
    assert counts["GOAT"] == 50
    assert counts["SHEEP"] == 50

    # Leak-free empirical conformal coverage guarantees
    cov_90 = df["covered_90"].mean() * 100.0
    cov_95 = df["covered_95"].mean() * 100.0

    assert cov_90 >= 90.0, f"Empirical 90% coverage {cov_90:.1f}% fell below nominal 90.0%"
    assert cov_95 >= 95.0, f"Empirical 95% coverage {cov_95:.1f}% fell below nominal 95.0%"


# ---------------------------------------------------------------------------
# 6. Multi-Species & Multi-View API Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_multi_species_weight_routing(client):
    img_bytes = create_mock_image_bytes()

    # 1. Chicken with bird_type and plumage
    res_hen = client.post(
        "/api/v1/predict/weight",
        files={"file": ("hen.png", img_bytes, "image/png")},
        data={"species": "HEN", "bird_type": "BROILER", "plumage_condition": "FULL_PLUMAGE"},
    )
    assert res_hen.status_code == 200
    d_hen = res_hen.json()
    assert d_hen["species"] == "HEN"
    assert d_hen["estimated_weight_g"] is not None
    assert "90%" in d_hen["prediction_intervals_g"]

    # 2. Goat with morphometrics
    res_goat = client.post(
        "/api/v1/predict/weight",
        files={"file": ("goat.png", img_bytes, "image/png")},
        data={"species": "GOAT", "body_length_cm": 60.0, "heart_girth_cm": 65.0},
    )
    assert res_goat.status_code == 200
    d_goat = res_goat.json()
    assert d_goat["species"] == "GOAT"
    assert pytest.approx(d_goat["estimated_weight_kg"], rel=1e-2) == 23.56

    # 3. Sheep with fleece depth and morphometrics
    res_sheep = client.post(
        "/api/v1/predict/weight",
        files={"file": ("sheep.png", img_bytes, "image/png")},
        data={
            "species": "SHEEP",
            "body_length_cm": 65.0,
            "heart_girth_cm": 70.0,
            "fleece_depth_cm": 6.0,
            "is_shorn": False,
        },
    )
    assert res_sheep.status_code == 200
    d_sheep = res_sheep.json()
    assert d_sheep["species"] == "SHEEP"
    assert d_sheep["metadata"]["fleece_adjusted"] is True

    # 4. Cattle with multi-view upload (view_1 and view_2)
    res_cow_mv = client.post(
        "/api/v1/predict/weight",
        files={
            "file": ("view_0.png", img_bytes, "image/png"),
            "view_1": ("view_1.png", img_bytes, "image/png"),
            "view_2": ("view_2.png", img_bytes, "image/png"),
        },
        data={"species": "COW", "withers_height_cm": 125.0},
    )
    assert res_cow_mv.status_code == 200
    d_cow = res_cow_mv.json()
    assert d_cow["species"] == "COW"
    assert d_cow["metadata"]["views_processed"] == 3
    assert d_cow["metadata"]["multi_view_mode"] == "MULTI_VIEW_3"
