"""
Unit and integration tests for multi-species age estimation pipelines.
Verifies dental incisor staging, poultry cohort stages, and refusal to guess from casual photos.
"""

import pytest
from fastapi.testclient import TestClient

from LivestockVisionAI.models.shared.age_pipeline import UnifiedAgePipeline
from LivestockVisionAI.data.schemas.animal_record import Species
from LivestockVisionAI.api.main import app


@pytest.fixture
def age_pipeline():
    return UnifiedAgePipeline()


@pytest.fixture
def client():
    return TestClient(app)


def test_cattle_dental_staging(age_pipeline):
    # 0 pairs (milk teeth)
    res0 = age_pipeline.estimate_age(Species.COW, dentition_incisor_pairs=0)
    assert res0.status == "SUCCESS"
    assert res0.dental_stage == "MILK_TEETH"
    assert res0.estimated_age_range_months == [0, 18]

    # 2 pairs (~2-2.5 years)
    res2 = age_pipeline.estimate_age(Species.COW, dentition_incisor_pairs=2)
    assert res2.status == "SUCCESS"
    assert res2.dental_stage == "TWO_PAIRS"
    assert res2.estimated_age_range_months == [24, 30]

    # 4 pairs (full mouth)
    res4 = age_pipeline.estimate_age(Species.COW, dentition_incisor_pairs=4)
    assert res4.status == "SUCCESS"
    assert res4.dental_stage == "FULL_MOUTH"
    assert res4.estimated_age_range_months == [36, 96]


def test_goat_and_sheep_dental_staging(age_pipeline):
    res_goat = age_pipeline.estimate_age(Species.GOAT, dentition_incisor_pairs=1)
    assert res_goat.status == "SUCCESS"
    assert res_goat.estimated_age_range_months == [12, 18]

    res_sheep = age_pipeline.estimate_age(Species.SHEEP, dentition_incisor_pairs=3)
    assert res_sheep.status == "SUCCESS"
    assert res_sheep.estimated_age_range_months == [24, 36]


def test_poultry_hatch_cohort_staging(age_pipeline):
    # Chick: 2 weeks
    res_chick = age_pipeline.estimate_age(Species.HEN, weeks_since_hatch=2.0)
    assert res_chick.status == "SUCCESS"
    assert res_chick.life_stage == "CHICK"

    # Grower: 6 weeks
    res_grower = age_pipeline.estimate_age(Species.HEN, weeks_since_hatch=6.0)
    assert res_grower.life_stage == "GROWER"

    # Point of lay: 20 weeks
    res_pol = age_pipeline.estimate_age(Species.HEN, weeks_since_hatch=20.0)
    assert res_pol.life_stage == "POINT_OF_LAY"

    # Adult layer: 35 weeks
    res_layer = age_pipeline.estimate_age(Species.HEN, weeks_since_hatch=35.0)
    assert res_layer.life_stage == "ADULT_LAYER"


def test_casual_photo_rejection(age_pipeline):
    """Calling without dental pairs or hatch records must reject."""
    for spec in [Species.COW, Species.GOAT, Species.SHEEP, Species.HEN]:
        res = age_pipeline.estimate_age(spec)
        assert res.status == "REJECTED"
        assert "Unable to estimate" in res.rejection_reason


def test_api_predict_age_endpoint(client):
    # Ruminant with dental
    res_cow = client.post("/api/v1/predict/age", data={"species": "COW", "dentition_incisor_pairs": 2})
    assert res_cow.status_code == 200
    assert res_cow.json()["status"] == "SUCCESS"
    assert res_cow.json()["estimated_age_range_months"] == [24, 30]

    # Poultry with hatch weeks
    res_hen = client.post("/api/v1/predict/age", data={"species": "HEN", "weeks_since_hatch": 14.0})
    assert res_hen.status_code == 200
    assert res_hen.json()["status"] == "SUCCESS"
    assert res_hen.json()["life_stage"] == "DEVELOPER"

    # Casual photo rejection
    res_rej = client.post("/api/v1/predict/age", data={"species": "COW"})
    assert res_rej.status_code == 200
    assert res_rej.json()["status"] == "REJECTED"
    assert "Unable to estimate" in res_rej.json()["rejection_reason"]


import io
import numpy as np
from PIL import Image


def create_synthetic_dental_image(num_permanent_pairs: int = 2) -> bytes:
    """Generate a synthetic oral cavity image with gums and enamel teeth."""
    arr = np.zeros((250, 250, 3), dtype=np.uint8)
    # Oral mucosa background: pink/red (R=210, G=75, B=90)
    arr[:, :] = [210, 75, 90]

    # Dental arcade (y: 80 to 170)
    # White / ivory teeth (R=235, G=230, B=215)
    arr[80:170, 30:220] = [235, 230, 215]

    # Add vertical interdental separators (dark reddish gum gaps)
    # 8 teeth -> 7 vertical lines
    step = (220 - 30) // 8
    for i in range(1, 8):
        x = 30 + i * step
        arr[80:170, x - 2:x + 2] = [170, 50, 60]

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_synthetic_poultry_image(stage: str = "CHICK") -> bytes:
    """Generate a synthetic poultry photograph."""
    arr = np.zeros((250, 250, 3), dtype=np.uint8)
    if stage == "CHICK":
        # Yellow downy plumage (R=235, G=215, B=50)
        arr[40:210, 40:210] = [235, 215, 50]
        # Soft noise
        noise = np.random.randint(-15, 15, (170, 170, 3), dtype=np.int16)
        arr[40:210, 40:210] = np.clip(arr[40:210, 40:210].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:  # ADULT_LAYER
        # Brown/white body plumage
        arr[80:230, 50:200] = [180, 140, 100]
        # Prominent crimson red comb on head (y: 20-75, x: 100-160)
        arr[20:75, 100:160] = [225, 25, 35]

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_vision_dental_incisor_derivation(age_pipeline):
    """Verifies that uploading a dental image derives tooth pairs and age bracket."""
    dental_bytes = create_synthetic_dental_image(num_permanent_pairs=2)
    res = age_pipeline.estimate_age(Species.COW, image_input=dental_bytes)

    assert res.status == "SUCCESS"
    assert res.method == "COMPUTER_VISION_DENTAL_INCISOR_DETECTION"
    assert res.detected_incisor_pairs is not None
    assert 0 <= res.detected_incisor_pairs <= 4
    assert res.dental_stage is not None
    assert res.estimated_age_range_months is not None
    assert res.landmarks_detected["oral_mucosa_verified"] is True


def test_vision_poultry_aging_from_photo(age_pipeline):
    """Verifies that uploading a chicken photo derives life stage and age range."""
    # Chick test
    chick_bytes = create_synthetic_poultry_image(stage="CHICK")
    res_chick = age_pipeline.estimate_age(Species.HEN, image_input=chick_bytes)
    assert res_chick.status == "SUCCESS"
    assert res_chick.life_stage == "CHICK"
    assert res_chick.estimated_age_weeks_range == [0.5, 3.0]
    assert res_chick.method == "COMPUTER_VISION_POULTRY_MORPHOLOGY_STAGING"

    # Adult layer test with crimson comb
    adult_bytes = create_synthetic_poultry_image(stage="ADULT_LAYER")
    res_adult = age_pipeline.estimate_age(Species.HEN, image_input=adult_bytes)
    assert res_adult.status == "SUCCESS"
    assert res_adult.life_stage in ["ADULT_LAYER", "POINT_OF_LAY"]
    assert res_adult.estimated_age_weeks_range[0] >= 18.0


def test_non_dental_body_photo_rejection_for_ruminant(age_pipeline):
    """Uploading a non-dental hide/pasture image for cattle age estimation must reject."""
    green_field = np.full((250, 250, 3), [40, 140, 50], dtype=np.uint8)
    res = age_pipeline.estimate_age(Species.COW, image_input=green_field)
    assert res.status == "REJECTED"
    assert "does not appear to be an anterior oral cavity" in res.rejection_reason


def test_api_age_endpoint_with_uploaded_images(client):
    """Test API endpoint with uploaded dental image and poultry photo."""
    # 1. Cow with uploaded dental photo
    dental_bytes = create_synthetic_dental_image()
    res_cow = client.post(
        "/api/v1/predict/age",
        data={"species": "COW"},
        files={"file": ("cow_dental.png", dental_bytes, "image/png")},
    )
    assert res_cow.status_code == 200
    data_cow = res_cow.json()
    assert data_cow["status"] == "SUCCESS"
    assert data_cow["method"] == "COMPUTER_VISION_DENTAL_INCISOR_DETECTION"
    assert data_cow["detected_incisor_pairs"] is not None

    # 2. Hen with uploaded bird photo
    chick_bytes = create_synthetic_poultry_image(stage="CHICK")
    res_hen = client.post(
        "/api/v1/predict/age",
        data={"species": "HEN"},
        files={"file": ("chick.png", chick_bytes, "image/png")},
    )
    assert res_hen.status_code == 200
    data_hen = res_hen.json()
    assert data_hen["status"] == "SUCCESS"
    assert data_hen["life_stage"] == "CHICK"
    assert data_hen["method"] == "COMPUTER_VISION_POULTRY_MORPHOLOGY_STAGING"
