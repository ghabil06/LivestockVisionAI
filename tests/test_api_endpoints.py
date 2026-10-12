"""
Integration tests for FastAPI endpoints using TestClient.
Verifies HTTP routing, input validation, and rejection responses.
"""

import io
import numpy as np
import cv2
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from LivestockVisionAI.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def create_test_image_bytes(sharp: bool = True) -> bytes:
    """Create a synthetic PNG image in memory."""
    if sharp:
        # High contrast checkerboard with high Laplacian variance and midtone brightness
        arr = (((np.indices((250, 250)).sum(axis=0) // 10) % 2) * 160 + 50).astype(np.uint8)
    else:
        # Extremely blurred flat image
        arr = np.ones((250, 250), dtype=np.uint8) * 128
        arr = cv2.GaussianBlur(arr, (31, 31), 15.0)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["platform"] == "LivestockVisionAI"
    assert "COW" in data["supported_species"]


def test_quality_check_endpoint(client):
    # Sharp image
    sharp_bytes = create_test_image_bytes(sharp=True)
    res_sharp = client.post(
        "/api/v1/quality/check",
        files={"file": ("test_sharp.png", sharp_bytes, "image/png")},
    )
    assert res_sharp.status_code == 200
    assert res_sharp.json()["passed"] is True

    # Blurred image
    blur_bytes = create_test_image_bytes(sharp=False)
    res_blur = client.post(
        "/api/v1/quality/check",
        files={"file": ("test_blur.png", blur_bytes, "image/png")},
    )
    assert res_blur.status_code == 200
    data_blur = res_blur.json()
    assert data_blur["passed"] is False
    assert data_blur["status"] == "REJECTED_BLUR"


def test_predict_weight_rejects_blurry_capture(client):
    blur_bytes = create_test_image_bytes(sharp=False)
    res = client.post(
        "/api/v1/predict/weight",
        files={"file": ("blur.png", blur_bytes, "image/png")},
        data={"species": "COW"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "REJECTED"
    assert "Unable to estimate reliably" in data["rejection_reason"]


def test_predict_weight_sharp_capture(client):
    sharp_bytes = create_test_image_bytes(sharp=True)
    res = client.post(
        "/api/v1/predict/weight",
        files={"file": ("sharp.png", sharp_bytes, "image/png")},
        data={"species": "COW", "reference_weight_kg": "280.0"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["estimated_weight_kg"] == 280.0
    assert "90%" in data["prediction_intervals"]
    assert "95%" in data["prediction_intervals"]


def test_predict_age_photo_only_rejected(client):
    """Refuse to predict age from casual photo alone."""
    res = client.post(
        "/api/v1/predict/age",
        data={"species": "COW"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "REJECTED"
    reason_msg = data.get("reason") or data.get("rejection_reason") or ""
    assert "Unable to estimate age reliably from full-body 2D photograph" in reason_msg


def test_predict_age_dentition(client):
    res = client.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["dental_stage"] == "TWO_PAIRS"
    assert data["estimated_age_range_months"] == [24, 30]


def test_animal_registry_flow(client):
    import uuid
    aid = f"TEST_COW_{uuid.uuid4().hex[:6].upper()}"
    mid = f"M_{uuid.uuid4().hex[:6].upper()}"

    # Enroll
    enroll_payload = {
        "animal_id": aid,
        "species": "COW",
        "farm_id": "FARM_COIMBATORE",
        "breed": "KANGAYAM",
        "sex": "FEMALE",
        "scale_weight_kg": 310.0,
    }
    r_enroll = client.post("/api/v1/animals/", json=enroll_payload)
    assert r_enroll.status_code == 201

    # Fetch
    r_get = client.get(f"/api/v1/animals/{aid}")
    assert r_get.status_code == 200
    assert r_get.json()["animal_id"] == aid

    # Add measurement
    m_payload = {
        "measurement_id": mid,
        "animal_id": aid,
        "measurement_type": "WEIGHT_SCALE",
        "value": 312.5,
        "unit": "KG",
        "timestamp": "2026-10-12T10:00:00Z",
    }
    r_m = client.post(f"/api/v1/animals/{aid}/measurements", json=m_payload)
    assert r_m.status_code == 201

    # List measurements
    r_m_list = client.get(f"/api/v1/animals/{aid}/measurements")
    assert r_m_list.status_code == 200
    assert len(r_m_list.json()) == 1
    assert r_m_list.json()[0]["value"] == 312.5
