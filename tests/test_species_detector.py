"""
Unit tests for 4-species classification and out-of-distribution detection.
Verifies species routing, confidence gating, and entropy OOD rejection.
"""

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from LivestockVisionAI.models.shared.species_detector import SpeciesDetector, SpeciesDetectionResult
from LivestockVisionAI.data.schemas.animal_record import Species
from LivestockVisionAI.api.main import app


@pytest.fixture
def detector():
    return SpeciesDetector(confidence_threshold=0.65, max_entropy_threshold=1.25)


@pytest.fixture
def client():
    return TestClient(app)


def create_test_image_bytes(sharp: bool = True) -> bytes:
    if sharp:
        arr = (((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8)
    else:
        arr = np.ones((224, 224), dtype=np.uint8) * 128
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_high_confidence_species_classification(detector):
    """High-contrast logits should pass OOD gate and return target species."""
    # Simulating definitive cattle prediction
    cow_logits = np.array([8.0, -2.0, -3.0, -1.0])
    res_cow = detector.evaluate_synthetic_logits(cow_logits)

    assert res_cow.predicted_species == Species.COW
    assert res_cow.confidence > 0.95
    assert not res_cow.is_out_of_distribution
    assert res_cow.rejection_reason is None

    # Simulating definitive hen prediction
    hen_logits = np.array([-2.0, 8.0, -1.0, -3.0])
    res_hen = detector.evaluate_synthetic_logits(hen_logits)
    assert res_hen.predicted_species == Species.HEN
    assert not res_hen.is_out_of_distribution


def test_ambiguous_species_ood_rejection(detector):
    """Uniform / ambiguous logits (e.g. 25% each) must trigger UNKNOWN and OOD rejection."""
    flat_logits = np.array([1.0, 1.0, 1.0, 1.0])
    res = detector.evaluate_synthetic_logits(flat_logits)

    assert res.predicted_species == Species.UNKNOWN
    assert res.is_out_of_distribution is True
    assert "Unable to estimate reliably" in res.rejection_reason
    assert res.confidence < 0.65


def test_api_species_endpoint(client):
    img_bytes = create_test_image_bytes(sharp=True)
    res = client.post(
        "/api/v1/predict/species",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "predicted_species" in data
    assert "probabilities" in data
    assert "confidence" in data
    assert len(data["probabilities"]) == 4


def test_api_species_endpoint_rejects_blur(client):
    blur_bytes = create_test_image_bytes(sharp=False)
    res = client.post(
        "/api/v1/predict/species",
        files={"file": ("test_blur.png", blur_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "REJECTED"
    assert "Unable to estimate reliably" in data["rejection_reason"]


def test_auto_detect_in_predict_weight(client):
    img_bytes = create_test_image_bytes(sharp=True)
    # Notice: No species specified in form data
    res = client.post(
        "/api/v1/predict/weight",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    # Should either succeed with auto-detected species or reject if ambiguous, but must be valid schema
    assert data["status"] in ("SUCCESS", "REJECTED")
    if data["status"] == "SUCCESS":
        assert data["species"] in ["COW", "HEN", "GOAT", "SHEEP"]
        assert "prediction_intervals" in data
