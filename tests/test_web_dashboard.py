"""
Unit and integration tests for Web Dashboard, SQLite persistence, and Audit Trail.
"""

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from LivestockVisionAI.api.main import app
from LivestockVisionAI.data.schemas.animal_record import AnimalRecord, Species, Sex
from LivestockVisionAI.api.database import (
    save_animal_to_db,
    get_animal_from_db,
    list_animals_from_db,
    get_recent_audit_logs,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_html_rendering(client):
    """GET / should serve the interactive web dashboard HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "LivestockVisionAI Platform" in res.text
    assert "Live Vision Appraisal" in res.text
    assert "Biological Age Staging" in res.text


def test_sqlite_persistence():
    animal = AnimalRecord(
        animal_id="PERSIST_COW_99",
        species=Species.COW,
        farm_id="FARM_SALEM",
        breed="ALAMBADI",
        sex=Sex.MALE,
        scale_weight_kg=295.0,
    )
    save_animal_to_db(animal)

    # Retrieve
    fetched = get_animal_from_db("PERSIST_COW_99")
    assert fetched is not None
    assert fetched.animal_id == "PERSIST_COW_99"
    assert fetched.breed == "ALAMBADI"
    assert fetched.scale_weight_kg == 295.0


def test_audit_logs_endpoint(client):
    # Query audit logs via API
    res = client.get("/api/v1/animals/audit/logs?limit=10")
    assert res.status_code == 200
    logs = res.json()
    assert isinstance(logs, list)

    # Trigger a prediction to populate audit log
    arr = (((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    r_pred = client.post(
        "/api/v1/predict/weight",
        files={"file": ("audit_test.png", buf.getvalue(), "image/png")},
        data={"species": "GOAT"},
    )
    assert r_pred.status_code == 200

    # Verify log entry exists
    res_after = client.get("/api/v1/animals/audit/logs?limit=5")
    assert res_after.status_code == 200
    logs_after = res_after.json()
    assert len(logs_after) > 0
    latest = logs_after[0]
    assert "timestamp" in latest
    assert "species" in latest
    assert "status" in latest
