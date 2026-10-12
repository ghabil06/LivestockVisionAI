"""
Unit and integration tests for Batch B6:
API & Web Dashboard Platform (Animal Registry CRUD, Offline Queue Sync, and UI Features).
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from LivestockVisionAI.api.main import app
from LivestockVisionAI.data.schemas.animal_record import Species, Sex, HealthStatus
from LivestockVisionAI.api.database import (
    save_animal_to_db,
    get_animal_from_db,
    update_animal_in_db,
    delete_animal_from_db,
    list_animals_from_db,
    sync_offline_batch_to_db,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_animal_registry_crud_lifecycle(client):
    test_id = f"TEST_B6_{uuid.uuid4().hex[:6].upper()}"

    # 1. Create Animal
    create_payload = {
        "animal_id": test_id,
        "species": "COW",
        "farm_id": "FARM_COIMBATORE_01",
        "breed": "KANGAYAM",
        "sex": "FEMALE",
        "health_status": "HEALTHY",
        "scale_weight_kg": 285.0,
        "dentition_incisor_pairs": 2,
    }
    r_create = client.post("/api/v1/animals/", json=create_payload)
    assert r_create.status_code == 201
    d_create = r_create.json()
    assert d_create["animal_id"] == test_id
    assert d_create["breed"] == "KANGAYAM"

    # Duplicate rejection (409 Conflict)
    r_dup = client.post("/api/v1/animals/", json=create_payload)
    assert r_dup.status_code == 409

    # 2. Read Animal
    r_get = client.get(f"/api/v1/animals/{test_id}")
    assert r_get.status_code == 200
    assert r_get.json()["animal_id"] == test_id
    assert r_get.json()["scale_weight_kg"] == 285.0

    # 3. Update Animal
    update_payload = dict(create_payload)
    update_payload["scale_weight_kg"] = 292.5
    update_payload["notes"] = "Healthy post-calving recovery"

    r_put = client.put(f"/api/v1/animals/{test_id}", json=update_payload)
    assert r_put.status_code == 200
    assert r_put.json()["scale_weight_kg"] == 292.5
    assert r_put.json()["notes"] == "Healthy post-calving recovery"

    # Verify updated in DB
    updated_animal = get_animal_from_db(test_id)
    assert updated_animal.scale_weight_kg == 292.5

    # 4. Delete Animal
    r_del = client.delete(f"/api/v1/animals/{test_id}")
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "DELETED"

    # Verify 404 after deletion
    r_get_del = client.get(f"/api/v1/animals/{test_id}")
    assert r_get_del.status_code == 404


def test_animal_registry_filtering(client):
    id_cow = f"FLT_COW_{uuid.uuid4().hex[:4].upper()}"
    id_goat = f"FLT_GOAT_{uuid.uuid4().hex[:4].upper()}"

    client.post("/api/v1/animals/", json={
        "animal_id": id_cow,
        "species": "COW",
        "farm_id": "FARM_FILTER_A",
        "breed": "BARGUR",
        "scale_weight_kg": 250.0,
    })
    client.post("/api/v1/animals/", json={
        "animal_id": id_goat,
        "species": "GOAT",
        "farm_id": "FARM_FILTER_B",
        "breed": "TELLICHERRY",
        "scale_weight_kg": 35.0,
    })

    # Filter by species
    r_spec = client.get("/api/v1/animals/?species=COW")
    assert r_spec.status_code == 200
    ids_spec = [a["animal_id"] for a in r_spec.json()]
    assert id_cow in ids_spec
    assert id_goat not in ids_spec

    # Filter by farm_id
    r_farm = client.get("/api/v1/animals/?farm_id=FARM_FILTER_B")
    assert r_farm.status_code == 200
    ids_farm = [a["animal_id"] for a in r_farm.json()]
    assert id_goat in ids_farm
    assert id_cow not in ids_farm

    # Clean up
    client.delete(f"/api/v1/animals/{id_cow}")
    client.delete(f"/api/v1/animals/{id_goat}")


def test_measurements_recording_and_retrieval(client):
    animal_id = f"MEAS_COW_{uuid.uuid4().hex[:4].upper()}"
    client.post("/api/v1/animals/", json={
        "animal_id": animal_id,
        "species": "COW",
        "farm_id": "FARM_MEAS_TEST",
        "scale_weight_kg": 260.0,
    })

    # Record certified measurement
    meas_id = f"M_TEST_{uuid.uuid4().hex[:6]}"
    meas_payload = {
        "measurement_id": meas_id,
        "animal_id": animal_id,
        "measurement_type": "WEIGHT_SCALE",
        "value": 264.2,
        "unit": "KG",
        "device_model": "Avery Weigh-Tronix Digital Scale",
        "calibration_certificate_id": "CAL-CERT-2026-99",
        "operator_name": "Senior Field Technician",
        "timestamp": "2026-10-12T10:00:00Z",
        "notes": "Post-grazing certified weight check",
    }
    r_m = client.post(f"/api/v1/animals/{animal_id}/measurements", json=meas_payload)
    assert r_m.status_code == 201
    assert r_m.json()["measurement_id"] == meas_id

    # Retrieve measurements
    r_list = client.get(f"/api/v1/animals/{animal_id}/measurements")
    assert r_list.status_code == 200
    items = r_list.json()
    assert len(items) >= 1
    assert items[0]["measurement_id"] == meas_id
    assert items[0]["value"] == 264.2

    # Clean up
    client.delete(f"/api/v1/animals/{animal_id}")


def test_offline_queue_batch_sync_api(client):
    q_id_1 = f"queue-b6-{uuid.uuid4().hex[:6]}"
    q_id_2 = f"queue-b6-{uuid.uuid4().hex[:6]}"
    animal_1 = f"SYNC_COW_{uuid.uuid4().hex[:4].upper()}"
    animal_2 = f"SYNC_HEN_{uuid.uuid4().hex[:4].upper()}"

    batch_payload = {
        "captures": [
            {
                "queue_id": q_id_1,
                "animal_id": animal_1,
                "species_hint": "COW",
                "farm_id": "FARM_REMOTE_VILLAGE_03",
                "local_weight_est_kg": 288.0,
                "operator_name": "Field Technician Alpha",
            },
            {
                "queue_id": q_id_2,
                "animal_id": animal_2,
                "species_hint": "HEN",
                "farm_id": "FARM_POULTRY_CLUSTER_07",
                "local_weight_est_kg": 2.25,
                "operator_name": "Poultry Extension Officer",
            },
        ]
    }

    # Post batch to sync endpoint
    r_sync = client.post("/api/v1/animals/sync/queue", json=batch_payload)
    assert r_sync.status_code == 200
    data = r_sync.json()
    assert data["total_submitted"] == 2
    assert data["synced_count"] == 2
    assert data["failed_count"] == 0

    # Verify both animals now exist in central registry
    r_a1 = client.get(f"/api/v1/animals/{animal_1}")
    assert r_a1.status_code == 200
    assert r_a1.json()["scale_weight_kg"] == 288.0

    r_a2 = client.get(f"/api/v1/animals/{animal_2}")
    assert r_a2.status_code == 200
    assert r_a2.json()["scale_weight_kg"] == 2.25

    # Clean up
    client.delete(f"/api/v1/animals/{animal_1}")
    client.delete(f"/api/v1/animals/{animal_2}")


def test_web_dashboard_ui_elements(client):
    """Verify HTML serves all Batch B6 production components."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Core tab titles
    assert "Live Vision Appraisal" in html
    assert "Biological Age Staging" in html
    assert "Animal Registry" in html
    assert "Offline Sync Monitor" in html
    assert "Audit Logs" in html

    # Multi-view inputs
    assert "fileView0" in html
    assert "fileView1" in html
    assert "fileView2" in html
    assert "fileView3" in html

    # Species options
    assert "birdTypeSelect" in html
    assert "fleeceDepthInput" in html
    assert "bodyLengthInput" in html
    assert "heartGirthInput" in html

    # Conformal UI elements
    assert "conformal-bar-container" in html
    assert "conformal-bar-fill" in html
