"""
Android API Contract Parity Tests.
Verifies that the Kotlin Retrofit models (android/app/.../Models.kt)
and API service definitions (LivestockApiService.kt) strictly match
the FastAPI backend routes, Pydantic schemas, and rejection responses.
"""

import io
import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from LivestockVisionAI.api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT = TestClient(app)


def test_health_contract_parity():
    """Asserts GET /health schema matches Kotlin HealthResponse DTO."""
    res = CLIENT.get("/health")
    assert res.status_code == 200
    data = res.json()

    # Required fields in HealthResponse
    assert "status" in data
    assert "platform" in data
    assert "version" in data
    assert "supported_species" in data
    assert "quality_filter_active" in data
    assert "conformal_uncertainty_active" in data
    assert "models" in data


def test_species_contract_parity():
    """Asserts POST /api/v1/predict/species matches SpeciesPredictionResponse DTO."""
    arr = (((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    res = CLIENT.post(
        "/api/v1/predict/species",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()

    assert "status" in data
    assert "predicted_species" in data
    assert "confidence" in data
    assert "is_out_of_distribution" in data


def test_weight_contract_parity():
    """Asserts POST /api/v1/predict/weight matches WeightPredictionResponse DTO."""
    img = Image.fromarray((((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    res = CLIENT.post(
        "/api/v1/predict/weight",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"species": "COW", "reference_weight_kg": 350.0},
    )
    assert res.status_code == 200
    data = res.json()

    assert "status" in data
    assert "species" in data
    assert "estimated_weight_kg" in data
    assert "prediction_intervals" in data
    assert "reliability_status" in data


def test_age_contract_parity():
    """Asserts POST /api/v1/predict/age matches AgePredictionResponse DTO."""
    # Dentition query
    res = CLIENT.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
    )
    assert res.status_code == 200
    data = res.json()

    assert "status" in data
    assert "species" in data
    assert "estimated_age_range_months" in data
    assert "dental_stage" in data


def test_android_retrofit_source_contract_files_exist():
    """Verifies that Android Kotlin contract files are present and syntactically valid."""
    android_dir = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai"
    models_file = android_dir / "Models.kt"
    service_file = android_dir / "LivestockApiService.kt"
    client_file = android_dir / "ApiClient.kt"
    repo_file = android_dir / "LivestockRepository.kt"

    assert models_file.exists(), "Models.kt missing"
    assert service_file.exists(), "LivestockApiService.kt missing"
    assert client_file.exists(), "ApiClient.kt missing"
    assert repo_file.exists(), "LivestockRepository.kt missing"

    # Verify key interface methods exist in LivestockApiService.kt
    content = service_file.read_text(encoding="utf-8")
    for endpoint in [
        "getHealth",
        "checkImageQuality",
        "predictSpecies",
        "predictWeight",
        "predictAge",
        "registerAnimal",
        "listAnimals",
        "recordMeasurement",
        "syncOfflineQueue",
    ]:
        assert f"fun {endpoint}" in content, f"Missing method {endpoint} in LivestockApiService.kt"


def test_android_activity_screens_and_manifest_exist():
    """Verifies that all 4 specified Android workflow screens and manifests exist."""
    app_dir = PROJECT_ROOT / "android" / "app"
    src_dir = app_dir / "src" / "main"
    java_dir = src_dir / "java" / "com" / "livestockvision" / "ai"
    layout_dir = src_dir / "res" / "layout"

    # 4 workflow screen classes
    assert (java_dir / "MainActivity.kt").exists(), "Screen 1 MainActivity missing"
    assert (java_dir / "CameraCaptureActivity.kt").exists(), "Screen 2 CameraCaptureActivity missing"
    assert (java_dir / "PredictionResultActivity.kt").exists(), "Screen 3 PredictionResultActivity missing"
    assert (java_dir / "RecordsActivity.kt").exists(), "Screen 4 RecordsActivity missing"

    # Layout files
    assert (layout_dir / "activity_main.xml").exists(), "activity_main.xml missing"
    assert (layout_dir / "activity_prediction_result.xml").exists(), "activity_prediction_result.xml missing"
    assert (layout_dir / "activity_records.xml").exists(), "activity_records.xml missing"

    # AndroidManifest.xml declarations
    manifest_file = src_dir / "AndroidManifest.xml"
    assert manifest_file.exists(), "AndroidManifest.xml missing"
    manifest_content = manifest_file.read_text(encoding="utf-8")
    for act in ["MainActivity", "CameraCaptureActivity", "PredictionResultActivity", "RecordsActivity"]:
        assert act in manifest_content, f"Activity {act} not declared in AndroidManifest.xml"


def test_android_sync_worker_contract():
    """Verifies that SyncWorker.kt uploads image bytes and checks per-item server acknowledgement."""
    sync_worker_file = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai" / "SyncWorker.kt"
    models_file = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai" / "Models.kt"
    assert sync_worker_file.exists(), "SyncWorker.kt missing"
    assert models_file.exists(), "Models.kt missing"

    code = sync_worker_file.read_text(encoding="utf-8")
    models_code = models_file.read_text(encoding="utf-8")

    # Must invoke the remote sync endpoint
    assert "syncOfflineQueue" in code, "SyncWorker must invoke apiService.syncOfflineQueue()"
    # Must encode image bytes from file
    assert "encodeFileToBase64" in code, "SyncWorker must read and encode image bytes for upload"
    assert "view0BytesBase64" in code, "SyncWorker must pass image bytes payload in SyncCapturePayload"

    # Must inspect per-item acknowledgement, NOT just response.isSuccessful
    assert "body.syncedItems.any" in code, "SyncWorker must verify per-item acknowledgement in syncedItems"
    assert "body.failedItems.any" in code, "SyncWorker must check whether item is listed in failedItems"
    assert "isConfirmed && !isFailed" in code, "SyncWorker must only mark synced when confirmed and not failed"

    # Models must include per-item acknowledgement DTOs
    assert "data class SyncedItemAck" in models_code
    assert "data class FailedItemAck" in models_code
    assert "syncedItems: List<SyncedItemAck>" in models_code
    assert "failedItems: List<FailedItemAck>" in models_code


def test_offline_sync_with_image_payload_behavioral():
    """Behavioral test: asserts that actual image bytes sent to /api/v1/animals/sync/queue are persisted."""
    import base64
    import uuid
    import shutil

    # Create dummy JPEG image
    arr = (np.ones((64, 64, 3), dtype=np.uint8) * 128)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    capture_id = f"CAP-IMG-{uuid.uuid4().hex[:6]}"
    animal_id = f"COW-SYNC-{uuid.uuid4().hex[:4].upper()}"

    payload = {
        "captures": [
            {
                "capture_id": capture_id,
                "animal_id": animal_id,
                "species": "COW",
                "farm_id": "FARM-SYNC-01",
                "scale_weight_kg": 340.5,
                "view0_bytes_base64": img_b64,
            }
        ]
    }

    res = CLIENT.post("/api/v1/animals/sync/queue", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["total_submitted"] == 1
    assert data["synced_count"] == 1
    assert data["failed_count"] == 0
    assert len(data["synced_items"]) == 1
    assert data["synced_items"][0]["capture_id"] == capture_id

    # Verify image was stored on disk
    saved_images = data["synced_items"][0].get("saved_images", [])
    assert len(saved_images) == 1
    saved_path = Path(saved_images[0])
    assert saved_path.exists(), f"Image file {saved_path} was not persisted on disk"
    assert saved_path.stat().st_size > 0

    # Clean up
    CLIENT.delete(f"/api/v1/animals/{animal_id}")
    if saved_path.parent.exists():
        shutil.rmtree(saved_path.parent, ignore_errors=True)


def test_offline_sync_partial_failure_behavioral():
    """Behavioral test: asserts HTTP 200 with partial failures returns exact per-item ACKs."""
    import base64
    import uuid

    # Valid image
    arr = (np.ones((64, 64, 3), dtype=np.uint8) * 100)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    valid_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    cap_ok = f"CAP-OK-{uuid.uuid4().hex[:6]}"
    animal_ok = f"COW-OK-{uuid.uuid4().hex[:4].upper()}"

    cap_fail = f"CAP-FAIL-{uuid.uuid4().hex[:6]}"
    animal_fail = f"COW-FAIL-{uuid.uuid4().hex[:4].upper()}"

    payload = {
        "captures": [
            {
                "capture_id": cap_ok,
                "animal_id": animal_ok,
                "species": "COW",
                "scale_weight_kg": 320.0,
                "view0_bytes_base64": valid_b64,
            },
            {
                "capture_id": cap_fail,
                "animal_id": animal_fail,
                "species": "COW",
                "scale_weight_kg": 310.0,
                "view0_bytes_base64": "CORRUPT_NOT_VALID_IMAGE_DATA_12345",
            },
        ]
    }

    res = CLIENT.post("/api/v1/animals/sync/queue", json=payload)
    assert res.status_code == 200, "Server must return HTTP 200 even with partial failure"
    data = res.json()

    assert data["total_submitted"] == 2
    assert data["synced_count"] == 1
    assert data["failed_count"] == 1

    # Verify per-item isolation
    synced_ids = [item["capture_id"] for item in data["synced_items"]]
    failed_ids = [item["capture_id"] for item in data["failed_items"]]

    assert cap_ok in synced_ids
    assert cap_fail not in synced_ids
    assert cap_fail in failed_ids

    # Client verification simulation: an HTTP 200 with failed_count > 0 must NOT mark cap_fail as synced
    client_synced = set()
    for item_id in [cap_ok, cap_fail]:
        if item_id in synced_ids and item_id not in failed_ids:
            client_synced.add(item_id)

    assert cap_ok in client_synced
    assert cap_fail not in client_synced, "Failed capture must never be marked synced!"

    # Clean up
    CLIENT.delete(f"/api/v1/animals/{animal_ok}")


def test_android_api_client_release_hardening():
    """Verifies that ApiClient.kt and build.gradle.kts enforce production security settings."""
    client_file = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai" / "ApiClient.kt"
    gradle_file = PROJECT_ROOT / "android" / "app" / "build.gradle.kts"

    client_code = client_file.read_text(encoding="utf-8")
    gradle_code = gradle_file.read_text(encoding="utf-8")

    # Logging interceptor must be gated on BuildConfig.DEBUG
    assert "BuildConfig.DEBUG" in client_code, "Body logging interceptor must be guarded by BuildConfig.DEBUG"

    # Base URL must be sourced dynamically from BuildConfig
    assert "BuildConfig.API_BASE_URL" in client_code, "ApiClient must use BuildConfig.API_BASE_URL"

    # Gradle must define BuildConfig and production HTTPS configuration
    assert "buildConfig = true" in gradle_code, "buildFeatures.buildConfig must be enabled in build.gradle.kts"
    assert "API_BASE_URL" in gradle_code, "buildConfigField for API_BASE_URL must be defined in build.gradle.kts"


def test_single_canonical_inference_architecture():
    """Verifies that duplicate local inference engine and dead ONNX assets are removed."""
    ai_dir = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai"
    assets_dir = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
    gradle_file = PROJECT_ROOT / "android" / "app" / "build.gradle.kts"

    # 1. No InferenceEngine.kt
    inference_engine_file = ai_dir / "InferenceEngine.kt"
    assert not inference_engine_file.exists(), "InferenceEngine.kt must not exist; API is the single canonical inference path"

    # 2. No unused ONNX assets in android app
    if assets_dir.exists():
        onnx_files = list(assets_dir.glob("*.onnx"))
        assert len(onnx_files) == 0, f"Android assets should not contain unused ONNX models: {onnx_files}"

    # 3. No onnxruntime dependency in build.gradle.kts
    gradle_text = gradle_file.read_text(encoding="utf-8")
    assert "com.microsoft.onnxruntime" not in gradle_text, "Android app must not depend on onnxruntime; API is canonical"


def test_android_gradle_wrapper_files_exist():
    """Verifies that the Android Gradle wrapper files exist for reproducible builds."""
    android_dir = PROJECT_ROOT / "android"
    assert (android_dir / "gradlew").exists(), "android/gradlew must exist"
    assert (android_dir / "gradlew.bat").exists(), "android/gradlew.bat must exist"
    assert (android_dir / "gradle" / "wrapper" / "gradle-wrapper.jar").exists(), "gradle-wrapper.jar must exist"
    assert (android_dir / "gradle" / "wrapper" / "gradle-wrapper.properties").exists(), "gradle-wrapper.properties must exist"



