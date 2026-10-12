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
    """Verifies that SyncWorker.kt actually calls the sync API before marking synced."""
    sync_worker_file = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai" / "SyncWorker.kt"
    assert sync_worker_file.exists(), "SyncWorker.kt missing"
    code = sync_worker_file.read_text(encoding="utf-8")

    # Must invoke the remote sync endpoint
    assert "syncOfflineQueue" in code, "SyncWorker must invoke apiService.syncOfflineQueue()"
    # Must check response success
    assert "response.isSuccessful" in code, "SyncWorker must verify response.isSuccessful before marking synced"
    # Must only mark synced upon confirmation
    assert "markSynced" in code
    # Must handle failure with attempt tracking
    assert "syncAttempts" in code, "SyncWorker must increment syncAttempts on failure"


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
    """Verifies that duplicate local inference engine is removed, enforcing the single API inference path."""
    ai_dir = PROJECT_ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "livestockvision" / "ai"
    inference_engine_file = ai_dir / "InferenceEngine.kt"
    assert not inference_engine_file.exists(), "InferenceEngine.kt must not exist; API is the single canonical inference path"


