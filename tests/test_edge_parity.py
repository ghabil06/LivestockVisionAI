"""
Unit & Integration Tests for Batch B7: Android & Edge Runtime.
Validates ONNX model export, dynamic INT8 quantization, numerical parity,
and offline SQLite capture queue synchronization.
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from LivestockVisionAI.edge.export_onnx import (
    export_species_detector_to_onnx,
    export_cattle_weight_to_onnx,
    quantize_onnx_model,
    export_all_edge_models,
    HAS_ONNX,
)
from LivestockVisionAI.edge.parity_tester import (
    verify_weight_parity,
    verify_species_parity,
    benchmark_edge_latency,
    HAS_ORT,
)
from LivestockVisionAI.edge.offline_queue import OfflineCaptureQueue
from LivestockVisionAI.api.main import app


@pytest.fixture
def temp_export_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="edge_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_queue_db():
    temp_dir = Path(tempfile.mkdtemp(prefix="queue_test_"))
    db_file = temp_dir / "test_offline.db"
    yield db_file
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not HAS_ONNX, reason="ONNX library is not installed in environment")
def test_onnx_model_export(temp_export_dir):
    """Verifies that species and weight models export cleanly to valid ONNX graphs."""
    species_path = temp_export_dir / "species_test.onnx"
    res_species = export_species_detector_to_onnx(species_path)
    assert res_species is not None
    assert species_path.exists()
    assert species_path.stat().st_size > 10_000

    weight_path = temp_export_dir / "weight_test.onnx"
    res_weight = export_cattle_weight_to_onnx(weight_path)
    assert res_weight is not None
    assert weight_path.exists()
    assert weight_path.stat().st_size > 10_000


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="ONNX or onnxruntime not installed in environment")
def test_int8_quantization(temp_export_dir):
    """Verifies that dynamic INT8 quantization produces valid quantized model files."""
    weight_path = temp_export_dir / "weight_model.onnx"
    export_cattle_weight_to_onnx(weight_path)

    int8_path = quantize_onnx_model(weight_path)
    assert int8_path is not None
    assert int8_path.exists()
    assert int8_path.stat().st_size > 5_000


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="ONNX or onnxruntime not installed in environment")
def test_numerical_parity_cattle_weight(temp_export_dir):
    """Asserts that ONNX runtime produces predictions within 0.10 kg of PyTorch model."""
    import torchvision.models as models

    pt_model = models.resnet18(weights=None)
    pt_model.fc = nn.Linear(pt_model.fc.in_features, 1)
    pt_model.eval()

    onnx_path = temp_export_dir / "cattle_weight.onnx"
    export_cattle_weight_to_onnx(onnx_path, model=pt_model)

    parity_result = verify_weight_parity(
        pytorch_model=pt_model,
        onnx_model_path=onnx_path,
        tolerance_kg=0.10,
        num_samples=5,
    )

    assert parity_result["status"] == "PASSED"
    assert parity_result["passed"] is True
    # Max discrepancy in FP32 should be tiny (typically < 1e-4 kg)
    assert parity_result["max_discrepancy_kg"] <= 0.10


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="ONNX or onnxruntime not installed in environment")
def test_numerical_parity_species_classifier(temp_export_dir):
    """Asserts that ONNX runtime matches PyTorch species classification probabilities."""
    import torchvision.models as models

    pt_model = models.resnet18(weights=None)
    pt_model.fc = nn.Linear(pt_model.fc.in_features, 4)
    pt_model.eval()

    onnx_path = temp_export_dir / "species_clf.onnx"
    export_species_detector_to_onnx(onnx_path, model=pt_model)

    parity_result = verify_species_parity(
        pytorch_model=pt_model,
        onnx_model_path=onnx_path,
        tolerance_prob=0.005,
        num_samples=5,
    )

    assert parity_result["status"] == "PASSED"
    assert parity_result["passed"] is True
    assert parity_result["class_match_rate"] == 1.0


@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="ONNX or onnxruntime not installed in environment")
def test_edge_latency_benchmark(temp_export_dir):
    """Verifies that edge latency benchmarking runs and reports execution time."""
    weight_path = temp_export_dir / "weight_model.onnx"
    export_cattle_weight_to_onnx(weight_path)
    int8_path = quantize_onnx_model(weight_path)

    bench = benchmark_edge_latency(weight_path, int8_path, num_runs=5, warmup=2)
    assert "fp32_latency_ms" in bench
    assert bench["fp32_latency_ms"] > 0
    assert bench["num_runs"] == 5


def test_offline_queue_enqueue_and_quality_rejection(temp_queue_db):
    """Tests that offline queue accepts high-quality images and rejects blurry captures."""
    queue = OfflineCaptureQueue(db_path=temp_queue_db)

    # Sharp image
    rng = np.random.default_rng(42)
    sharp_img = (rng.normal(128, 50, (300, 300, 3))).clip(0, 255).astype(np.uint8)

    res_sharp = queue.enqueue_capture(
        image_input=sharp_img,
        farm_id="TN-FARM-01",
        animal_id="COW-TN-101",
        species_hint="COW",
    )
    assert res_sharp["status"] == "QUEUED"
    assert res_sharp["quality_code"] == "PASSED"
    assert res_sharp["local_prediction"]["weight_kg"] is not None

    # Blurry image
    from PIL import ImageFilter
    from PIL import Image
    pil_sharp = Image.fromarray(sharp_img)
    blurred_img = pil_sharp.filter(ImageFilter.GaussianBlur(radius=15))

    res_blur = queue.enqueue_capture(
        image_input=blurred_img,
        farm_id="TN-FARM-01",
        animal_id="COW-TN-102",
        species_hint="COW",
    )
    assert res_blur["status"] == "REJECTED"
    assert res_blur["quality_code"] == "REJECTED_BLUR"
    assert res_blur["local_prediction"] is None


def test_offline_queue_sync_lifecycle(temp_queue_db):
    """Tests synchronization of pending offline captures with central FastAPI backend."""
    queue = OfflineCaptureQueue(db_path=temp_queue_db)
    client = TestClient(app)

    # Enqueue two sharp captures
    rng = np.random.default_rng(101)
    img = (rng.normal(128, 50, (300, 300, 3))).clip(0, 255).astype(np.uint8)

    import uuid
    id_1 = f"SYNC-COW-{uuid.uuid4().hex[:6]}"
    id_2 = f"SYNC-COW-{uuid.uuid4().hex[:6]}"
    queue.enqueue_capture(img, farm_id="SYNC-FARM", animal_id=id_1)
    queue.enqueue_capture(img, farm_id="SYNC-FARM", animal_id=id_2)

    pending = queue.get_pending_sync()
    assert len(pending) == 2

    # Sync with central API
    sync_result = queue.sync_with_api(api_client=client, batch_size=10)
    assert sync_result["synced"] == 2
    assert sync_result["failed"] == 0
    assert sync_result["remaining_pending"] == 0

    stats = queue.get_queue_stats()
    assert stats["SYNCED"] == 2
    assert stats["PENDING"] == 0


def test_offline_queue_retry_and_failure(temp_queue_db):
    """Tests retry count increments and transition to FAILED status upon repeated errors."""
    queue = OfflineCaptureQueue(db_path=temp_queue_db)
    rng = np.random.default_rng(202)
    img = (rng.normal(128, 50, (300, 300, 3))).clip(0, 255).astype(np.uint8)

    res = queue.enqueue_capture(img, farm_id="ERR-FARM", animal_id="ERR-COW")
    qid = res["queue_id"]

    # Mark failed 2 times (should stay PENDING)
    queue.mark_failed(qid, "Network timeout 1", max_retries=3)
    pending = queue.get_pending_sync()
    assert len(pending) == 1

    queue.mark_failed(qid, "Network timeout 2", max_retries=3)
    pending = queue.get_pending_sync()
    assert len(pending) == 1

    # 3rd failure reaches max_retries=3 -> transition to FAILED
    queue.mark_failed(qid, "Network timeout 3", max_retries=3)
    pending = queue.get_pending_sync()
    assert len(pending) == 0

    stats = queue.get_queue_stats()
    assert stats["FAILED"] == 1
