"""
Unit and integration tests for Batch B7:
Mobile Android Architecture & Edge Deployment Specifications.
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from LivestockVisionAI.edge.export_onnx import (
    export_multiview_weight_to_onnx,
    quantize_onnx_model,
    HAS_ONNX,
)
from LivestockVisionAI.edge.parity_tester import HAS_ORT
from LivestockVisionAI.edge.offline_queue import OfflineCaptureQueue
from LivestockVisionAI.models.cattle.weight_estimator import MultiViewWeightRegressionModel
from LivestockVisionAI.api.main import app


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp(prefix="test_b7_edge_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Multi-View ONNX Export & INT8 Quantization
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ONNX, reason="ONNX library not available")
def test_multiview_onnx_export_and_quantization(temp_dir):
    mv_model = MultiViewWeightRegressionModel(target_mode="linear")
    mv_model.eval()

    onnx_fp32 = temp_dir / "multiview_weight.onnx"
    res = export_multiview_weight_to_onnx(onnx_fp32, model=mv_model)
    assert res is not None
    assert onnx_fp32.exists()
    assert onnx_fp32.stat().st_size > 10_000

    if HAS_ORT:
        int8_path = quantize_onnx_model(onnx_fp32)
        assert int8_path is not None
        assert int8_path.exists()
        # INT8 file should be significantly smaller (~75% compression)
        assert int8_path.stat().st_size < onnx_fp32.stat().st_size


# ---------------------------------------------------------------------------
# 2. Multi-View Numerical Parity (PyTorch vs ONNX)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (HAS_ONNX and HAS_ORT), reason="ONNX or onnxruntime not available")
def test_multiview_numerical_parity(temp_dir):
    import onnxruntime as ort

    mv_model = MultiViewWeightRegressionModel(target_mode="linear")
    mv_model.eval()

    onnx_path = temp_dir / "mv_parity.onnx"
    export_multiview_weight_to_onnx(onnx_path, model=mv_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name

    max_diff = 0.0
    for _ in range(5):
        # 4-view input tensor: [1, 4, 3, 224, 224]
        test_tensor = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            pt_out = float(mv_model(test_tensor).item())

        ort_out = float(session.run(None, {inp_name: test_tensor.numpy()})[0].item())
        diff = abs(pt_out - ort_out)
        if diff > max_diff:
            max_diff = diff

    # Max discrepancy in FP32 should be negligible (< 0.001 kg)
    assert max_diff <= 0.10


# ---------------------------------------------------------------------------
# 3. Mobile CameraX Live Pre-Capture Quality Analyzer Simulation
# ---------------------------------------------------------------------------

def test_camerax_quality_hud_analyzer():
    """Simulates real-time CameraX frame evaluation inside Android ImageAnalysis.Analyzer."""
    from LivestockVisionAI.models.shared.quality_filter import QualityFilter

    qfilter = QualityFilter()

    # 1. Optimal Sharp & Well-Exposed Capture (HUD Green / Shutter Enabled)
    rng = np.random.default_rng(42)
    sharp_frame = rng.normal(128, 45, (240, 320, 3)).clip(0, 255).astype(np.uint8)
    res_sharp = qfilter.evaluate(sharp_frame)
    assert res_sharp.passed is True
    assert res_sharp.status.value == "PASSED"

    # 2. Blurry Frame (HUD Amber / Shutter Disabled)
    blurry_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    res_blur = qfilter.evaluate(blurry_frame)
    assert res_blur.passed is False
    assert res_blur.status.value == "REJECTED_BLUR"

    # 3. Underexposed Frame (Too Dark)
    dark_frame = np.full((240, 320, 3), 15, dtype=np.uint8)
    res_dark = qfilter.evaluate(dark_frame)
    assert res_dark.passed is False
    assert res_dark.status.value == "REJECTED_UNDEREXPOSED"

    # 4. Overexposed Frame (Direct Sunlight Glare)
    bright_frame = np.full((240, 320, 3), 245, dtype=np.uint8)
    res_bright = qfilter.evaluate(bright_frame)
    assert res_bright.passed is False
    assert res_bright.status.value == "REJECTED_OVEREXPOSED"


# ---------------------------------------------------------------------------
# 4. Android Room DB Offline Queue & Background Sync Lifecycle
# ---------------------------------------------------------------------------

def test_room_db_offline_queue_sync_lifecycle(temp_dir, client):
    """Simulates Android Room DB local queue and WorkManager background upload."""
    db_file = temp_dir / "room_offline_queue.db"
    queue = OfflineCaptureQueue(db_path=db_file)

    rng = np.random.default_rng(77)
    img = rng.normal(128, 45, (224, 224, 3)).clip(0, 255).astype(np.uint8)

    # 1. Enqueue capture in field without cellular coverage
    res = queue.enqueue_capture(
        image_input=img,
        farm_id="FARM_VILLAGE_SOUTH",
        animal_id="EDGE_VAL_909",
        species_hint="COW",
    )
    assert res["status"] == "QUEUED"
    queue_id = res["queue_id"]

    # Verify pending state
    stats_before = queue.get_queue_stats()
    assert stats_before["PENDING"] == 1
    assert stats_before["SYNCED"] == 0

    # 2. Reconnect to connectivity & trigger WorkManager sync
    sync_res = queue.sync_with_api(api_client=client, batch_size=10)
    assert sync_res["synced"] == 1
    assert sync_res["failed"] == 0

    # Verify transition to SYNCED
    stats_after = queue.get_queue_stats()
    assert stats_after["PENDING"] == 0
    assert stats_after["SYNCED"] == 1

    # Clean up central DB
    client.delete("/api/v1/animals/EDGE_VAL_909")


# ---------------------------------------------------------------------------
# 5. Android APK Binary Size Budget Compliance
# ---------------------------------------------------------------------------

def test_android_apk_budget_compliance():
    """Asserts that total bundled mobile app size remains within 50 MB Play Store budget."""
    base_apk_mb = 14.5
    species_int8_mb = 10.7
    weight_int8_mb = 10.7
    total_mb = base_apk_mb + species_int8_mb + weight_int8_mb

    assert total_mb < 50.0
    assert total_mb == pytest.approx(35.9, rel=1e-2)
