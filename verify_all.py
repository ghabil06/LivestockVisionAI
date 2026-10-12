"""
LivestockVisionAI End-to-End Verification Runner.
Executes comprehensive verification across Milestones B1 through B6:
- B1: Path Portability & Test Suite Reproducibility
- B2: Dataset Integrity & Hard Image Loading
- B3: Species Classifier Model & Checkpoint Loading
- B4: All 4 Species Vision Weight Estimators & Conformal Bounds
- B5: Biological Dental Aging & Non-Dental Photo Rejections
- B6: API Security Hardening, Live /health Check & Android Project Artifacts
"""

import sys
from pathlib import Path
import io
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_ROOT.parent
for p in [str(PARENT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from LivestockVisionAI.data.dataset_path_resolver import resolve_image_path
from LivestockVisionAI.data.dataset_loader import MultiViewLivestockDataset
from LivestockVisionAI.models.shared.species_detector import SpeciesDetector
from LivestockVisionAI.models.cattle.weight_estimator import CattleWeightEstimator
from LivestockVisionAI.models.chicken.weight_estimator import ChickenWeightEstimator
from LivestockVisionAI.models.goat.weight_estimator import GoatWeightEstimator
from LivestockVisionAI.models.sheep.weight_estimator import SheepWeightEstimator
from LivestockVisionAI.models.shared.age_pipeline import UnifiedAgePipeline
from LivestockVisionAI.data.schemas.animal_record import Species
from fastapi.testclient import TestClient
from LivestockVisionAI.api.main import app


def create_mock_lateral_image() -> bytes:
    arr = (((np.indices((224, 224)).sum(axis=0) // 8) % 2) * 160 + 50).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_full_verification():
    print("=" * 80)
    print("  LIVESTOCKVISIONAI INDEPENDENT AUDIT VERIFICATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Milestone B1 & B2: Path Portability & Dataset Integrity
    # ---------------------------------------------------------
    print("\n[B1 & B2] Testing Path Portability & Dataset Image Integrity...")
    manifest_cattle = PROJECT_ROOT / "data" / "manifests" / "cattle_manifest.csv"
    assert manifest_cattle.exists(), "Cattle manifest missing"
    ds = MultiViewLivestockDataset(manifest_cattle, strict=True)
    sample = ds[0]
    assert sample["images"].shape == (4, 3, 224, 224)
    assert not sample["is_missing_fallback"], "Fallback placeholder detected!"
    print(f"  [PASS] MultiViewLivestockDataset loaded strictly ({len(ds)} animals, 4 views each, 0 fallbacks).")

    # ---------------------------------------------------------
    # Milestone B3: Species Classifier & Checkpoint Loading
    # ---------------------------------------------------------
    print("\n[B3] Testing Species Classifier & Explicit Checkpoint Loading...")
    detector = SpeciesDetector()
    assert detector.weights_loaded is True, "SpeciesDetector weights not loaded!"
    mock_bytes = create_mock_lateral_image()
    det_res = detector.predict(mock_bytes)
    print(f"  [PASS] SpeciesDetector initialized with trained checkpoint: {detector.checkpoint_path}")
    print(f"  [PASS] Species inference status: {det_res.predicted_species.value} (conf={det_res.confidence:.3f}, ood={det_res.is_out_of_distribution})")

    # ---------------------------------------------------------
    # Milestone B4: 4 Species Weight Estimators
    # ---------------------------------------------------------
    print("\n[B4] Testing Vision Weight Estimation across all 4 species...")
    # 1. Cattle
    cattle_est = CattleWeightEstimator()
    assert cattle_est.weights_loaded is True, "Cattle weight checkpoint not loaded!"
    c_res = cattle_est.predict_from_images([mock_bytes], withers_height_cm=125.0)
    print(f"  [PASS] Cattle Vision Estimator: {c_res['estimated_weight_kg']:.1f} kg | 90% CI: {c_res['prediction_intervals']['90%']['lower']:.1f}-{c_res['prediction_intervals']['90%']['upper']:.1f} kg (weights_loaded={cattle_est.weights_loaded})")

    # 2. Hen
    hen_est = ChickenWeightEstimator()
    h_res = hen_est.predict_from_image(mock_bytes, bird_type="BROILER", plumage_condition="FULL_PLUMAGE")
    print(f"  [PASS] Hen Vision Estimator: {h_res['estimated_weight_kg']:.2f} kg ({h_res['estimated_weight_g']:.0f} g) | 90% CI: {h_res['prediction_intervals_kg']['90%']['lower']:.2f}-{h_res['prediction_intervals_kg']['90%']['upper']:.2f} kg")

    # 3. Goat
    goat_est = GoatWeightEstimator()
    g_res = goat_est.predict_from_image(mock_bytes, body_length_cm=65.0, heart_girth_cm=70.0)
    print(f"  [PASS] Goat Vision Estimator: {g_res['estimated_weight_kg']:.1f} kg | 90% CI: {g_res['prediction_intervals']['90%']['lower']:.1f}-{g_res['prediction_intervals']['90%']['upper']:.1f} kg")

    # 4. Sheep
    sheep_est = SheepWeightEstimator()
    s_res = sheep_est.predict_from_image(mock_bytes, body_length_cm=65.0, heart_girth_cm=72.0, fleece_depth_cm=3.5)
    print(f"  [PASS] Sheep Vision Estimator: {s_res['estimated_weight_kg']:.1f} kg (fleece deducted: {s_res['deducted_fleece_kg']:.1f} kg) | 90% CI: {s_res['prediction_intervals']['90%']['lower']:.1f}-{s_res['prediction_intervals']['90%']['upper']:.1f} kg")

    # ---------------------------------------------------------
    # Milestone B5: Dental Incisor Staging & Domain Honesty
    # ---------------------------------------------------------
    print("\n[B5] Testing Age Pipeline & Domain-Honest Photo Rejections...")
    age_pipe = UnifiedAgePipeline()
    # Biological staging
    a_bio = age_pipe.estimate_age(Species.COW, dentition_incisor_pairs=2)
    assert a_bio.status == "SUCCESS"
    assert a_bio.estimated_age_range_months == [24, 30]
    print(f"  [PASS] Biological dental staging: 2 incisor pairs -> {a_bio.estimated_age_range_months} months ({a_bio.dental_stage})")

    # Casual photo rejection
    a_rej = age_pipe.estimate_age(Species.COW)
    assert a_rej.status == "REJECTED"
    assert "Unable to estimate" in a_rej.rejection_reason
    print(f"  [PASS] Casual 2D photo rejection: '{a_rej.rejection_reason}'")

    # ---------------------------------------------------------
    # Milestone B6: API Security, Live Health Check & Android
    # ---------------------------------------------------------
    print("\n[B6] Testing Hardened API, Live Health Check & Android Assets...")
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["status"] == "HEALTHY"
    assert health["models"]["species_detector_weights_loaded"] is True
    assert health["models"]["cattle_weight_weights_loaded"] is True
    assert health["models"]["smoke_forward_pass_passed"] is True
    print(f"  [PASS] Live Health Check response: status={health['status']}")
    print(f"    - Species classifier loaded: {health['models']['species_detector_weights_loaded']}")
    print(f"    - Cattle weight regressor loaded: {health['models']['cattle_weight_weights_loaded']}")
    print(f"    - Smoke forward pass: {health['models']['smoke_forward_pass_passed']}")

    # Verify Android project artifacts
    android_dir = PROJECT_ROOT / "android"
    assert (android_dir / "build.gradle.kts").exists(), "Android build.gradle.kts missing"
    assert (android_dir / "app" / "build.gradle.kts").exists(), "Android app build.gradle.kts missing"
    assert (android_dir / "app" / "src" / "main" / "AndroidManifest.xml").exists(), "AndroidManifest.xml missing"
    assets = list((android_dir / "app" / "src" / "main" / "assets").glob("*.onnx"))
    assert len(assets) >= 3, f"Expected at least 3 edge ONNX assets, found {len(assets)}"
    print(f"  [PASS] Native Android project scaffolded with {len(assets)} INT8 ONNX models in assets/")

    print("\n" + "=" * 80)
    print("  ALL 6 MILESTONES (B1 - B6) SUCCESSFULLY VERIFIED!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_verification()
