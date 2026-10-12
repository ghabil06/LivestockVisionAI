"""
Numerical Parity Verification & Edge Latency Benchmark.
Asserts that exported ONNX models produce predictions within 0.10 kg of native PyTorch models
and measures on-device inference speedups.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def verify_weight_parity(
    pytorch_model: nn.Module,
    onnx_model_path: Path,
    tolerance_kg: float = 0.10,
    num_samples: int = 20,
    input_shape: tuple = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """
    Verifies that the ONNX weight model matches PyTorch outputs within tolerance_kg.
    """
    if not HAS_ORT:
        return {
            "status": "SKIPPED",
            "reason": "onnxruntime not installed",
            "passed": False,
        }

    pytorch_model.eval()
    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_diff = 0.0
    diffs = []

    with torch.no_grad():
        for _ in range(num_samples):
            test_tensor = torch.randn(*input_shape)
            pt_out = pytorch_model(test_tensor).cpu().numpy().flatten()[0]

            ort_out = session.run(None, {input_name: test_tensor.numpy()})[0].flatten()[0]

            diff = float(np.abs(pt_out - ort_out))
            diffs.append(diff)
            if diff > max_diff:
                max_diff = diff

    passed = max_diff <= tolerance_kg
    return {
        "status": "PASSED" if passed else "FAILED",
        "max_discrepancy_kg": round(max_diff, 6),
        "mean_discrepancy_kg": round(float(np.mean(diffs)), 6),
        "tolerance_kg": tolerance_kg,
        "num_samples": num_samples,
        "passed": passed,
    }


def verify_species_parity(
    pytorch_model: nn.Module,
    onnx_model_path: Path,
    tolerance_prob: float = 0.005,
    num_samples: int = 20,
    input_shape: tuple = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """
    Verifies species classification logits and softmax probabilities match between PyTorch and ONNX.
    """
    if not HAS_ORT:
        return {
            "status": "SKIPPED",
            "reason": "onnxruntime not installed",
            "passed": False,
        }

    pytorch_model.eval()
    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_prob_diff = 0.0
    class_match_count = 0

    with torch.no_grad():
        for _ in range(num_samples):
            test_tensor = torch.randn(*input_shape)
            pt_logits = pytorch_model(test_tensor)
            pt_probs = torch.softmax(pt_logits, dim=-1).cpu().numpy().flatten()
            pt_pred = int(np.argmax(pt_probs))

            ort_logits = session.run(None, {input_name: test_tensor.numpy()})[0].flatten()
            # Softmax on ORT logits
            exp_logits = np.exp(ort_logits - np.max(ort_logits))
            ort_probs = exp_logits / np.sum(exp_logits)
            ort_pred = int(np.argmax(ort_probs))

            if pt_pred == ort_pred:
                class_match_count += 1

            diff = float(np.max(np.abs(pt_probs - ort_probs)))
            if diff > max_prob_diff:
                max_prob_diff = diff

    all_matched = (class_match_count == num_samples)
    passed = (max_prob_diff <= tolerance_prob) and all_matched

    return {
        "status": "PASSED" if passed else "FAILED",
        "max_prob_discrepancy": round(max_prob_diff, 6),
        "class_match_rate": round(class_match_count / num_samples, 4),
        "tolerance_prob": tolerance_prob,
        "passed": passed,
    }


def benchmark_edge_latency(
    onnx_fp32_path: Path,
    onnx_int8_path: Optional[Path] = None,
    num_runs: int = 50,
    warmup: int = 10,
    input_shape: tuple = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """
    Measures CPU latency in milliseconds across FP32 and quantized INT8 ONNX models.
    """
    if not HAS_ORT:
        return {"status": "SKIPPED"}

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    def measure_session(session_path: Path) -> float:
        session = ort.InferenceSession(str(session_path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        # Warmup
        for _ in range(warmup):
            session.run(None, {name: dummy_input})
        # Benchmark
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run(None, {name: dummy_input})
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))

    fp32_latency = measure_session(onnx_fp32_path)
    int8_latency = measure_session(onnx_int8_path) if onnx_int8_path and onnx_int8_path.exists() else None

    speedup = round(fp32_latency / int8_latency, 2) if int8_latency else 1.0

    return {
        "fp32_latency_ms": round(fp32_latency, 2),
        "int8_latency_ms": round(int8_latency, 2) if int8_latency else None,
        "speedup_factor": speedup,
        "num_runs": num_runs,
    }
