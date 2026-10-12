# Batch B7: Android Architecture & Edge Deployment Benchmark Report

**Platform:** `LivestockVisionAI`  
**Execution Timestamp:** 2026-10-12T08:30:10Z  
**Target Mobile OS:** Android 9.0 (API 28) - Android 15 (API 35)  
**Hardware Acceleration Provider:** Android NNAPI / Qualcomm Hexagon NPU / ARM Mali GPU  

## 1. On-Device Model Compression & Quantization Audit

| Model Component | Task / Graph | FP32 Size (MB) | Dynamic INT8 Size (MB) | Compression Ratio |
| :--- | :--- | :---: | :---: | :---: |
| **Species Detector** | 4-Species Logits [B, 4] | 42.6 MB | **10.7 MB** | **74.9%** |
| **Single-View Weight** | Linear Regressor [B, 1] | 42.6 MB | **10.7 MB** | **74.9%** |
| **Multi-View Weight** | 4-View Pooled Regressor [B, 4, 3, 224, 224] | 43.2 MB | **10.9 MB** | **74.8%** |

## 2. Numerical Parity Verification (PyTorch vs ONNX Runtime)

- **Cattle Weight Regressor Parity:** Status = **PASSED**  
  * Maximum Discrepancy: `2e-06 kg` (Tolerance: `0.1 kg`)  
  * Mean Discrepancy: `1e-06 kg` across 20 synthetic samples  
- **Species Classifier Parity:** Status = **PASSED**  
  * Class Match Rate: **100.0%**  
  * Maximum Probability Discrepancy: `0.0` (Tolerance: `0.005`)  

## 3. Mobile CPU Inference Latency Benchmark

- **Species Classifier FP32 Latency:** `328.7 ms`  
- **Species Classifier INT8 Latency:** `326.5 ms` (**1.01x speedup**)  
- **Cattle Weight Regressor FP32 Latency:** `193.4 ms`  
- **Cattle Weight Regressor INT8 Latency:** `308.1 ms` (**0.63x speedup**)  

## 4. Android APK Binary Size & Memory Budget Audit

- **Base APK Size:** `14.5 MB`  
- **Bundled Quantized Models:** `21.4 MB`  
- **Total Android App Download Footprint:** **35.9 MB** (Budget: `< 50.0 MB` -> **COMPLIANT**)  
- **Estimated Active Runtime RAM:** `85.0 MB` (Fits easily within low-tier 2GB Android RAM devices)  
