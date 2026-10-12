# Batch B1 Engineering Requirements Specification

## 1. System Objective
The **LivestockVisionAI** platform is an enterprise-grade multi-species computer-vision measurement system engineered to deliver calibrated weight estimation, life-stage biometrics, and rigorous quality-controlled rejections across four livestock species:
* **Cattle** (*Bos indicus* / *Bos taurus*)
* **Poultry / Chicken** (*Gallus gallus domesticus*)
* **Goat** (*Capra hircus*)
* **Sheep** (*Ovis aries*)

---

## 2. Non-Negotiable Operational Principles

### 2.1 Rejection of Degraded Captures ("Unable to Estimate Reliably")
The system strictly refuses to output arbitrary point estimates on degraded or out-of-distribution inputs. Every capture must pass automated quality gates:
1. **Blur Detection:** Laplacian variance threshold $\sigma^2_{\text{Laplacian}} \ge 100.0$.
2. **Exposure Limits:** Mean luminance $\mu_{\text{gray}} \in [40.0, 225.0]$.
3. **Subject Framing:** Animal bounding box area ratio must occupy between $15\%$ and $88\%$ of total frame area to prevent distant captures or truncated silhouettes.

### 2.2 Calibrated Prediction Intervals
* No naked point estimates are served to end users.
* Every weight prediction must be accompanied by finite-sample, distribution-free **conformal prediction intervals** at the nominal 90% ($\alpha=0.10$) and 95% ($\alpha=0.05$) levels.
* Conformal quantiles must be calibrated strictly on out-of-fold or held-out test residuals.

### 2.3 Domain Honesty for Age Estimation
* Single 2D body photographs **cannot** reliably predict chronological age due to environmental, nutritional, and sexual dimorphism variations.
* For ruminants (Cattle, Goats, Sheep), age staging is strictly grounded in **veterinary dental incisor pair eruption** (Milk teeth, 1 pair, 2 pairs, 3 pairs, Full mouth).
* For poultry, age is determined by **verified hatch-date cohort tracking**, not single-bird visual guessing.

---

## 3. Data Schema & Architecture Contract

### 3.1 Schemas
* **`AnimalRecord`:** Biological identity, species, breed, sex, certified scale weight, and dentition.
* **`MeasurementRecord`:** Physical sensor measurement events, calibration certificates, technician IDs, timestamps.
* **`CaptureMetadata`:** Camera make, focal length, estimated distance, lighting condition, quality metrics.

### 3.2 Modularity
* `models/shared/`: Cross-cutting quality filtering and conformal uncertainty calibration engines.
* `models/{species}/`: Independent species pipelines preventing cross-species feature bleed.
* `api/`: RESTful endpoints built with FastAPI, enabling decoupled integration for web dashboards and mobile clients.
* `edge/`: ONNX export and quantization parity checkers ensuring edge and offline parity ($\max \Delta \le 0.10\text{ kg}$).
