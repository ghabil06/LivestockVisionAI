# LivestockVisionAI Platform

Enterprise Multi-Species Livestock Measurement and Computer-Vision Platform.
Engineered for cattle, poultry, caprine, and ovine livestock management across web, mobile, and edge environments.

---

## 1. Quickstart & Environment Setup

### Prerequisites
- Python 3.9+ (tested through Python 3.14)
- Virtual environment recommended (`venv` or `conda`)

### Installation
Clone or navigate to the repository:
```bash
cd LivestockVisionAI
```

Install core dependencies:
```bash
pip install -r requirements.txt
```

Optional: Install edge/mobile export and quantization tools:
```bash
pip install onnx>=1.14.0 onnxruntime>=1.15.0
```

Editable development installation:
```bash
pip install -e .
```

---

## 2. Running Automated Tests

The test suite is structured with portable relative path resolution and skip guards for optional edge runtimes.

Run all tests from within the repository root:
```bash
pytest tests -v
# or
python -m pytest tests -v
```

Run specific test modules:
```bash
# Data ingestion & fold isolation
pytest tests/test_data_ingestion.py -v

# Computer vision biological age pipelines
pytest tests/test_age_pipelines.py -v

# Adaptive conformal uncertainty bounds
pytest tests/test_uncertainty_engine.py -v

# Edge & ONNX parity
pytest tests/test_edge_parity.py -v
```

---

## 3. Comprehensive System Verification

Run the end-to-end multi-pillar verification suite:
```bash
python verify_all.py
```
This executes automated checks across:
- **Pillar 1:** Data Ingestion, Fold Isolation & Unit Conversions
- **Pillar 2:** Multi-Species Recognition & Rejection Handling
- **Pillar 3:** Weight Estimation & Conformal Prediction Intervals
- **Pillar 4:** Biological Age Staging (Dental CV & Poultry Plumage)
- **Pillar 5:** Adaptive Uncertainty Engine (<200 kg, 200–300 kg, >300 kg)
- **Pillar 6:** REST API & Interactive Web Dashboard
- **Pillar 7:** Android SQLite Offline Queue & Edge Parity
- **Pillar 8:** Field Trial Governance & Model Documentation

---

## 4. Starting the Server & Web Dashboard

Start the FastAPI ASGI application:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation:** Open `http://localhost:8000/docs`
- **Livestock Vision Dashboard:** Open `http://localhost:8000/`

---

## 5. Architectural Guardrails & Scientific Integrity

1. **Domain Honesty:** Chronological age is **never** estimated from casual full-body photographs. Ruminants require veterinary anterior dental inspection; poultry requires hatch cohort records or comb/plumage morphological staging.
2. **Adaptive Conformal Intervals:** Uncertainty intervals expand conservatively for heavy animals (>300 kg), triggering a mandatory `CAUTION_HEAVY_ANIMAL_USE_PHYSICAL_SCALE` physical scale advisory.
3. **Data Isolation:** Cross-validation folds are strictly grouped by animal ID (`fold_id`), guaranteeing zero animal leakage between training and evaluation splits.
4. **Strict Dataset Loading:** In training mode (`strict=True`), dataset loaders fail loudly on missing or corrupted image files instead of masking errors with silent fallback tensors.
