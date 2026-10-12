# LivestockVisionAI Platform

Enterprise Multi-Species Livestock Measurement and Computer-Vision Platform.

## Quick Links
- **Root Setup & Test Guide:** [README.md](../README.md)
- **Model Card & Scientific Integrity:** [MODEL_CARD.md](MODEL_CARD.md)
- **Edge & Android Architecture:** [android_edge_spec.md](android_edge_spec.md)
- **Operations & Rollback Guide:** [operations_and_rollback.md](operations_and_rollback.md)
- **Field Trial Governance:** [B8_prospective_trial_report.md](B8_prospective_trial_report.md)

---

## Quickstart

### Running Tests
From the project root:
```bash
pytest tests -v
# or
python verify_all.py
```

### Running the API
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is accessible at `http://localhost:8000/docs`.
Interactive Web Dashboard is accessible at `http://localhost:8000/`.
