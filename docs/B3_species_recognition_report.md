# Batch B3: Species Recognition MVP Benchmark Report

- Total Evaluated Records: 660
- OOD Rejection Count: 1 (0.1%)
- Overall Accuracy (on Accepted Captures): 100.00%

### Confusion Matrix (Rows: True, Cols: Predicted)

| Species | COW | HEN | GOAT | SHEEP |
| :--- | :---: | :---: | :---: | :---: |
| **COW** | 510 | 0 | 0 | 0 |
| **HEN** | 0 | 49 | 0 | 0 |
| **GOAT** | 0 | 0 | 50 | 0 |
| **SHEEP** | 0 | 0 | 0 | 50 |

### Per-Species Metrics

| Species | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **COW** | 100.0% | 100.0% | 100.0% | 510.0 |
| **HEN** | 100.0% | 100.0% | 100.0% | 49.0 |
| **GOAT** | 100.0% | 100.0% | 100.0% | 50.0 |
| **SHEEP** | 100.0% | 100.0% | 100.0% | 50.0 |
