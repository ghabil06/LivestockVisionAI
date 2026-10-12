# Batch B4: Species-Specific Weight Estimation Benchmark Report

- **Evaluated Cattle Count:** 510 animals (Grouped 5-Fold Cross-Validation)
- **Linear Model:** MAE = 32.34 kg, RMSE = 47.88 kg, R² = 0.584
- **Log-Target Model:** MAE = 28.56 kg, RMSE = 45.26 kg, R² = 0.629

### Conformal Uncertainty Performance

- **Nominal 90% Interval:** ±65.03 kg | Empirical Coverage = 90.0%
- **Nominal 95% Interval:** ±90.93 kg | Empirical Coverage = 95.3%

### Weight-Stratified Error Breakdown

| Weight Bracket | Sample Count | Linear MAE (kg) | Log-Target MAE (kg) |
| :--- | :---: | :---: | :---: |
| **< 220 kg** | 152 | 25.73 | 20.45 |
| **220 - 300 kg** | 292 | 26.05 | 21.38 |
| **> 300 kg** | 66 | 75.38 | 79.01 |
