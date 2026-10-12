# Batch B2: Error & Uncertainty Analysis Report

**Evaluated Animals:** 510 | **Evaluation Protocol:** 5-Fold Grouped Animal Cross-Validation

---

## 1. Overall Model Comparison: Linear vs. Log-Target Regression

| Model Configuration | Target Function | MAE (kg) | RMSE (kg) | $R^2$ | MAPE (%) | Mean Bias (kg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Model** | $y = \text{weight}$ | **25.78** | **37.48** | **0.745** | 10.0% | **-0.08** |
| **Log Model (Raw Exponentiation)** | $z = \log(y), \; \hat{y} = \exp(\hat{z})$ | **21.27** (Best) | **34.16** (Best) | **0.788** (Best) | **7.9%** (Best) | **-2.01** |
| **Log Model (Variance-Adjusted)** | $\hat{y} = \exp(\hat{z} + \frac{1}{2}\sigma^2)$ | **21.30** | **34.11** | **0.789** | **7.9%** | **-1.54** |

### Key Observations:
- The log-target model reduces overall MAE from **25.78 kg to 21.27 kg** (a **17.5% relative error reduction**).
- Coefficient of determination ($R^2$) improves from **0.745 to 0.788**.

---

## 2. Weight Bracket Stratified Comparison (Heavy-Tail Trade-off)

| Weight Bracket | Herd Cohort ($N$) | Linear MAE | Log MAE | $\Delta$ MAE | Linear Bias | Log Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LIGHT (< 220 kg)** | 152 (29.8%) | 21.88 kg | **15.91 kg** | -5.97 kg (-27.3%) | +5.59 kg | +7.70 kg |
| **MEDIUM (220 - 300 kg)** | 292 (57.3%) | 23.72 kg | **18.08 kg** | -5.65 kg (-23.8%) | +4.97 kg | +1.92 kg |
| **HEAVY (> 300 kg)** | 66 (12.9%) | 43.89 kg | **47.72 kg** | +3.83 kg (+8.7%) | -35.50 kg | -41.80 kg |

### Heavy-Tail Subgroup Trade-off Analysis:
1. **Normal-Weight Animals (<300 kg, 87.1% of herd):** Log transformation delivers dramatic improvements, cutting MAE by 27.3% on light cattle and 23.8% on medium cattle.
2. **Heavy Animals (>300 kg, 12.9% of herd):** Because log-transformed regression minimizes relative percentage errors rather than absolute kilograms, errors on cattle >300 kg expand from 43.89 kg to 47.72 kg (+8.7%).
3. **Commercial Safeguard:** When point estimates exceed 300 kg, the system returns `CAUTION_HEAVY_ANIMAL_USE_PHYSICAL_SCALE` advising the use of a physical platform scale for trade settlement.

---

## 3. Conformal Prediction Interval Performance (Strict Leak-Free Calibration)

Conformal intervals were calibrated strictly within training folds using out-of-fold nonconformity scores, guaranteeing zero test-label leakage.

| Metric | Linear Model (90%) | Linear Model (95%) | Log Model (90%) | Log Model (95%) |
| :--- | :---: | :---: | :---: | :---: |
| **Empirical Coverage (%)** | **89.8%** | **94.9%** | **90.6%** | **95.5%** |
| **Average Half-Width (kg)** | ±53.88 kg | ±62.00 kg | ±43.35 kg | ±58.57 kg |
| **Total Interval Width (kg)** | 107.75 kg | 124.01 kg | 86.70 kg | 117.15 kg |

### Stratified Coverage by Weight Bracket:

| Weight Bracket | Linear Standard 90% | Linear Adaptive 90% | Log Standard 90% | Log Adaptive 90% |
| :--- | :---: | :---: | :---: | :---: |
| **LIGHT (< 220 kg)** | 93.4% | **87.5%** | 97.4% | **91.4%** |
| **MEDIUM (220 - 300 kg)** | 91.1% | **91.4%** | 93.2% | **91.4%** |
| **HEAVY (> 300 kg)** | 75.8% | **87.9%** | 63.6% | **86.4%** |

> [!NOTE]
> Adaptive conformal prediction resolves the heteroscedastic coverage deficit on cattle >300 kg, lifting 90% coverage on heavy stock from 75.8% to **87.9%** (Linear) and from 63.6% to **86.4%** (Log).
