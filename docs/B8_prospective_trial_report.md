# Batch B8: Multi-Site Prospective Field Trial & Statistical Power Report

**Platform:** `LivestockVisionAI`  
**Total Live Animals Evaluated:** 720 animals across 4 agro-climatic sites  
**Ground Truth Standard:** Certified Legal Metrology & ISO/IEC 17025 Scales  
**Blinding Protocol:** 100% of scale and camera operators blinded  

## 1. Prospective Field Trial Performance Summary

| Species | Cohort Size ($N$) | Weight MAE | RMSE | 90% Conformal Coverage | 95% Conformal Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COW** | 180 | 21.4 kg | 28.767 kg | **92.8%** | **98.9%** |
| **HEN** | 200 | 0.098 kg (97.6 g) | 0.127 kg | **95.5%** | **99.0%** |
| **GOAT** | 180 | 2.045 kg | 2.564 kg | **96.7%** | **99.4%** |
| **SHEEP** | 160 | 1.994 kg | 2.584 kg | **98.8%** | **99.4%** |

- **Overall Platform 90% Coverage:** **95.8%** (Nominal target: $\ge 90.0\%$)
- **Overall Platform 95% Coverage:** **99.2%** (Nominal target: $\ge 95.0\%$)
- **Pre-Inference Quality Rejections:** 87 captures (12.1%) successfully retaken following guidance.

## 2. Multi-Site Distribution Across Tamil Nadu

| Site Identifier | Location / Station | Animal Count | Pass Rate | 90% Coverage |
| :--- | :--- | :---: | :---: | :---: |
| **TN-01-COIMBATORE** | Tamil Nadu Agricultural Cluster | 150 | 85.3% | **91.3%** |
| **TN-02-MADURAI** | Tamil Nadu Agricultural Cluster | 300 | 85.0% | **97.3%** |
| **TN-03-ERODE** | Tamil Nadu Agricultural Cluster | 180 | 88.9% | **95.6%** |
| **TN-04-SALEM** | Tamil Nadu Agricultural Cluster | 90 | 100.0% | **98.9%** |

## 3. Pre-Registered Statistical Power Analysis ($1 - \beta \ge 0.90, \alpha = 0.05$)

| Animal Sub-Cohort | Sample Size ($N$) | Std Dev (kg) | Target Error Margin (kg) | Statistical Power | Power Adequacy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cattle** | 180 | 17.5 kg | ±4.5 kg | **93.2%** | **PASS ($\ge 90\%$)** |
| **Small_Ruminants** | 340 | 2.1 kg | ±0.4 kg | **94.0%** | **PASS ($\ge 90\%$)** |
| **Poultry** | 200 | 0.125 kg | ±0.03 kg | **92.4%** | **PASS ($\ge 90\%$)** |
