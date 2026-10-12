"""
Multi-Species Weight Estimation Training and Benchmark Pipeline.
Evaluates linear vs log regression targets, out-of-fold cross-validation,
weight-stratified error distributions, and empirical conformal coverage.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LivestockVisionAI.models.shared.uncertainty_engine import ConformalUncertaintyEngine


def run_weight_benchmark(
    cattle_manifest_path: Path,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Runs deterministic grouped cross-validation benchmark on cattle scale weights,
    evaluating linear vs log targets and conformal coverage.
    """
    df = pd.read_csv(cattle_manifest_path)
    num_folds = 5

    # Features: withers height if present, plus synthetic visual feature simulation matching pilot
    y_true = df["ground_truth_weight_kg"].values
    heights = df["withers_height_cm"].fillna(df["withers_height_cm"].median()).values

    oof_linear = np.zeros(len(df))
    oof_log = np.zeros(len(df))

    # Evaluate across 5 folds
    for f in range(num_folds):
        train_mask = (df["fold_id"] != f).values
        val_mask = (df["fold_id"] == f).values

        X_train = heights[train_mask].reshape(-1, 1)
        y_train = y_true[train_mask]
        X_val = heights[val_mask].reshape(-1, 1)

        # 1. Linear model
        reg_lin = Ridge(alpha=1.0)
        reg_lin.fit(X_train, y_train)
        oof_linear[val_mask] = reg_lin.predict(X_val)

        # 2. Log model
        reg_log = Ridge(alpha=1.0)
        reg_log.fit(X_train, np.log(y_train))
        oof_log[val_mask] = np.exp(reg_log.predict(X_val))

    # Metrics
    mae_lin = float(mean_absolute_error(y_true, oof_linear))
    rmse_lin = float(np.sqrt(mean_squared_error(y_true, oof_linear)))
    r2_lin = float(r2_score(y_true, oof_linear))

    mae_log = float(mean_absolute_error(y_true, oof_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true, oof_log)))
    r2_log = float(r2_score(y_true, oof_log))

    # Calibrate conformal engine on linear residuals
    engine = ConformalUncertaintyEngine(default_species="COW")
    resids = y_true - oof_linear
    quantiles = engine.calibrate("COW", resids, alphas=[0.10, 0.05])

    # Empirical coverage check
    cov_90 = float(np.mean(np.abs(resids) <= quantiles[0.10]))
    cov_95 = float(np.mean(np.abs(resids) <= quantiles[0.05]))

    # Stratified analysis by weight bracket
    brackets = [
        ("< 220 kg", y_true < 220),
        ("220 - 300 kg", (y_true >= 220) & (y_true <= 300)),
        ("> 300 kg", y_true > 300),
    ]

    stratified_metrics = []
    for label, mask in brackets:
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            m_lin = mean_absolute_error(y_true[mask], oof_linear[mask])
            m_log = mean_absolute_error(y_true[mask], oof_log[mask])
            stratified_metrics.append({
                "bracket": label,
                "count": n_samples,
                "mae_linear": round(float(m_lin), 2),
                "mae_log": round(float(m_log), 2),
            })

    results = {
        "n_animals": len(df),
        "linear": {
            "mae": round(mae_lin, 2),
            "rmse": round(rmse_lin, 2),
            "r2": round(r2_lin, 3),
        },
        "log": {
            "mae": round(mae_log, 2),
            "rmse": round(rmse_log, 2),
            "r2": round(r2_log, 3),
        },
        "conformal": {
            "q90": quantiles[0.10],
            "q95": quantiles[0.05],
            "empirical_coverage_90": round(cov_90 * 100, 1),
            "empirical_coverage_95": round(cov_95 * 100, 1),
        },
        "stratified": stratified_metrics,
    }

    if output_report_path:
        with open(output_report_path, "w") as f:
            f.write("# Batch B4: Species-Specific Weight Estimation Benchmark Report\n\n")
            f.write(f"- **Evaluated Cattle Count:** {len(df)} animals (Grouped 5-Fold Cross-Validation)\n")
            f.write(f"- **Linear Model:** MAE = {results['linear']['mae']:.2f} kg, RMSE = {results['linear']['rmse']:.2f} kg, R² = {results['linear']['r2']:.3f}\n")
            f.write(f"- **Log-Target Model:** MAE = {results['log']['mae']:.2f} kg, RMSE = {results['log']['rmse']:.2f} kg, R² = {results['log']['r2']:.3f}\n\n")
            f.write("### Conformal Uncertainty Performance\n\n")
            f.write(f"- **Nominal 90% Interval:** ±{results['conformal']['q90']:.2f} kg | Empirical Coverage = {results['conformal']['empirical_coverage_90']:.1f}%\n")
            f.write(f"- **Nominal 95% Interval:** ±{results['conformal']['q95']:.2f} kg | Empirical Coverage = {results['conformal']['empirical_coverage_95']:.1f}%\n\n")
            f.write("### Weight-Stratified Error Breakdown\n\n")
            f.write("| Weight Bracket | Sample Count | Linear MAE (kg) | Log-Target MAE (kg) |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for sm in stratified_metrics:
                f.write(f"| **{sm['bracket']}** | {sm['count']} | {sm['mae_linear']:.2f} | {sm['mae_log']:.2f} |\n")

    return results


if __name__ == "__main__":
    man_path = Path(__file__).resolve().parent.parent / "data" / "manifests" / "cattle_manifest.csv"
    rep_path = Path(__file__).resolve().parent.parent / "docs" / "B4_weight_estimation_report.md"
    run_weight_benchmark(man_path, rep_path)
    print(f"Generated weight estimation report at {rep_path}")
