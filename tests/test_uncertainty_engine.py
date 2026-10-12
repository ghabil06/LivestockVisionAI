"""
Unit tests for Conformal Uncertainty Engine.
Verifies finite-sample conformal quantiles, interval bounds, and OOD flagging.
"""

import numpy as np
import pytest

from LivestockVisionAI.models.shared.uncertainty_engine import ConformalUncertaintyEngine


def test_default_cattle_quantiles():
    engine = ConformalUncertaintyEngine(default_species="COW")
    q90 = engine.get_quantile("COW", 0.10)
    q95 = engine.get_quantile("COW", 0.05)

    assert q90 == 53.88
    assert q95 == 62.00


def test_conformal_intervals_computation():
    engine = ConformalUncertaintyEngine()
    res = engine.compute_intervals(point_prediction=250.0, species="COW", min_physio_bound=35.0)

    assert res["point_estimate"] == 250.0
    assert not res["is_out_of_distribution"]
    assert res["reliability_status"] == "CALIBRATED"

    # 90% interval
    int90 = res["intervals"]["90%"]
    assert int90["lower"] == round(250.0 - 53.88, 2)
    assert int90["upper"] == round(250.0 + 53.88, 2)
    assert int90["half_width"] == 53.88

    # 95% interval
    int95 = res["intervals"]["95%"]
    assert int95["lower"] == round(250.0 - 62.00, 2)
    assert int95["upper"] == round(250.0 + 62.00, 2)
    assert int95["half_width"] == 62.00


def test_empirical_calibration_formula():
    engine = ConformalUncertaintyEngine()
    # 100 known residuals: 1.0, 2.0, ..., 100.0
    fake_residuals = np.arange(1.0, 101.0)
    quantiles = engine.calibrate("GOAT", fake_residuals, alphas=[0.10, 0.05])

    # n = 100. For alpha = 0.10, ceil((101)*0.90) = ceil(90.9) = 91 -> index 90 -> value 91.0
    assert quantiles[0.10] == 91.0
    # For alpha = 0.05, ceil((101)*0.95) = ceil(95.95) = 96 -> index 95 -> value 96.0
    assert quantiles[0.05] == 96.0


def test_ood_detection():
    engine = ConformalUncertaintyEngine()
    # 750 kg is far above the 95-465 kg training envelope for the Bengal cattle dataset
    res = engine.compute_intervals(point_prediction=750.0, species="COW")
    assert res["is_out_of_distribution"]
    assert res["reliability_status"] == "UNRELIABLE_OOD"
    assert "falls outside validated support range" in res["ood_warning"]


def test_lower_bound_clipping():
    engine = ConformalUncertaintyEngine()
    # Point prediction 40.0 kg with +/- 53.88 would go negative without clipping
    res = engine.compute_intervals(point_prediction=40.0, species="COW", min_physio_bound=35.0)
    int90 = res["intervals"]["90%"]
    assert int90["lower"] == 35.0  # Clipped to minimum physiological bound
    assert int90["upper"] == round(40.0 + 53.88, 2)


def test_adaptive_weight_stratified_conformal():
    engine = ConformalUncertaintyEngine()

    # Bracket 1: Light (<200 kg) -> 90% half-width 28.5 kg
    res_light = engine.compute_intervals(point_prediction=160.0, species="COW", adaptive=True)
    assert res_light["weight_bracket"] == "LIGHT"
    assert res_light["intervals"]["90%"]["half_width"] == 28.50
    assert res_light["warning_code"] is None

    # Bracket 2: Medium (200-300 kg) -> 90% half-width 42.0 kg
    res_med = engine.compute_intervals(point_prediction=260.0, species="COW", adaptive=True)
    assert res_med["weight_bracket"] == "MEDIUM"
    assert res_med["intervals"]["90%"]["half_width"] == 42.00
    assert res_med["warning_code"] is None

    # Bracket 3: Heavy (>300 kg) -> 90% half-width 84.5 kg, flags warning
    res_heavy = engine.compute_intervals(point_prediction=360.0, species="COW", adaptive=True)
    assert res_heavy["weight_bracket"] == "HEAVY"
    assert res_heavy["intervals"]["90%"]["half_width"] == 84.50
    assert res_heavy["intervals"]["95%"]["half_width"] == 108.00
    assert res_heavy["warning_code"] == "CAUTION_HEAVY_ANIMAL_USE_PHYSICAL_SCALE"
    assert "exceeds 300 kg" in res_heavy["heavy_animal_warning"]
    assert res_heavy["reliability_status"] == "CAUTION_HEAVY"
