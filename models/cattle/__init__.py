"""Cattle specific models: weight regression and dental age estimation."""

from .weight_estimator import CattleWeightEstimator
from .age_estimator import CattleAgeEstimator

__all__ = ["CattleWeightEstimator", "CattleAgeEstimator"]
