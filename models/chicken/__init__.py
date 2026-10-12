"""Chicken / poultry specific model interfaces."""

from .weight_estimator import ChickenWeightEstimator
from .age_estimator import ChickenAgeEstimator

__all__ = ["ChickenWeightEstimator", "ChickenAgeEstimator"]
