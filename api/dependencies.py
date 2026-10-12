"""
API Dependencies and Singletons.
Manages global lifecycle instances of filters, engines, and model estimators.
"""

from functools import lru_cache
from pathlib import Path
import yaml

from LivestockVisionAI.models.shared.quality_filter import QualityFilter
from LivestockVisionAI.models.shared.uncertainty_engine import ConformalUncertaintyEngine
from LivestockVisionAI.models.shared.species_detector import SpeciesDetector
from LivestockVisionAI.models.cattle.weight_estimator import CattleWeightEstimator
from LivestockVisionAI.models.cattle.age_estimator import CattleAgeEstimator
from LivestockVisionAI.models.chicken.weight_estimator import ChickenWeightEstimator
from LivestockVisionAI.models.chicken.age_estimator import ChickenAgeEstimator
from LivestockVisionAI.models.goat.weight_estimator import GoatWeightEstimator
from LivestockVisionAI.models.goat.age_estimator import GoatAgeEstimator
from LivestockVisionAI.models.sheep.weight_estimator import SheepWeightEstimator
from LivestockVisionAI.models.sheep.age_estimator import SheepAgeEstimator


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "platform_config.yaml"


@lru_cache()
def get_platform_config() -> dict:
    """Load configuration YAML with caching."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


@lru_cache()
def get_quality_filter() -> QualityFilter:
    """Singleton instance of image quality gate."""
    cfg = get_platform_config().get("quality_filter", {})
    return QualityFilter(
        min_laplacian_variance=cfg.get("min_laplacian_variance", 100.0),
        min_brightness_mean=cfg.get("min_brightness_mean", 40.0),
        max_brightness_mean=cfg.get("max_brightness_mean", 225.0),
        min_bbox_area_ratio=cfg.get("min_bbox_area_ratio", 0.15),
        max_bbox_area_ratio=cfg.get("max_bbox_area_ratio", 0.88),
    )


@lru_cache()
def get_uncertainty_engine() -> ConformalUncertaintyEngine:
    """Singleton instance of the conformal uncertainty engine."""
    return ConformalUncertaintyEngine(default_species="COW")


@lru_cache()
def get_species_detector() -> SpeciesDetector:
    """Singleton instance of the 4-species classifier."""
    return SpeciesDetector(confidence_threshold=0.65)


@lru_cache()
def get_cattle_weight_estimator() -> CattleWeightEstimator:
    return CattleWeightEstimator(uncertainty_engine=get_uncertainty_engine())


@lru_cache()
def get_cattle_age_estimator() -> CattleAgeEstimator:
    return CattleAgeEstimator()


@lru_cache()
def get_chicken_weight_estimator() -> ChickenWeightEstimator:
    return ChickenWeightEstimator(uncertainty_engine=get_uncertainty_engine())


@lru_cache()
def get_chicken_age_estimator() -> ChickenAgeEstimator:
    return ChickenAgeEstimator()


@lru_cache()
def get_goat_weight_estimator() -> GoatWeightEstimator:
    return GoatWeightEstimator(uncertainty_engine=get_uncertainty_engine())


@lru_cache()
def get_goat_age_estimator() -> GoatAgeEstimator:
    return GoatAgeEstimator()


@lru_cache()
def get_sheep_weight_estimator() -> SheepWeightEstimator:
    return SheepWeightEstimator(uncertainty_engine=get_uncertainty_engine())


@lru_cache()
def get_sheep_age_estimator() -> SheepAgeEstimator:
    return SheepAgeEstimator()


@lru_cache()
def get_unified_age_pipeline():
    from LivestockVisionAI.models.shared.age_pipeline import UnifiedAgePipeline
    return UnifiedAgePipeline()
