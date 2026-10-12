"""
Prediction API Router.
Handles multi-species weight regression and age estimation with conformal bounds,
multi-view pooling, and strict 'Unable to estimate reliably' rejection.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from pydantic import BaseModel, Field

from LivestockVisionAI.data.schemas.animal_record import Species
from LivestockVisionAI.data.schemas.capture_metadata import QualityStatus
from LivestockVisionAI.models.shared.quality_filter import QualityFilter
from LivestockVisionAI.models.shared.species_detector import SpeciesDetector
from LivestockVisionAI.api.dependencies import (
    get_quality_filter,
    get_species_detector,
    get_cattle_weight_estimator,
    get_cattle_age_estimator,
    get_chicken_weight_estimator,
    get_chicken_age_estimator,
    get_goat_weight_estimator,
    get_goat_age_estimator,
    get_sheep_weight_estimator,
    get_sheep_age_estimator,
    get_unified_age_pipeline,
)
from LivestockVisionAI.models.cattle.weight_estimator import CattleWeightEstimator
from LivestockVisionAI.models.cattle.age_estimator import CattleAgeEstimator
from LivestockVisionAI.models.chicken.weight_estimator import ChickenWeightEstimator
from LivestockVisionAI.models.chicken.age_estimator import ChickenAgeEstimator
from LivestockVisionAI.models.goat.weight_estimator import GoatWeightEstimator
from LivestockVisionAI.models.goat.age_estimator import GoatAgeEstimator
from LivestockVisionAI.models.sheep.weight_estimator import SheepWeightEstimator
from LivestockVisionAI.models.sheep.age_estimator import SheepAgeEstimator
from LivestockVisionAI.api.database import log_prediction_audit
from LivestockVisionAI.api.security import verify_api_key


router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])


class WeightPredictionResponse(BaseModel):
    status: str
    species: str
    rejection_reason: Optional[str] = None
    recommendation: Optional[str] = None
    estimated_weight_kg: Optional[float] = None
    estimated_weight_g: Optional[float] = None
    prediction_intervals: Optional[Dict[str, Any]] = None
    prediction_intervals_g: Optional[Dict[str, Any]] = None
    reliability_status: Optional[str] = None
    is_out_of_distribution: Optional[bool] = None
    warning_code: Optional[str] = None
    warning_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/species")
async def detect_species(
    file: UploadFile = File(..., description="Livestock capture image"),
    quality_filter: QualityFilter = Depends(get_quality_filter),
    species_detector: SpeciesDetector = Depends(get_species_detector),
    _auth: bool = Depends(verify_api_key),
):
    """Identify animal species (Cow, Hen, Goat, Sheep) with OOD rejection."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Quality Gate first
    q_res = quality_filter.evaluate(image_input=contents)
    if not q_res.passed:
        return {
            "status": "REJECTED",
            "predicted_species": "UNKNOWN",
            "rejection_reason": q_res.rejection_reason,
            "recommendation": q_res.recommendation,
        }

    det_res = species_detector.predict(image_input=contents)
    res_dict = det_res.to_dict()
    res_dict["status"] = "REJECTED_OOD" if det_res.is_out_of_distribution else "SUCCESS"
    return res_dict


@router.post("/weight", response_model=WeightPredictionResponse)
async def predict_weight(
    file: UploadFile = File(..., description="Primary animal capture image (lateral left)"),
    view_1: Optional[UploadFile] = File(None, description="Optional secondary body view (lateral right)"),
    view_2: Optional[UploadFile] = File(None, description="Optional tertiary body view (cranial/anterior)"),
    view_3: Optional[UploadFile] = File(None, description="Optional quaternary body view (caudal/dorsal)"),
    species: Optional[Species] = Form(None, description="Target species (COW, HEN, GOAT, SHEEP, or leave empty for auto-detect)"),
    reference_weight_kg: Optional[float] = Form(None, description="Optional baseline or prior weight estimate"),
    withers_height_cm: Optional[float] = Form(None, description="Withers height in cm (cattle)"),
    body_length_cm: Optional[float] = Form(None, description="Body length in cm (goats/sheep morphometrics)"),
    heart_girth_cm: Optional[float] = Form(None, description="Heart girth in cm (goats/sheep morphometrics)"),
    fleece_depth_cm: Optional[float] = Form(None, description="Fleece depth in cm (sheep only)"),
    is_shorn: bool = Form(False, description="Whether sheep is freshly shorn"),
    bird_type: Optional[str] = Form(None, description="Poultry breed type: BROILER, LAYER, or DUAL_PURPOSE"),
    plumage_condition: Optional[str] = Form(None, description="Plumage condition: FULL_PLUMAGE, MOLTING, JUVENILE_DOWN"),
    quality_filter: QualityFilter = Depends(get_quality_filter),
    species_detector: SpeciesDetector = Depends(get_species_detector),
    cattle_estimator: CattleWeightEstimator = Depends(get_cattle_weight_estimator),
    chicken_estimator: ChickenWeightEstimator = Depends(get_chicken_weight_estimator),
    goat_estimator: GoatWeightEstimator = Depends(get_goat_weight_estimator),
    sheep_estimator: SheepWeightEstimator = Depends(get_sheep_weight_estimator),
    _auth: bool = Depends(verify_api_key),
):
    """
    Run multi-species weight estimation pipeline:
    1. Screen image with QualityFilter.
    2. If degraded, reject with 'Unable to estimate reliably'.
    3. Auto-detect species if omitted or UNKNOWN.
    4. Collect optional additional views for multi-view vision pooling.
    5. Route to species-specific regressor with conformal prediction intervals.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Step 1: Enforce non-negotiable Quality Gate on primary image
    q_result = quality_filter.evaluate(image_input=contents)
    if not q_result.passed:
        log_prediction_audit(
            species=species.value if species else "UNKNOWN",
            prediction_type="WEIGHT",
            quality_status=q_result.status.value,
            status="REJECTED",
            laplacian_score=q_result.laplacian_variance,
            brightness_mean=q_result.brightness_mean,
            rejection_reason=q_result.rejection_reason,
        )
        return WeightPredictionResponse(
            status="REJECTED",
            species=species.value if species else "UNKNOWN",
            rejection_reason=q_result.rejection_reason,
            recommendation=q_result.recommendation,
            metadata={
                "laplacian_variance": q_result.laplacian_variance,
                "brightness_mean": q_result.brightness_mean,
                "quality_status": q_result.status.value,
            },
        )

    # Collect additional multi-view perspectives if uploaded
    views_bytes = [contents]
    for additional_view in [view_1, view_2, view_3]:
        if additional_view is not None:
            extra_bytes = await additional_view.read()
            if extra_bytes:
                views_bytes.append(extra_bytes)

    # Step 2: Determine target species (Explicit or Auto-Detected)
    target_species = species
    det_meta = None

    if target_species is None or target_species == Species.UNKNOWN:
        det_result = species_detector.predict(image_input=contents)
        det_meta = det_result.to_dict()
        if det_result.is_out_of_distribution or det_result.predicted_species == Species.UNKNOWN:
            return WeightPredictionResponse(
                status="REJECTED",
                species="UNKNOWN",
                rejection_reason=(
                    det_result.rejection_reason
                    or "Unable to estimate reliably: Species is UNKNOWN or unsupported."
                ),
                recommendation="Provide a clear, unobstructed lateral capture or manually select the animal species.",
                metadata=det_meta,
            )
        target_species = det_result.predicted_species

    # Step 3: Route to species pipeline
    if target_species == Species.COW:
        if reference_weight_kg is not None:
            result = cattle_estimator.predict_from_features_or_value(
                raw_prediction_kg=reference_weight_kg,
                adaptive=True,
                views_count=len(views_bytes),
            )
        else:
            result = cattle_estimator.predict_from_images(
                views=views_bytes,
                withers_height_cm=withers_height_cm,
                adaptive=True,
            )

        resp = WeightPredictionResponse(
            status="SUCCESS",
            species="COW",
            estimated_weight_kg=result["estimated_weight_kg"],
            prediction_intervals=result["prediction_intervals"],
            reliability_status=result["reliability_status"],
            is_out_of_distribution=result["is_out_of_distribution"],
            warning_code=result.get("warning_code"),
            warning_message=result.get("heavy_animal_warning"),
            metadata={
                "target_mode": result["target_mode"],
                "weight_bracket": result.get("weight_bracket"),
                "views_processed": len(views_bytes),
                "multi_view_mode": result.get("multi_view_mode", f"MULTI_VIEW_{len(views_bytes)}"),
                "warning": result.get("warning"),
                "warning_code": result.get("warning_code"),
                "species_detection": det_meta,
            },
        )

    elif target_species == Species.HEN:
        if reference_weight_kg is not None:
            result = chicken_estimator.predict_from_raw_weight(
                raw_weight_kg=reference_weight_kg,
                bird_type=bird_type,
                plumage_condition=plumage_condition,
            )
        else:
            result = chicken_estimator.predict_from_image(
                image_input=contents,
                bird_type=bird_type,
                plumage_condition=plumage_condition,
            )

        if result.get("status") == "REJECTED":
            return WeightPredictionResponse(
                status="REJECTED",
                species="HEN",
                rejection_reason=result.get("rejection_reason", "Unable to estimate reliably"),
                recommendation=result.get("recommendation", "Provide a clear bird capture"),
                is_out_of_distribution=True,
                metadata={"species_detection": det_meta},
            )

        resp = WeightPredictionResponse(
            status="SUCCESS",
            species="HEN",
            estimated_weight_kg=result["estimated_weight_kg"],
            estimated_weight_g=result["estimated_weight_g"],
            prediction_intervals=result["prediction_intervals_kg"],
            prediction_intervals_g=result["prediction_intervals_g"],
            reliability_status=result["reliability_status"],
            is_out_of_distribution=result["is_out_of_distribution"],
            metadata={
                "bird_type": result.get("bird_type"),
                "plumage_adjusted": result.get("plumage_adjusted"),
                "plumage_condition": result.get("plumage_condition"),
                "species_detection": det_meta,
            },
        )

    elif target_species == Species.GOAT:
        if reference_weight_kg is not None:
            result = goat_estimator.predict_from_raw_weight(raw_weight_kg=reference_weight_kg)
        else:
            result = goat_estimator.predict_from_image(
                image_input=contents,
                body_length_cm=body_length_cm,
                heart_girth_cm=heart_girth_cm,
            )

        if result.get("status") == "REJECTED":
            return WeightPredictionResponse(
                status="REJECTED",
                species="GOAT",
                rejection_reason=result.get("rejection_reason", "Unable to estimate reliably"),
                recommendation="Provide an unobstructed lateral capture of the goat.",
                is_out_of_distribution=True,
                metadata={"species_detection": det_meta},
            )

        resp = WeightPredictionResponse(
            status="SUCCESS",
            species="GOAT",
            estimated_weight_kg=result["estimated_weight_kg"],
            prediction_intervals=result["prediction_intervals"],
            reliability_status=result["reliability_status"],
            is_out_of_distribution=result["is_out_of_distribution"],
            metadata={
                "morphometrics": result.get("morphometrics"),
                "species_detection": det_meta,
            },
        )

    elif target_species == Species.SHEEP:
        if reference_weight_kg is not None:
            result = sheep_estimator.predict_from_raw_weight(
                raw_weight_kg=reference_weight_kg,
                fleece_depth_cm=fleece_depth_cm,
                is_shorn=is_shorn,
            )
        else:
            result = sheep_estimator.predict_from_image(
                image_input=contents,
                body_length_cm=body_length_cm,
                heart_girth_cm=heart_girth_cm,
                fleece_depth_cm=fleece_depth_cm,
                is_shorn=is_shorn,
            )

        if result.get("status") == "REJECTED":
            return WeightPredictionResponse(
                status="REJECTED",
                species="SHEEP",
                rejection_reason=result.get("rejection_reason", "Unable to estimate reliably"),
                recommendation="Provide an unobstructed lateral capture of the sheep.",
                is_out_of_distribution=True,
                metadata={"species_detection": det_meta},
            )

        resp = WeightPredictionResponse(
            status="SUCCESS",
            species="SHEEP",
            estimated_weight_kg=result["estimated_weight_kg"],
            prediction_intervals=result["prediction_intervals"],
            reliability_status=result["reliability_status"],
            is_out_of_distribution=result["is_out_of_distribution"],
            metadata={
                "fleece_adjusted": result.get("fleece_adjusted"),
                "fleece_depth_cm": result.get("fleece_depth_cm"),
                "is_shorn": result.get("is_shorn"),
                "deducted_fleece_kg": result.get("deducted_fleece_kg"),
                "morphometrics": result.get("morphometrics"),
                "species_detection": det_meta,
            },
        )

    # Log successful prediction event
    if resp and resp.status == "SUCCESS":
        c90 = resp.prediction_intervals.get("90%") if resp.prediction_intervals else None
        log_prediction_audit(
            species=resp.species,
            prediction_type="WEIGHT",
            quality_status="PASSED",
            status="SUCCESS",
            point_estimate=resp.estimated_weight_kg,
            conf_90_lower=c90["lower"] if c90 else None,
            conf_90_upper=c90["upper"] if c90 else None,
        )
    return resp


@router.post("/age")
async def estimate_age(
    species: Species = Form(..., description="Target animal species"),
    dentition_incisor_pairs: Optional[int] = Form(None, ge=0, le=4, description="Permanent incisor pairs (0 to 4)"),
    weeks_since_hatch: Optional[float] = Form(None, ge=0.0, description="Weeks since hatch (chickens only)"),
    file: Optional[UploadFile] = File(None, description="Optional photo upload (oral dental close-up for ruminants, bird photo for poultry)"),
    is_oral_dental_capture: bool = Form(False, description="Whether photo is an anterior dental inspection"),
    age_pipeline: Any = Depends(get_unified_age_pipeline),
    _auth: bool = Depends(verify_api_key),
):
    """
    Veterinary age estimation endpoint:
    - Ruminants (Cattle, Goats, Sheep): Automatically detects incisor tooth pairs from oral image,
      or accepts veterinary dentition input. Rejects non-dental body photos.
    - Poultry (Hens): Automatically estimates life stage and weeks from bird photograph,
      or accepts flock hatch cohort placement records.
    """
    contents = None
    if file:
        contents = await file.read()

    res = age_pipeline.estimate_age(
        species=species,
        dentition_incisor_pairs=dentition_incisor_pairs,
        weeks_since_hatch=weeks_since_hatch,
        is_oral_dental_capture=is_oral_dental_capture,
        image_input=contents,
    )
    return res.model_dump()
