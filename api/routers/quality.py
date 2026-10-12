"""
Quality Verification API Router.
Provides pre-inference image qualification endpoints.
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException

from LivestockVisionAI.data.schemas.capture_metadata import QualityCheckResult
from LivestockVisionAI.models.shared.quality_filter import QualityFilter
from LivestockVisionAI.api.dependencies import get_quality_filter


router = APIRouter(prefix="/api/v1/quality", tags=["Image Quality"])


@router.post("/check", response_model=QualityCheckResult)
async def check_image_quality(
    file: UploadFile = File(..., description="Livestock image file to evaluate"),
    bbox_x1: Optional[int] = Query(None, description="Bounding box xmin"),
    bbox_y1: Optional[int] = Query(None, description="Bounding box ymin"),
    bbox_x2: Optional[int] = Query(None, description="Bounding box xmax"),
    bbox_y2: Optional[int] = Query(None, description="Bounding box ymax"),
    quality_filter: QualityFilter = Depends(get_quality_filter),
):
    """
    Evaluate an uploaded image capture against quality gates (blur, exposure, framing).
    Returns QualityCheckResult with pass/fail status and prescriptive advice.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    bbox = None
    if all(v is not None for v in [bbox_x1, bbox_y1, bbox_x2, bbox_y2]):
        bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

    try:
        result = quality_filter.evaluate(image_input=contents, bbox=bbox)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image: {str(e)}")
