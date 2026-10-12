"""
Animals Registry Router.
Provides CRUD endpoints for managing animal profiles, certified measurements,
and offline sync queue reconciliation backed by persistent SQLite.
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel

from LivestockVisionAI.data.schemas.animal_record import AnimalRecord, Species
from LivestockVisionAI.data.schemas.measurement_record import MeasurementRecord
from LivestockVisionAI.api.database import (
    save_animal_to_db,
    update_animal_in_db,
    delete_animal_from_db,
    get_animal_from_db,
    list_animals_from_db,
    save_measurement_to_db,
    get_measurements_from_db,
    get_recent_audit_logs,
    sync_offline_batch_to_db,
)
from LivestockVisionAI.api.security import verify_api_key


router = APIRouter(prefix="/api/v1/animals", tags=["Animal Registry"])


class SyncQueueBatchRequest(BaseModel):
    captures: List[Dict[str, Any]]


class SyncQueueBatchResponse(BaseModel):
    total_submitted: int
    synced_count: int
    failed_count: int
    synced_items: List[Dict[str, Any]]
    failed_items: List[Dict[str, Any]]


@router.post("", response_model=AnimalRecord, status_code=status.HTTP_201_CREATED, include_in_schema=False)
@router.post("/", response_model=AnimalRecord, status_code=status.HTTP_201_CREATED)
async def register_animal(animal: AnimalRecord, _auth: bool = Depends(verify_api_key)):
    """Enroll a new animal into the platform registry."""
    existing = get_animal_from_db(animal.animal_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Animal with ID '{animal.animal_id}' already registered."
        )
    save_animal_to_db(animal)
    return animal


@router.get("/{animal_id}", response_model=AnimalRecord)
async def get_animal(animal_id: str):
    """Fetch an animal's profile by ID."""
    clean_id = animal_id.strip().upper()
    animal = get_animal_from_db(clean_id)
    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Animal '{clean_id}' not found."
        )
    return animal


@router.put("/{animal_id}", response_model=AnimalRecord)
async def update_animal(animal_id: str, animal: AnimalRecord):
    """Update an existing animal's profile."""
    clean_id = animal_id.strip().upper()
    if animal.animal_id != clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path animal_id '{clean_id}' does not match body animal_id '{animal.animal_id}'"
        )
    success = update_animal_in_db(animal)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Animal '{clean_id}' not found."
        )
    return animal


@router.delete("/{animal_id}", status_code=status.HTTP_200_OK)
async def delete_animal(animal_id: str):
    """Delete an animal and all its associated measurements."""
    clean_id = animal_id.strip().upper()
    success = delete_animal_from_db(clean_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Animal '{clean_id}' not found."
        )
    return {"status": "DELETED", "animal_id": clean_id}


@router.get("/", response_model=List[AnimalRecord])
async def list_animals(
    species: Optional[Species] = None,
    farm_id: Optional[str] = Query(None, description="Filter by farm identifier"),
    breed: Optional[str] = Query(None, description="Filter by breed keyword"),
):
    """List registered animals, with optional filters for species, farm, and breed."""
    return list_animals_from_db(species=species, farm_id=farm_id, breed=breed)


@router.post("/{animal_id}/measurements", response_model=MeasurementRecord, status_code=status.HTTP_201_CREATED)
async def record_measurement(animal_id: str, measurement: MeasurementRecord, _auth: bool = Depends(verify_api_key)):
    """Add a certified ground-truth measurement for an animal."""
    clean_id = animal_id.strip().upper()
    animal = get_animal_from_db(clean_id)
    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Animal '{clean_id}' not found."
        )
    if measurement.animal_id != clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Measurement animal_id '{measurement.animal_id}' does not match path ID '{clean_id}'"
        )
    save_measurement_to_db(measurement)
    return measurement


@router.get("/{animal_id}/measurements", response_model=List[MeasurementRecord])
async def get_animal_measurements(animal_id: str):
    """List all certified measurements recorded for an animal."""
    clean_id = animal_id.strip().upper()
    animal = get_animal_from_db(clean_id)
    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Animal '{clean_id}' not found."
        )
    return get_measurements_from_db(clean_id)


@router.get("/audit/logs", tags=["Audit"])
async def query_audit_logs(limit: int = 50):
    """Retrieve historical prediction appraisal audit logs."""
    return get_recent_audit_logs(limit=limit)


@router.post("/sync/queue", response_model=SyncQueueBatchResponse, tags=["Offline Sync"])
async def sync_offline_queue_batch(request: SyncQueueBatchRequest, _auth: bool = Depends(verify_api_key)):
    """
    Synchronizes a batch of offline field captures from mobile/edge devices into central registry.
    """
    res = sync_offline_batch_to_db(request.captures)
    return SyncQueueBatchResponse(**res)
