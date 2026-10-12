"""
Persistent SQLite Database Layer for LivestockVisionAI.
Stores animal records, certified measurements, and prediction appraisal audit logs.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

from LivestockVisionAI.data.schemas.animal_record import AnimalRecord, Species, Sex, HealthStatus
from LivestockVisionAI.data.schemas.measurement_record import MeasurementRecord, MeasurementType, MeasurementUnit


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "livestock_platform.db"


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Animals Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS animals (
                animal_id TEXT PRIMARY KEY,
                species TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                breed TEXT,
                sex TEXT,
                health_status TEXT,
                scale_weight_kg REAL,
                dentition_incisor_pairs INTEGER,
                verified_dob TEXT,
                verified_age_months REAL,
                scale_calibration_id TEXT,
                scale_measurement_timestamp TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Measurements Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id TEXT PRIMARY KEY,
                animal_id TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                device_model TEXT,
                calibration_certificate_id TEXT,
                operator_name TEXT,
                timestamp TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (animal_id) REFERENCES animals (animal_id)
            )
        """)

        # Prediction Audit Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                species TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                laplacian_score REAL,
                brightness_mean REAL,
                point_estimate REAL,
                conf_90_lower REAL,
                conf_90_upper REAL,
                status TEXT NOT NULL,
                rejection_reason TEXT
            )
        """)
        conn.commit()


# Initialize tables upon module import
init_db()


def save_animal_to_db(animal: AnimalRecord) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO animals (
                animal_id, species, farm_id, breed, sex, health_status,
                scale_weight_kg, dentition_incisor_pairs, verified_dob,
                verified_age_months, scale_calibration_id,
                scale_measurement_timestamp, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            animal.animal_id,
            animal.species.value,
            animal.farm_id,
            animal.breed,
            animal.sex.value,
            animal.health_status.value,
            animal.scale_weight_kg,
            animal.dentition_incisor_pairs,
            animal.verified_dob,
            animal.verified_age_months,
            animal.scale_calibration_id,
            animal.scale_measurement_timestamp,
            animal.notes,
            now_iso,
        ))
        conn.commit()


def update_animal_in_db(animal: AnimalRecord) -> bool:
    clean_id = animal.animal_id.strip().upper()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT animal_id FROM animals WHERE animal_id = ?", (clean_id,))
        if not cursor.fetchone():
            return False
        cursor.execute("""
            UPDATE animals SET
                species = ?, farm_id = ?, breed = ?, sex = ?, health_status = ?,
                scale_weight_kg = ?, dentition_incisor_pairs = ?, verified_dob = ?,
                verified_age_months = ?, scale_calibration_id = ?,
                scale_measurement_timestamp = ?, notes = ?
            WHERE animal_id = ?
        """, (
            animal.species.value,
            animal.farm_id,
            animal.breed,
            animal.sex.value,
            animal.health_status.value,
            animal.scale_weight_kg,
            animal.dentition_incisor_pairs,
            animal.verified_dob,
            animal.verified_age_months,
            animal.scale_calibration_id,
            animal.scale_measurement_timestamp,
            animal.notes,
            clean_id,
        ))
        conn.commit()
        return True


def delete_animal_from_db(animal_id: str) -> bool:
    clean_id = animal_id.strip().upper()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT animal_id FROM animals WHERE animal_id = ?", (clean_id,))
        if not cursor.fetchone():
            return False
        cursor.execute("DELETE FROM measurements WHERE animal_id = ?", (clean_id,))
        cursor.execute("DELETE FROM animals WHERE animal_id = ?", (clean_id,))
        conn.commit()
        return True


def get_animal_from_db(animal_id: str) -> Optional[AnimalRecord]:
    clean_id = animal_id.strip().upper()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM animals WHERE animal_id = ?", (clean_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return AnimalRecord(
            animal_id=row["animal_id"],
            species=Species(row["species"]),
            farm_id=row["farm_id"],
            breed=row["breed"],
            sex=Sex(row["sex"]) if row["sex"] else Sex.UNKNOWN,
            health_status=HealthStatus(row["health_status"]) if row["health_status"] else HealthStatus.HEALTHY,
            scale_weight_kg=row["scale_weight_kg"],
            dentition_incisor_pairs=row["dentition_incisor_pairs"],
            verified_dob=row["verified_dob"],
            verified_age_months=row["verified_age_months"],
            scale_calibration_id=row["scale_calibration_id"],
            scale_measurement_timestamp=row["scale_measurement_timestamp"],
            notes=row["notes"],
        )


def list_animals_from_db(
    species: Optional[Species] = None,
    farm_id: Optional[str] = None,
    breed: Optional[str] = None,
) -> List[AnimalRecord]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM animals WHERE 1=1"
        params: List[Any] = []

        if species:
            query += " AND species = ?"
            params.append(species.value)
        if farm_id:
            query += " AND farm_id = ?"
            params.append(farm_id)
        if breed:
            query += " AND breed LIKE ?"
            params.append(f"%{breed}%")

        query += " ORDER BY animal_id"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append(AnimalRecord(
                animal_id=row["animal_id"],
                species=Species(row["species"]),
                farm_id=row["farm_id"],
                breed=row["breed"],
                sex=Sex(row["sex"]) if row["sex"] else Sex.UNKNOWN,
                health_status=HealthStatus(row["health_status"]) if row["health_status"] else HealthStatus.HEALTHY,
                scale_weight_kg=row["scale_weight_kg"],
                dentition_incisor_pairs=row["dentition_incisor_pairs"],
                verified_dob=row["verified_dob"],
                verified_age_months=row["verified_age_months"],
                scale_calibration_id=row["scale_calibration_id"],
                scale_measurement_timestamp=row["scale_measurement_timestamp"],
                notes=row["notes"],
            ))
        return results


def save_measurement_to_db(measurement: MeasurementRecord) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO measurements (
                measurement_id, animal_id, measurement_type, value, unit,
                device_model, calibration_certificate_id, operator_name,
                timestamp, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            measurement.measurement_id,
            measurement.animal_id,
            measurement.measurement_type.value,
            measurement.value,
            measurement.unit.value,
            measurement.device_model,
            measurement.calibration_certificate_id,
            measurement.operator_name,
            measurement.timestamp,
            measurement.notes,
        ))
        conn.commit()


def get_measurements_from_db(animal_id: str) -> List[MeasurementRecord]:
    clean_id = animal_id.strip().upper()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM measurements WHERE animal_id = ? ORDER BY timestamp DESC", (clean_id,))
        rows = cursor.fetchall()
        return [
            MeasurementRecord(
                measurement_id=r["measurement_id"],
                animal_id=r["animal_id"],
                measurement_type=MeasurementType(r["measurement_type"]),
                value=r["value"],
                unit=MeasurementUnit(r["unit"]),
                device_model=r["device_model"],
                calibration_certificate_id=r["calibration_certificate_id"],
                operator_name=r["operator_name"],
                timestamp=r["timestamp"],
                notes=r["notes"],
            )
            for r in rows
        ]


def log_prediction_audit(
    species: str,
    prediction_type: str,
    quality_status: str,
    status: str,
    laplacian_score: Optional[float] = None,
    brightness_mean: Optional[float] = None,
    point_estimate: Optional[float] = None,
    conf_90_lower: Optional[float] = None,
    conf_90_upper: Optional[float] = None,
    rejection_reason: Optional[str] = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_audit_logs (
                timestamp, species, prediction_type, quality_status,
                laplacian_score, brightness_mean, point_estimate,
                conf_90_lower, conf_90_upper, status, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso, species, prediction_type, quality_status,
            laplacian_score, brightness_mean, point_estimate,
            conf_90_lower, conf_90_upper, status, rejection_reason
        ))
        conn.commit()


def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prediction_audit_logs ORDER BY log_id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


def sync_offline_batch_to_db(captures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconciles a batch of offline field captures from edge/Android nodes
    into the central animal registry and measurement database.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    synced = []
    failed = []

    for item in captures:
        queue_id = item.get("queue_id", str(uuid.uuid4()))
        animal_id = (item.get("animal_id") or f"EDGE-{queue_id[:8]}").strip().upper()
        species_str = item.get("species_hint") or item.get("species") or "COW"
        farm_id = item.get("farm_id", "OFFLINE_FIELD_UNKNOWN")
        weight_val = item.get("local_weight_est_kg") or item.get("scale_weight_kg")

        try:
            # Check if animal exists
            existing = get_animal_from_db(animal_id)
            if not existing:
                try:
                    spec_enum = Species(species_str.upper())
                except Exception:
                    spec_enum = Species.COW

                new_animal = AnimalRecord(
                    animal_id=animal_id,
                    species=spec_enum,
                    farm_id=farm_id,
                    scale_weight_kg=weight_val if weight_val and weight_val > 0 else None,
                    notes=f"Auto-enrolled from edge capture queue (ID: {queue_id})",
                )
                save_animal_to_db(new_animal)

            # Record measurement if weight present
            if weight_val and float(weight_val) > 0:
                meas = MeasurementRecord(
                    measurement_id=f"M-SYNC-{uuid.uuid4().hex[:8]}",
                    animal_id=animal_id,
                    measurement_type=MeasurementType.WEIGHT_SCALE,
                    value=float(weight_val),
                    unit=MeasurementUnit.KG,
                    operator_name=item.get("operator_name", "Edge Field Node"),
                    device_model="OfflineEdgeCaptureSync",
                    timestamp=now_iso,
                    notes=f"Synced from edge queue {queue_id}",
                )
                save_measurement_to_db(meas)

            synced.append({
                "queue_id": queue_id,
                "animal_id": animal_id,
                "status": "SYNCED",
            })
        except Exception as e:
            failed.append({
                "queue_id": queue_id,
                "animal_id": animal_id,
                "status": "FAILED",
                "error": str(e),
            })

    return {
        "total_submitted": len(captures),
        "synced_count": len(synced),
        "failed_count": len(failed),
        "synced_items": synced,
        "failed_items": failed,
    }
