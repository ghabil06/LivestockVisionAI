package com.livestockvision.ai

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.google.gson.annotations.SerializedName

// ---------------------------------------------------------------------------
// 1. Enums
// ---------------------------------------------------------------------------

enum class Species {
    COW, HEN, GOAT, SHEEP, UNKNOWN
}

enum class QualityStatus {
    PASSED, REJECTED_BLUR, REJECTED_LIGHTING, REJECTED_BOUNDING_BOX, REJECTED_UNDEREXPOSED, REJECTED_OVEREXPOSED
}

// ---------------------------------------------------------------------------
// 2. Health & System DTOs
// ---------------------------------------------------------------------------

data class HealthResponse(
    val status: String,
    val platform: String,
    val version: String,
    @SerializedName("supported_species") val supportedSpecies: List<String>,
    @SerializedName("quality_filter_active") val qualityFilterActive: Boolean,
    @SerializedName("conformal_uncertainty_active") val conformalUncertaintyActive: Boolean,
    val models: Map<String, Boolean>? = null
)

// ---------------------------------------------------------------------------
// 3. Quality & Prediction DTOs
// ---------------------------------------------------------------------------

data class QualityCheckResult(
    val passed: Boolean,
    val status: QualityStatus,
    @SerializedName("laplacian_variance") val laplacianVariance: Float,
    @SerializedName("brightness_mean") val brightnessMean: Float,
    @SerializedName("rejection_reason") val rejectionReason: String? = null,
    val recommendation: String? = null
)

data class SpeciesPredictionResponse(
    val status: String,
    @SerializedName("predicted_species") val predictedSpecies: String,
    val confidence: Float? = null,
    val probabilities: Map<String, Float>? = null,
    @SerializedName("is_out_of_distribution") val isOutOfDistribution: Boolean? = null,
    @SerializedName("rejection_reason") val rejectionReason: String? = null,
    val recommendation: String? = null
)

data class IntervalBounds(
    val lower: Float,
    val upper: Float
)

data class WeightPredictionResponse(
    val status: String, // "SUCCESS" or "REJECTED"
    val species: String,
    @SerializedName("rejection_reason") val rejectionReason: String? = null,
    val recommendation: String? = null,
    @SerializedName("estimated_weight_kg") val estimatedWeightKg: Float? = null,
    @SerializedName("estimated_weight_g") val estimatedWeightG: Float? = null,
    @SerializedName("prediction_intervals") val predictionIntervals: Map<String, IntervalBounds>? = null,
    @SerializedName("prediction_intervals_g") val predictionIntervalsG: Map<String, IntervalBounds>? = null,
    @SerializedName("reliability_status") val reliabilityStatus: String? = null,
    @SerializedName("is_out_of_distribution") val isOutOfDistribution: Boolean? = null,
    @SerializedName("warning_code") val warningCode: String? = null,
    @SerializedName("warning_message") val warningMessage: String? = null,
    val metadata: Map<String, Any>? = null
)

data class AgePredictionResponse(
    val status: String, // "SUCCESS" or "REJECTED"
    val species: String,
    @SerializedName("rejection_reason") val rejectionReason: String? = null,
    val recommendation: String? = null,
    @SerializedName("estimated_age_range_months") val estimatedAgeRangeMonths: List<Int>? = null,
    @SerializedName("point_estimate_months") val pointEstimateMonths: Float? = null,
    @SerializedName("dental_stage") val dentalStage: String? = null,
    @SerializedName("life_stage") val lifeStage: String? = null,
    val confidence: Float? = null,
    @SerializedName("is_visual_dental_derivation") val isVisualDentalDerivation: Boolean? = null
)

// ---------------------------------------------------------------------------
// 4. Animal Registry & Measurement History DTOs
// ---------------------------------------------------------------------------

data class AnimalRecordDto(
    @SerializedName("animal_id") val animalId: String,
    val species: String,
    @SerializedName("farm_id") val farmId: String,
    val breed: String? = null,
    val sex: String? = "UNKNOWN",
    @SerializedName("health_status") val healthStatus: String? = "HEALTHY",
    @SerializedName("verified_dob") val verifiedDob: String? = null,
    @SerializedName("verified_age_months") val verifiedAgeMonths: Float? = null,
    @SerializedName("dentition_incisor_pairs") val dentitionIncisorPairs: Int? = null,
    @SerializedName("scale_weight_kg") val scaleWeightKg: Float? = null,
    val notes: String? = null
)

data class MeasurementDto(
    @SerializedName("measurement_id") val measurementId: String? = null,
    @SerializedName("animal_id") val animalId: String,
    val timestamp: String? = null,
    @SerializedName("estimated_weight_kg") val estimatedWeightKg: Float,
    @SerializedName("conf_90_lower") val conf90Lower: Float? = null,
    @SerializedName("conf_90_upper") val conf90Upper: Float? = null,
    @SerializedName("views_count") val viewsCount: Int = 1,
    @SerializedName("quality_status") val qualityStatus: String = "PASSED"
)

// ---------------------------------------------------------------------------
// 5. Room SQLite Offline Persistence Entities
// ---------------------------------------------------------------------------

@Entity(tableName = "offline_captures")
data class OfflineCaptureEntity(
    @PrimaryKey val captureId: String,
    val animalId: String,
    val species: String,
    val view0Path: String,
    val view1Path: String? = null,
    val view2Path: String? = null,
    val view3Path: String? = null,
    val timestamp: Long = System.currentTimeMillis(),
    val isSynced: Boolean = false,
    val syncAttempts: Int = 0
)

data class SyncBatchRequest(
    val captures: List<OfflineCaptureEntity>
)

data class SyncBatchResponse(
    val status: String,
    @SerializedName("synced_count") val syncedCount: Int,
    @SerializedName("failed_count") val failedCount: Int
)
