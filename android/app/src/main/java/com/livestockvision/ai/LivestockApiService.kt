package com.livestockvision.ai

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.*

/**
 * Retrofit interface mapping the 8 canonical LivestockVisionAI backend endpoints.
 */
interface LivestockApiService {

    // 1. System Health
    @GET("health")
    suspend fun getHealth(): Response<HealthResponse>

    // 2. Image Quality Gate
    @Multipart
    @POST("api/v1/quality/check")
    suspend fun checkImageQuality(
        @Part file: MultipartBody.Part
    ): Response<QualityCheckResult>

    // 3. Species Identification
    @Multipart
    @POST("api/v1/predict/species")
    suspend fun predictSpecies(
        @Part file: MultipartBody.Part
    ): Response<SpeciesPredictionResponse>

    // 4. Weight Estimation (Multi-View & Morphometrics)
    @Multipart
    @POST("api/v1/predict/weight")
    suspend fun predictWeight(
        @Part file: MultipartBody.Part,
        @Part view1: MultipartBody.Part? = null,
        @Part view2: MultipartBody.Part? = null,
        @Part view3: MultipartBody.Part? = null,
        @Part("species") species: RequestBody? = null,
        @Part("reference_weight_kg") referenceWeightKg: RequestBody? = null,
        @Part("withers_height_cm") withersHeightCm: RequestBody? = null,
        @Part("body_length_cm") bodyLengthCm: RequestBody? = null,
        @Part("heart_girth_cm") heartGirthCm: RequestBody? = null,
        @Part("fleece_depth_cm") fleeceDepthCm: RequestBody? = null,
        @Part("is_shorn") isShorn: RequestBody? = null,
        @Part("bird_type") birdType: RequestBody? = null,
        @Part("plumage_condition") plumageCondition: RequestBody? = null
    ): Response<WeightPredictionResponse>

    // 5. Age Estimation (Dentition & Cohort)
    @Multipart
    @POST("api/v1/predict/age")
    suspend fun predictAge(
        @Part("species") species: RequestBody,
        @Part("dentition_incisor_pairs") dentitionPairs: RequestBody? = null,
        @Part("weeks_since_hatch") weeksSinceHatch: RequestBody? = null,
        @Part file: MultipartBody.Part? = null,
        @Part("is_oral_dental_capture") isOralCapture: RequestBody? = null
    ): Response<AgePredictionResponse>

    // 6. Animal Registry
    @POST("api/v1/animals/")
    suspend fun registerAnimal(
        @Body record: AnimalRecordDto
    ): Response<AnimalRecordDto>

    @GET("api/v1/animals/")
    suspend fun listAnimals(
        @Query("species") species: String? = null,
        @Query("limit") limit: Int = 50,
        @Query("offset") offset: Int = 0
    ): Response<List<AnimalRecordDto>>

    // 7. Measurement History
    @POST("api/v1/animals/{animal_id}/measurements")
    suspend fun recordMeasurement(
        @Path("animal_id") animalId: String,
        @Body measurement: MeasurementDto
    ): Response<MeasurementDto>

    // 8. Offline Queue Reconciliation
    @POST("api/v1/animals/sync/queue")
    suspend fun syncOfflineQueue(
        @Body batch: SyncBatchRequest
    ): Response<SyncBatchResponse>
}
