package com.livestockvision.ai

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File

/**
 * Repository layer handling business logic, remote API dispatch,
 * and offline SQLite capture queue persistence.
 */
class LivestockRepository(
    private val apiService: LivestockApiService,
    private val captureDao: OfflineCaptureDao
) {

    suspend fun checkHealth(): Result<HealthResponse> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.getHealth()
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                Result.failure(Exception("Health check failed with HTTP ${response.code()}: ${response.message()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun estimateWeight(
        primaryFile: File,
        species: Species?,
        viewFiles: List<File> = emptyList(),
        referenceWeightKg: Float? = null,
        withersHeightCm: Float? = null
    ): Result<WeightPredictionResponse> = withContext(Dispatchers.IO) {
        try {
            val primaryPart = createMultipartPart("file", primaryFile)
            val view1Part = viewFiles.getOrNull(0)?.let { createMultipartPart("view_1", it) }
            val view2Part = viewFiles.getOrNull(1)?.let { createMultipartPart("view_2", it) }
            val view3Part = viewFiles.getOrNull(2)?.let { createMultipartPart("view_3", it) }

            val speciesBody = species?.name?.toRequestBody("text/plain".toMediaTypeOrNull())
            val refWeightBody = referenceWeightKg?.toString()?.toRequestBody("text/plain".toMediaTypeOrNull())
            val withersBody = withersHeightCm?.toString()?.toRequestBody("text/plain".toMediaTypeOrNull())

            val response = apiService.predictWeight(
                file = primaryPart,
                view1 = view1Part,
                view2 = view2Part,
                view3 = view3Part,
                species = speciesBody,
                referenceWeightKg = refWeightBody,
                withersHeightCm = withersBody
            )

            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                Result.failure(Exception("Weight prediction error ${response.code()}: ${response.errorBody()?.string()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun enqueueOfflineCapture(
        animalId: String,
        species: String,
        viewPaths: List<String>
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val entity = OfflineCaptureEntity(
                captureId = "cap_${System.currentTimeMillis()}_${(1000..9999).random()}",
                animalId = animalId,
                species = species,
                view0Path = viewPaths.firstOrNull() ?: "",
                view1Path = viewPaths.getOrNull(1),
                view2Path = viewPaths.getOrNull(2),
                view3Path = viewPaths.getOrNull(3),
                isSynced = false
            )
            captureDao.insertCapture(entity)
            Result.success(entity.captureId)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun createMultipartPart(partName: String, file: File): MultipartBody.Part {
        val requestBody = file.asRequestBody("image/jpeg".toMediaTypeOrNull())
        return MultipartBody.Part.createFormData(partName, file.name, requestBody)
    }

    companion object {
        fun create(context: Context): LivestockRepository {
            val db = AppDatabase.getDatabase(context)
            val api = ApiClient.getService()
            return LivestockRepository(api, db.captureDao())
        }
    }
}
