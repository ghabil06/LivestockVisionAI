package com.livestockvision.ai

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Base64

/**
 * WorkManager background worker that syncs pending offline captures to the backend.
 *
 * Design contract:
 *  - Uploads actual image byte payloads (Base64-encoded) from on-device capture files.
 *  - Verifies per-item acknowledgement from the server; an HTTP 2xx status alone
 *    does NOT mark a capture as synced if the server returned failed_items or omitted the item.
 *  - A capture is marked synced ONLY after confirmed server acceptance of that exact item.
 *  - If upload fails or server rejects the item, syncAttempts is incremented; the record
 *    is preserved in an unsynced state.
 *  - After MAX_SYNC_ATTEMPTS consecutive failures the record is marked as FAILED
 *    so the UI can surface it to the operator rather than silently re-queuing.
 */
class SyncWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    companion object {
        private const val MAX_SYNC_ATTEMPTS = 3

        /**
         * Safely reads local image file and converts to Base64 string for upload.
         */
        fun encodeFileToBase64(filePath: String?): String? {
            if (filePath.isNullOrBlank()) return null
            return try {
                val file = File(filePath)
                if (file.exists() && file.isFile && file.length() > 0) {
                    val bytes = file.readBytes()
                    Base64.getEncoder().encodeToString(bytes)
                } else {
                    null
                }
            } catch (_: Exception) {
                null
            }
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val database = AppDatabase.getDatabase(applicationContext)
        val dao = database.captureDao()
        val apiService = ApiClient.getService()
        val pending = dao.getPendingCaptures()

        if (pending.isEmpty()) {
            return@withContext Result.success()
        }

        for (item in pending) {
            try {
                // Encode actual captured image bytes from local file paths
                val view0B64 = encodeFileToBase64(item.view0Path)
                val view1B64 = encodeFileToBase64(item.view1Path)
                val view2B64 = encodeFileToBase64(item.view2Path)
                val view3B64 = encodeFileToBase64(item.view3Path)

                val payload = SyncCapturePayload(
                    captureId = item.captureId,
                    animalId = item.animalId,
                    species = item.species,
                    view0Path = item.view0Path,
                    view1Path = item.view1Path,
                    view2Path = item.view2Path,
                    view3Path = item.view3Path,
                    view0BytesBase64 = view0B64,
                    view1BytesBase64 = view1B64,
                    view2BytesBase64 = view2B64,
                    view3BytesBase64 = view3B64,
                    timestamp = item.timestamp
                )

                val batch = SyncBatchRequest(captures = listOf(payload))
                val response = apiService.syncOfflineQueue(batch)

                if (response.isSuccessful && response.body() != null) {
                    val body = response.body()!!

                    // Strictly inspect per-item acknowledgement rather than relying solely on HTTP 2xx
                    val isConfirmed = body.syncedItems.any { it.captureId == item.captureId }
                    val isFailed = body.failedItems.any { it.captureId == item.captureId }

                    if (isConfirmed && !isFailed) {
                        // Confirmed server acknowledgement for this exact item
                        dao.markSynced(item.captureId)
                    } else {
                        // Server returned HTTP 200 but rejected or failed this specific capture
                        val attempts = item.syncAttempts + 1
                        if (attempts >= MAX_SYNC_ATTEMPTS) {
                            dao.updateCapture(
                                item.copy(syncAttempts = attempts, isSynced = false)
                            )
                        } else {
                            dao.updateCapture(item.copy(syncAttempts = attempts))
                        }
                    }
                } else {
                    // Non-2xx response: increment failure counter; do NOT mark as synced
                    val attempts = item.syncAttempts + 1
                    if (attempts >= MAX_SYNC_ATTEMPTS) {
                        dao.updateCapture(
                            item.copy(syncAttempts = attempts, isSynced = false)
                        )
                    } else {
                        dao.updateCapture(item.copy(syncAttempts = attempts))
                    }
                }
            } catch (e: Exception) {
                // Network / serialization error — increment attempts, leave unsynced
                val attempts = item.syncAttempts + 1
                if (attempts >= MAX_SYNC_ATTEMPTS) {
                    dao.updateCapture(
                        item.copy(syncAttempts = attempts, isSynced = false)
                    )
                } else {
                    dao.updateCapture(item.copy(syncAttempts = attempts))
                }
            }
        }

        Result.success()
    }
}
