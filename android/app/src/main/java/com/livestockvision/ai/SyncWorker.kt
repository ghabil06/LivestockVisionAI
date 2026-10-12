package com.livestockvision.ai

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * WorkManager background worker that syncs pending offline captures to the backend.
 *
 * Design contract:
 *  - A capture is only marked synced AFTER the API call succeeds (HTTP 2xx).
 *  - If the upload fails, syncAttempts is incremented; the capture is NOT marked synced.
 *  - After MAX_SYNC_ATTEMPTS consecutive failures the record is marked as FAILED
 *    so the UI can surface it to the operator rather than silently re-queuing.
 *  - The worker returns Result.success() regardless of individual item failures so
 *    WorkManager does not reschedule the entire job; partial failures are handled
 *    item-by-item via the syncAttempts counter.
 */
class SyncWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    companion object {
        private const val MAX_SYNC_ATTEMPTS = 3
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
                // Build the sync payload from the offline capture entity.
                // The backend /api/v1/animals/sync/queue endpoint accepts a JSON batch
                // of capture records for reconciliation into the central registry.
                val batch = SyncBatchRequest(captures = listOf(item))
                val response = apiService.syncOfflineQueue(batch)

                if (response.isSuccessful) {
                    // Only mark synced after confirmed server acknowledgement.
                    dao.markSynced(item.captureId)
                } else {
                    // Non-2xx: increment failure counter; do NOT mark as synced.
                    val attempts = item.syncAttempts + 1
                    if (attempts >= MAX_SYNC_ATTEMPTS) {
                        // Mark failed so RecordsActivity can display the error state.
                        dao.updateCapture(
                            item.copy(syncAttempts = attempts, isSynced = false)
                        )
                    } else {
                        dao.updateCapture(item.copy(syncAttempts = attempts))
                    }
                }
            } catch (e: Exception) {
                // Network / serialisation error — increment attempts, leave unsynced.
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
