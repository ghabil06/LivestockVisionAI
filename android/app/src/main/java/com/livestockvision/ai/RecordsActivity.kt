package com.livestockvision.ai

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Screen 4: Animal Records & Sync Screen.
 * Displays saved animal records, offline capture queue buffer,
 * and background synchronization triggers.
 */
class RecordsActivity : AppCompatActivity() {

    private lateinit var database: AppDatabase
    private lateinit var textQueueStatus: TextView
    private lateinit var textAnimalList: TextView
    private lateinit var btnSyncNow: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_records)

        database = AppDatabase.getDatabase(this)
        textQueueStatus = findViewById(R.id.text_queue_status)
        textAnimalList = findViewById(R.id.text_animal_list)
        btnSyncNow = findViewById(R.id.btn_sync_now)

        loadRecordsAndQueue()

        btnSyncNow.setOnClickListener {
            triggerBackgroundSync()
        }
    }

    private fun loadRecordsAndQueue() {
        lifecycleScope.launch {
            val pendingCaptures = withContext(Dispatchers.IO) {
                database.captureDao().getPendingCaptures()
            }
            textQueueStatus.text = "Pending Offline Captures: ${pendingCaptures.size}"

            // Fetch animals from API repository if available
            try {
                val api = ApiClient.getService()
                val response = withContext(Dispatchers.IO) { api.listAnimals(limit = 10) }
                if (response.isSuccessful && response.body() != null) {
                    val animals = response.body()!!
                    val sb = StringBuilder()
                    for (a in animals) {
                        sb.append("• [${a.species}] ID: ${a.animalId} | Breed: ${a.breed ?: "N/A"}\n")
                    }
                    textAnimalList.text = if (sb.isNotEmpty()) sb.toString() else "No registered animals found."
                } else {
                    textAnimalList.text = "Offline: Showing cached records."
                }
            } catch (e: Exception) {
                textAnimalList.text = "Backend unreachable (working offline)."
            }
        }
    }

    private fun triggerBackgroundSync() {
        val syncRequest = OneTimeWorkRequestBuilder<SyncWorker>().build()
        WorkManager.getInstance(this).enqueue(syncRequest)
        Toast.makeText(this, "Background sync initiated via WorkManager", Toast.LENGTH_SHORT).show()
        loadRecordsAndQueue()
    }
}
