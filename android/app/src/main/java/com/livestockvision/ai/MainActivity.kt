package com.livestockvision.ai

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Screen 1: Home Dashboard Screen.
 * Provides clear "Scan Animal" action, API connection health monitoring,
 * and quick access to Animal Records and Offline Capture Synchronization.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var tvApiStatus: TextView
    private lateinit var btnCapture: Button
    private lateinit var btnRecords: Button
    private lateinit var btnSync: Button
    private lateinit var repository: LivestockRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        repository = LivestockRepository.create(this)

        tvApiStatus = findViewById(R.id.tv_api_status)
        btnCapture = findViewById(R.id.btn_capture)
        btnRecords = findViewById(R.id.btn_records)
        btnSync = findViewById(R.id.btn_sync)

        btnCapture.setOnClickListener {
            val intent = Intent(this, CameraCaptureActivity::class.java)
            startActivity(intent)
        }

        btnRecords.setOnClickListener {
            val intent = Intent(this, RecordsActivity::class.java)
            startActivity(intent)
        }

        btnSync.setOnClickListener {
            val intent = Intent(this, RecordsActivity::class.java)
            startActivity(intent)
        }

        checkBackendHealth()
    }

    override fun onResume() {
        super.onResume()
        checkBackendHealth()
    }

    private fun checkBackendHealth() {
        lifecycleScope.launch {
            val result = repository.checkHealth()
            if (result.isSuccess) {
                val health = result.getOrNull()
                tvApiStatus.text = "API: CONNECTED (${health?.status ?: "HEALTHY"})"
                tvApiStatus.setBackgroundColor(0xFFD1E7DD.toInt())
                tvApiStatus.setTextColor(0xFF0F5132.toInt())
            } else {
                tvApiStatus.text = "API: OFFLINE (Operating locally)"
                tvApiStatus.setBackgroundColor(0xFFFFF3CD.toInt())
                tvApiStatus.setTextColor(0xFF664D03.toInt())
            }
        }
    }
}
