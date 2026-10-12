package com.livestockvision.ai

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraCaptureActivity : AppCompatActivity() {

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    private var currentViewIndex = 0
    private val viewNames = listOf("LATERAL_LEFT", "LATERAL_RIGHT", "CRANIAL", "CAUDAL")
    private val capturedFiles = mutableListOf<File>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Live Quality HUD Analyzer (Simulates Laplacian sharpness check)
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        val sharpness = evaluateSharpness(imageProxy)
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageCapture, imageAnalyzer
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun evaluateSharpness(image: ImageProxy): Float {
        // Simplified variance of pixel gradients on Y buffer
        val buffer = image.planes[0].buffer
        val data = ByteArray(buffer.remaining())
        buffer.get(data)
        var sum = 0L
        for (i in 0 until data.size step 8) {
            val v = data[i].toInt() and 0xFF
            sum += v
        }
        val mean = sum.toFloat() / (data.size / 8)
        var varSum = 0.0
        for (i in 0 until data.size step 8) {
            val v = (data[i].toInt() and 0xFF) - mean
            varSum += v * v
        }
        return (varSum / (data.size / 8)).toFloat()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun dispatchToPredictionApi(file: File) {
        val repo = LivestockRepository.create(this)
        lifecycleScope.launch {
            val result = repo.estimateWeight(primaryFile = file, species = null)
            val intent = android.content.Intent(this@CameraCaptureActivity, PredictionResultActivity::class.java)
            if (result.isSuccess) {
                val resp = result.getOrNull()
                intent.putExtra("EXTRA_STATUS", resp?.status ?: "SUCCESS")
                intent.putExtra("EXTRA_SPECIES", resp?.species ?: "COW")
                intent.putExtra("EXTRA_WEIGHT_KG", resp?.estimatedWeightKg ?: 0f)
                val c90 = resp?.predictionIntervals?.get("90%")
                intent.putExtra("EXTRA_LOWER_90", c90?.lower ?: 0f)
                intent.putExtra("EXTRA_UPPER_90", c90?.upper ?: 0f)
                val c95 = resp?.predictionIntervals?.get("95%")
                intent.putExtra("EXTRA_LOWER_95", c95?.lower ?: 0f)
                intent.putExtra("EXTRA_UPPER_95", c95?.upper ?: 0f)
                intent.putExtra("EXTRA_WARNING", resp?.warningMessage)
                intent.putExtra("EXTRA_REJECTION_REASON", resp?.rejectionReason)
            } else {
                intent.putExtra("EXTRA_STATUS", "REJECTED")
                intent.putExtra("EXTRA_REJECTION_REASON", result.exceptionOrNull()?.message)
            }
            startActivity(intent)
            finish()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
