package com.livestockvision.ai

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Screen 3: AI Results Screen.
 * Displays real API predictions, conformal uncertainty intervals,
 * domain-honest age limitations, and quality rejection feedback.
 */
class PredictionResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_prediction_result)

        val statusText = findViewById<TextView>(R.id.text_status)
        val speciesText = findViewById<TextView>(R.id.text_species)
        val weightText = findViewById<TextView>(R.id.text_weight)
        val interval90Text = findViewById<TextView>(R.id.text_interval_90)
        val interval95Text = findViewById<TextView>(R.id.text_interval_95)
        val ageText = findViewById<TextView>(R.id.text_age)
        val warningText = findViewById<TextView>(R.id.text_warning)
        val rejectionText = findViewById<TextView>(R.id.text_rejection)
        val btnDone = findViewById<Button>(R.id.btn_done)

        // Parse extras from Intent
        val status = intent.getStringExtra("EXTRA_STATUS") ?: "SUCCESS"
        val species = intent.getStringExtra("EXTRA_SPECIES") ?: "UNKNOWN"
        val weightKg = intent.getFloatExtra("EXTRA_WEIGHT_KG", -1f)
        val lower90 = intent.getFloatExtra("EXTRA_LOWER_90", -1f)
        val upper90 = intent.getFloatExtra("EXTRA_UPPER_90", -1f)
        val lower95 = intent.getFloatExtra("EXTRA_LOWER_95", -1f)
        val upper95 = intent.getFloatExtra("EXTRA_UPPER_95", -1f)
        val ageRange = intent.getStringExtra("EXTRA_AGE_RANGE")
        val warning = intent.getStringExtra("EXTRA_WARNING")
        val rejectionReason = intent.getStringExtra("EXTRA_REJECTION_REASON")

        if (status == "REJECTED") {
            statusText.text = "Prediction Rejected"
            statusText.setTextColor(getColor(android.R.color.holo_red_dark))
            rejectionText.visibility = View.VISIBLE
            rejectionText.text = rejectionReason ?: "Unable to estimate reliably."
            weightText.text = "--"
            interval90Text.text = "--"
            interval95Text.text = "--"
        } else {
            statusText.text = "Inference Verified"
            statusText.setTextColor(getColor(android.R.color.holo_green_dark))
            rejectionText.visibility = View.GONE
            speciesText.text = "Species: $species"
            weightText.text = String.format("%.1f kg", weightKg)

            if (lower90 > 0 && upper90 > 0) {
                interval90Text.text = String.format("90%% Confidence: %.1f - %.1f kg", lower90, upper90)
                interval95Text.text = String.format("95%% Confidence: %.1f - %.1f kg", lower95, upper95)
            } else {
                interval90Text.text = "Conformal Bounds: Pending calibration"
                interval95Text.text = ""
            }

            // Domain-honest age handling
            if (!ageRange.isNullOrEmpty()) {
                ageText.text = "Estimated Age: $ageRange"
            } else {
                ageText.text = "Age: Requires dental exam (honest rejection for casual body photos)"
            }

            // Caution flag for heavy livestock (>300 kg)
            if (!warning.isNullOrEmpty()) {
                warningText.visibility = View.VISIBLE
                warningText.text = warning
            } else {
                warningText.visibility = View.GONE
            }
        }

        btnDone.setOnClickListener {
            finish()
        }
    }
}
