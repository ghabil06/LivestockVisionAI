# LivestockVisionAI — Android Mobile Client

Native Android client for field livestock appraisal, automated CameraX multi-view capture, on-device ONNX Runtime Mobile INT8 inference, and offline SQLite synchronization.

---

## 1. Features
- **CameraX Multi-View Capture**: Guided step-by-step 4-view capture (Lateral Left, Lateral Right, Cranial, Caudal/Dorsal).
- **On-Device Quality Screening**: Fast real-time Laplacian blur & exposure detection before storing or transmitting images.
- **ONNX Runtime Mobile (INT8 Quantized)**:
  - `species_classifier_int8.onnx` (10.7 MB)
  - `cattle_weight_regressor_int8.onnx` (10.9 MB)
  - `cattle_multiview_weight_regressor_int8.onnx` (10.9 MB)
- **Offline-First Persistence**: Room SQLite captures queue with WorkManager background sync retry when mobile data is reconnected.
- **Conformal Uncertainty Bounds**: Displays 90% and 95% confidence intervals directly in the mobile UI.

---

## 2. Directory Layout
```
android/
├── build.gradle.kts           # Top-level Gradle configuration
├── settings.gradle.kts        # Repository and module settings
├── gradle/
│   └── wrapper/
│       └── gradle-wrapper.properties
└── app/
    ├── build.gradle.kts       # Android app module config (SDK 28-35)
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/            # Bundled INT8 quantized ONNX models
        │   ├── species_classifier_int8.onnx
        │   ├── cattle_weight_regressor_int8.onnx
        │   └── cattle_multiview_weight_regressor_int8.onnx
        ├── java/com/livestockvision/ai/
        │   ├── MainActivity.kt          # Primary dashboard & live status
        │   ├── CameraCaptureActivity.kt # CameraX multi-view capture UI
        │   ├── InferenceEngine.kt       # OnnxRuntime ONNX inference engine
        │   ├── Models.kt                # Data models & schemas
        │   ├── OfflineCaptureQueue.kt   # Room database offline buffer
        │   └── SyncWorker.kt            # WorkManager background sync
        └── res/
            ├── layout/activity_main.xml
            └── values/ (colors, strings, themes)
```

---

## 3. Building the Project

### Prerequisites
- Android Studio Hedgehog (2023.1.1) or newer
- Android SDK 35 (Android 15) installed
- JDK 17 or JDK 21

### Build via Command Line
```bash
cd android
./gradlew assembleDebug
```
The compiled debug APK will be generated at:
`app/build/outputs/apk/debug/app-debug.apk`

---

## 4. Hardware Acceleration
The app configures ONNX Runtime with NNAPI (`nnapi_flags`) to utilize hardware NPUs (Qualcomm Snapdragon NPU, MediaTek APU, Google Tensor TPU) on supported hardware. On devices without dedicated NPU drivers, it transparently falls back to optimized multi-threaded ARM CPU execution.
