package com.livestockvision.ai

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Singleton Retrofit HTTP client builder for LivestockVisionAI.
 *
 * Build-variant behaviour:
 *  - DEBUG builds: HTTP body logging is enabled; default URL points to the Android
 *    emulator host alias (10.0.2.2:8000) so developers can run the FastAPI server
 *    locally without additional configuration.
 *  - RELEASE builds: Body logging is disabled to prevent credential / payload leakage
 *    in production log aggregators; the base URL is sourced from BuildConfig so it
 *    can be set to the production HTTPS endpoint at build time without touching source.
 *
 * Note: BuildConfig.API_BASE_URL and BuildConfig.DEBUG are injected by the Gradle
 * buildConfigField declarations in android/app/build.gradle.kts.
 */
object ApiClient {

    // Fallback URL used when BuildConfig is not yet available (unit tests, etc.)
    private const val FALLBACK_BASE_URL = "http://10.0.2.2:8000/"

    private var currentBaseUrl: String = try {
        BuildConfig.API_BASE_URL
    } catch (_: Exception) {
        FALLBACK_BASE_URL
    }

    private var apiServiceInstance: LivestockApiService? = null

    fun setBaseUrl(url: String) {
        val sanitized = if (url.endsWith("/")) url else "$url/"
        if (sanitized != currentBaseUrl) {
            currentBaseUrl = sanitized
            apiServiceInstance = null // Invalidate cached service
        }
    }

    fun getService(): LivestockApiService {
        return apiServiceInstance ?: synchronized(this) {
            apiServiceInstance ?: buildRetrofit().create(LivestockApiService::class.java).also {
                apiServiceInstance = it
            }
        }
    }

    private fun buildRetrofit(): Retrofit {
        val clientBuilder = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)

        // Body-level logging is restricted to debug builds only.
        // In release builds this interceptor is NOT added, preventing
        // sensitive payload data from appearing in logcat or crash reporters.
        if (BuildConfig.DEBUG) {
            val loggingInterceptor = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }
            clientBuilder.addInterceptor(loggingInterceptor)
        }

        return Retrofit.Builder()
            .baseUrl(currentBaseUrl)
            .client(clientBuilder.build())
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }
}
