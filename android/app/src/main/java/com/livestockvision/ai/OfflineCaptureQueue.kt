package com.livestockvision.ai

import android.content.Context
import androidx.room.*

@Dao
interface OfflineCaptureDao {
    @Query("SELECT * FROM offline_captures WHERE isSynced = 0 ORDER BY timestamp ASC")
    suspend fun getPendingCaptures(): List<OfflineCaptureEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCapture(capture: OfflineCaptureEntity)

    @Update
    suspend fun updateCapture(capture: OfflineCaptureEntity)

    @Query("UPDATE offline_captures SET isSynced = 1 WHERE captureId = :captureId")
    suspend fun markSynced(captureId: String)

    @Query("DELETE FROM offline_captures WHERE isSynced = 1")
    suspend fun purgeSyncedCaptures()
}

@Database(entities = [OfflineCaptureEntity::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun captureDao(): OfflineCaptureDao

    companion object {
        @Volatile private var instance: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "livestock_offline.db"
                ).build().also { instance = it }
            }
    }
}
