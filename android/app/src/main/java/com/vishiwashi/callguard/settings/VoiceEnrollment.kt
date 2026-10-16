package com.vishiwashi.callguard.settings

import android.Manifest
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import androidx.annotation.RequiresPermission
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Records a short sample of the owner's voice and turns it into a voiceprint on the edge
 * server (`POST /api/voiceprint`). Only the resulting numbers are kept, on this phone, in
 * [ProtectionSettings.voiceprint]; neither the recording nor the voiceprint is stored by
 * the server.
 */
class VoiceEnrollment(private val httpClient: OkHttpClient) {

    interface Listener {
        /** 0..1 while recording, then 1 while the server computes the voiceprint. */
        fun onProgress(fraction: Float, uploading: Boolean)
        fun onSaved(speechSeconds: Double)
        fun onFailed(message: String)
    }

    private val main = Handler(Looper.getMainLooper())
    private val cancelled = AtomicBoolean(false)
    private var worker: Thread? = null

    val isRunning: Boolean get() = worker?.isAlive == true

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start(endpoint: EdgeEndpoint, settings: ProtectionSettings, listener: Listener) {
        if (isRunning) return
        cancelled.set(false)
        worker = Thread({ run(endpoint, settings, listener) }, "CallGuard-VoiceEnrollment").apply { start() }
    }

    fun cancel() {
        cancelled.set(true)
    }

    @SuppressLint("MissingPermission") // Checked by the caller (start is annotated).
    private fun run(endpoint: EdgeEndpoint, settings: ProtectionSettings, listener: Listener) {
        val samples = try {
            record(listener) ?: return
        } catch (e: Exception) {
            post { listener.onFailed("Couldn't use the microphone (${e.message ?: e.javaClass.simpleName}).") }
            return
        }
        post { listener.onProgress(1f, uploading = true) }

        val bytes = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            .apply { asShortBuffer().put(samples) }.array()
        val request = Request.Builder()
            .url("${endpoint.httpBase}/api/voiceprint")
            .post(bytes.toRequestBody("application/octet-stream".toMediaType()))
            .build()
        try {
            httpClient.newCall(request).execute().use { response ->
                val json = JSONObject(response.body?.string().orEmpty().ifBlank { "{}" })
                if (!response.isSuccessful || !json.optBoolean("ok")) {
                    val detail = json.optString("detail").ifBlank { "Server answered HTTP ${response.code}." }
                    post { listener.onFailed(detail) }
                    return
                }
                val array = json.getJSONArray("embedding")
                val embedding = FloatArray(array.length()) { array.getDouble(it).toFloat() }
                if (embedding.size != ProtectionSettings.VOICEPRINT_DIMS) {
                    post { listener.onFailed("Unexpected voiceprint from server.") }
                    return
                }
                settings.voiceprint = embedding
                val seconds = json.optDouble("speech_seconds", 0.0)
                post { listener.onSaved(seconds) }
            }
        } catch (e: IOException) {
            post { listener.onFailed("Can't reach the server: ${e.message ?: e.javaClass.simpleName}") }
        } catch (e: Exception) {
            post { listener.onFailed("Unexpected reply from the server.") }
        }
    }

    @SuppressLint("MissingPermission")
    private fun record(listener: Listener): ShortArray? {
        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuffer * 2, SAMPLE_RATE / 5 * 2),
        )
        try {
            check(record.state == AudioRecord.STATE_INITIALIZED) { "microphone unavailable" }
            val total = SAMPLE_RATE * RECORD_SECONDS
            val samples = ShortArray(total)
            record.startRecording()
            var offset = 0
            var lastReported = -1
            while (offset < total) {
                if (cancelled.get()) return null
                val read = record.read(samples, offset, minOf(SAMPLE_RATE / 10, total - offset))
                if (read < 0) error("AudioRecord error $read")
                offset += read
                val tenth = offset * 10 / total
                if (tenth != lastReported) {
                    lastReported = tenth
                    val fraction = offset.toFloat() / total
                    post { listener.onProgress(fraction, uploading = false) }
                }
            }
            return samples
        } finally {
            try {
                record.stop()
            } catch (_: IllegalStateException) {
            }
            record.release()
        }
    }

    private fun post(block: () -> Unit) {
        if (!cancelled.get()) main.post(block)
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val RECORD_SECONDS = 10

        /** Read aloud during enrolment: mixes Hindi and English, like most calls. */
        const val SCRIPT =
            "Namaste, main VishyWashy ke liye apni awaaz record kar raha hoon. " +
                "This is my normal speaking voice. Aaj mausam accha hai aur main ghar par hoon."
    }
}
