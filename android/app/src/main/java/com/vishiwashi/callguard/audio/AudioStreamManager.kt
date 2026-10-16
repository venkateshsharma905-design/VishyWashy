package com.vishiwashi.callguard.audio

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioRecordingConfiguration
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AudioEffect
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONException
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.min
import kotlin.random.Random

/**
 * Captures 16 kHz / 16-bit / mono PCM and streams 2.5 s sliding windows to the edge
 * server over an OkHttp WebSocket.
 *
 * Wire protocol (matches `server.py` → `/ws/telemetry/{session_id}`):
 *  - client → server, binary: little-endian Int16 PCM @ 16 kHz mono
 *  - client → server, text:   `{"action":"ping"}` / `{"action":"terminate"}`
 *  - server → client, text:   telemetry JSON every ~500 ms, `{"event":"pong"}`
 *
 * Windowing: every [Config.hopMs] (0.5 s) the newest 2.5 s window is sent. During the first
 * 2.5 s (and with [TransmitMode.HOP_ONLY]) only the new hop is sent. The server's
 * `SlidingWindowBuffer` slides hops in and *replaces* its buffer with any frame ≥ 2.5 s, so
 * both modes produce the identical analysis window server-side. HOP_ONLY uses 5× less uplink.
 *
 * All [Listener] callbacks are delivered on the main thread.
 */
class AudioStreamManager(
    context: Context,
    endpointBaseUrl: String,
    private val listener: Listener,
    private val config: Config = Config(),
    private val httpClient: OkHttpClient = defaultHttpClient(),
) {

    enum class TransmitMode { SLIDING_WINDOW, HOP_ONLY }

    enum class ConnectionState { IDLE, CONNECTING, CONNECTED, RECONNECTING, CLOSED }

    data class Config(
        val sampleRateHz: Int = 16_000,
        val windowMs: Int = 2_500,
        val hopMs: Int = 500,
        val transmitMode: TransmitMode = TransmitMode.SLIDING_WINDOW,
        val enableEchoCanceler: Boolean = true,
        /**
         * Capture sources tried in order. Call sources ([MediaRecorder.AudioSource.VOICE_CALL],
         * VOICE_DOWNLINK, VOICE_UPLINK) are only usable on devices that expose them to
         * non-system apps; when one fails to open, or delivers nothing but digital silence for
         * [callSourceProbeMs], the next source is used.
         */
        val audioSources: List<Int> = listOf(MediaRecorder.AudioSource.VOICE_COMMUNICATION),
        /** When set, this session id is used on the wire (so the phone can fetch the report later). */
        val sessionId: String? = null,
        /** Text frames sent every time the socket opens (e.g. the owner's voiceprint). */
        val openingMessages: List<String> = emptyList(),
        val callSourceProbeMs: Int = 3_000,
        /** Frames are dropped (not queued) once this much data is waiting to be sent: stale audio has no value. */
        val maxQueuedBytes: Long = 256L * 1024,
        val rttProbeIntervalMs: Long = 2_000,
        val reconnectBaseDelayMs: Long = 500,
        val reconnectMaxDelayMs: Long = 15_000,
    ) {
        init {
            require(sampleRateHz > 0) { "sampleRateHz must be positive" }
            require(hopMs in 1..windowMs) { "hopMs must be in 1..windowMs" }
            require(windowMs % hopMs == 0) { "windowMs must be a multiple of hopMs" }
            require(audioSources.isNotEmpty()) { "audioSources must not be empty" }
        }

        val windowSamples: Int get() = sampleRateHz * windowMs / 1000
        val hopSamples: Int get() = sampleRateHz * hopMs / 1000
    }

    interface Listener {
        fun onTelemetry(telemetry: RiskTelemetry)
        fun onConnectionStateChanged(state: ConnectionState)
        fun onNetworkRtt(rttMs: Long) {}

        /**
         * Android 10+ silences ordinary apps' capture while another app (typically the
         * dialer during a cellular call) has audio-input priority. Reported so the UI never
         * shows a reassuring "safe" score computed from silence.
         */
        fun onCaptureSilencedChanged(silenced: Boolean) {}

        /**
         * The source audio is now coming from. For call sources this fires only once real
         * (non-silent) call audio has been observed, so it means "both parties are captured".
         */
        fun onAudioSourceChanged(source: Int) {}
        fun onFatalError(message: String, cause: Throwable?)
    }

    private val appContext = context.applicationContext
    private val baseUrl = endpointBaseUrl.trim().trimEnd('/')
    private val mainHandler = Handler(Looper.getMainLooper())
    private val running = AtomicBoolean(false)
    private val pingSentAtMs = AtomicLong(0L)

    /** Incremented per connection attempt and on stop(); callbacks from older sockets are ignored. */
    private val connectionGeneration = AtomicInteger(0)

    @Volatile private var webSocket: WebSocket? = null
    @Volatile private var socketOpen = false
    @Volatile private var reconnectAttempts = 0

    @Volatile private var audioRecord: AudioRecord? = null
    @Volatile private var activeSourceIndex = 0
    private var echoCanceler: AcousticEchoCanceler? = null
    private var recordingCallback: AudioManager.AudioRecordingCallback? = null
    private var captureThread: Thread? = null
    private var scope: CoroutineScope? = null
    private var rttJob: Job? = null

    /** Server-side session id; a new one is generated on every [start]. */
    @Volatile var sessionId: String? = null
        private set

    @Volatile var framesSent: Long = 0L
        private set

    @Volatile var framesDropped: Long = 0L
        private set

    init {
        require(baseUrl.startsWith("ws://") || baseUrl.startsWith("wss://")) {
            "Endpoint must be a ws:// or wss:// URL, got: $endpointBaseUrl"
        }
    }

    val isRunning: Boolean get() = running.get()

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start() {
        if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            dispatchFatal("Microphone permission has not been granted.", null)
            return
        }
        if (!running.compareAndSet(false, true)) return

        val record = openFirstWorkingSource(fromIndex = 0)
        if (record == null) {
            running.set(false)
            dispatchFatal("The microphone is unavailable (it may be in use by another app).", null)
            return
        }

        sessionId = config.sessionId ?: UUID.randomUUID().toString()
        framesSent = 0
        framesDropped = 0
        reconnectAttempts = 0
        scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        connect()

        captureThread = Thread({ captureLoop(record) }, "CallGuard-AudioCapture").apply { start() }
    }

    /** Idempotent. Safe to call from the main thread; blocks for at most ~1 s while the capture thread exits. */
    fun stop() {
        if (!running.compareAndSet(true, false)) return

        rttJob?.cancel()
        rttJob = null

        connectionGeneration.incrementAndGet() // invalidates callbacks from the current socket
        val ws = webSocket
        webSocket = null
        if (ws != null) {
            if (socketOpen) ws.send(TERMINATE_MESSAGE)
            ws.close(NORMAL_CLOSURE, "client stop")
        }
        socketOpen = false

        // AudioRecord.stop() unblocks a pending read() so the capture thread can exit.
        try {
            audioRecord?.stop()
        } catch (e: IllegalStateException) {
            Log.w(TAG, "AudioRecord.stop failed", e)
        }
        captureThread?.let { thread ->
            try {
                thread.join(1_000)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }
        captureThread = null
        releaseAudio()

        scope?.cancel()
        scope = null
        pingSentAtMs.set(0L)
        dispatchState(ConnectionState.CLOSED)
    }

    // ------------------------------------------------------------------------------------------
    // Audio capture
    // ------------------------------------------------------------------------------------------

    /**
     * Opens and starts the first source from [fromIndex] that initialises and records. Sets
     * [audioRecord] / [activeSourceIndex]. Returns null when no source works.
     */
    private fun openFirstWorkingSource(fromIndex: Int): AudioRecord? {
        for (index in fromIndex until config.audioSources.size) {
            if (!running.get()) return null
            val source = config.audioSources[index]
            val record = try {
                createAudioRecord(source)
            } catch (t: Throwable) {
                Log.i(TAG, "Audio source ${sourceName(source)} unavailable: ${t.message}")
                continue
            }
            audioRecord = record
            activeSourceIndex = index
            attachEchoCanceler(record, source)
            registerSilenceMonitor(record)
            try {
                record.startRecording()
            } catch (e: IllegalStateException) {
                Log.w(TAG, "startRecording failed for ${sourceName(source)}", e)
            }
            if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                Log.i(TAG, "Capturing from ${sourceName(source)}")
                if (source !in CALL_SOURCES) mainHandler.post { if (running.get()) listener.onAudioSourceChanged(source) }
                return record
            }
            releaseAudio()
        }
        return null
    }

    @SuppressLint("MissingPermission") // Checked in start().
    private fun createAudioRecord(source: Int): AudioRecord {
        val channel = AudioFormat.CHANNEL_IN_MONO
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val minBuffer = AudioRecord.getMinBufferSize(config.sampleRateHz, channel, encoding)
        check(minBuffer > 0) { "16 kHz mono PCM16 capture is not supported on this device ($minBuffer)" }

        // Room for at least two hops so a scheduling hiccup never overruns the HAL buffer.
        val bufferBytes = maxOf(minBuffer * 4, config.hopSamples * BYTES_PER_SAMPLE * 2)

        val record = AudioRecord.Builder()
            .setAudioSource(source)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(config.sampleRateHz)
                    .setChannelMask(channel)
                    .setEncoding(encoding)
                    .build(),
            )
            .setBufferSizeInBytes(bufferBytes)
            .build()

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            error("AudioRecord failed to initialise")
        }
        return record
    }

    private fun attachEchoCanceler(record: AudioRecord, source: Int) {
        // Echo cancellation would remove the other party's voice from a speakerphone call, so it
        // is only used with VOICE_COMMUNICATION outside calls.
        if (!config.enableEchoCanceler || source != MediaRecorder.AudioSource.VOICE_COMMUNICATION) return
        if (!AcousticEchoCanceler.isAvailable()) {
            Log.i(TAG, "AcousticEchoCanceler not available; relying on VOICE_COMMUNICATION processing")
            return
        }
        echoCanceler = try {
            AcousticEchoCanceler.create(record.audioSessionId)?.also { aec ->
                val status = aec.setEnabled(true)
                if (status != AudioEffect.SUCCESS) Log.w(TAG, "AEC enable returned $status")
            }
        } catch (e: RuntimeException) {
            Log.w(TAG, "Unable to attach AcousticEchoCanceler", e)
            null
        }
    }

    private fun registerSilenceMonitor(record: AudioRecord) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        val sessionIdForRecord = record.audioSessionId
        val callback = object : AudioManager.AudioRecordingCallback() {
            private var lastSilenced: Boolean? = null

            override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>) {
                val mine = configs.firstOrNull { it.clientAudioSessionId == sessionIdForRecord } ?: return
                val silenced = mine.isClientSilenced
                if (silenced != lastSilenced) {
                    lastSilenced = silenced
                    if (silenced) Log.w(TAG, "Capture silenced by the system (concurrent capture policy)")
                    if (running.get()) listener.onCaptureSilencedChanged(silenced)
                }
            }
        }
        record.registerAudioRecordingCallback(appContext.mainExecutor, callback)
        recordingCallback = callback
    }

    private fun captureLoop(initial: AudioRecord) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)

        val hopSamples = config.hopSamples
        val windowSamples = config.windowSamples
        val hop = ShortArray(hopSamples)
        val window = ShortArray(windowSamples)
        var filled = 0

        var record = initial
        var probing = config.audioSources[activeSourceIndex] in CALL_SOURCES
        var probedMs = 0

        while (running.get()) {
            var offset = 0
            while (offset < hopSamples && running.get()) {
                val read = record.read(hop, offset, hopSamples - offset)
                if (read < 0) {
                    if (running.get()) dispatchFatal("Audio capture failed (AudioRecord error $read).", null)
                    return
                }
                offset += read
            }
            if (offset < hopSamples) return // stopped mid-hop

            if (probing) {
                // Many devices open call sources for ordinary apps but feed them pure zeros.
                if (hop.any { it.toInt() != 0 }) {
                    probing = false
                    val source = config.audioSources[activeSourceIndex]
                    Log.i(TAG, "Call audio confirmed on ${sourceName(source)}")
                    mainHandler.post { if (running.get()) listener.onAudioSourceChanged(source) }
                } else {
                    probedMs += config.hopMs
                    if (probedMs >= config.callSourceProbeMs) {
                        Log.i(TAG, "${sourceName(config.audioSources[activeSourceIndex])} is silent; falling back")
                        try {
                            record.stop()
                        } catch (e: IllegalStateException) {
                            Log.w(TAG, "stop during fallback failed", e)
                        }
                        releaseAudio()
                        if (!running.get()) return
                        record = openFirstWorkingSource(activeSourceIndex + 1) ?: run {
                            if (running.get()) dispatchFatal("No usable audio source during the call.", null)
                            return
                        }
                        probing = config.audioSources[activeSourceIndex] in CALL_SOURCES
                        probedMs = 0
                        continue
                    }
                    continue // don't stream silence while probing
                }
            }

            // Slide the 2.5 s window left by one hop and append the newest 0.5 s.
            System.arraycopy(window, hopSamples, window, 0, windowSamples - hopSamples)
            System.arraycopy(hop, 0, window, windowSamples - hopSamples, hopSamples)
            filled = min(windowSamples, filled + hopSamples)

            val payload = if (config.transmitMode == TransmitMode.SLIDING_WINDOW && filled == windowSamples) {
                window
            } else {
                hop
            }
            transmit(payload)
        }
    }

    private fun transmit(samples: ShortArray) {
        val ws = webSocket
        if (ws == null || !socketOpen || ws.queueSize() > config.maxQueuedBytes) {
            framesDropped++
            return
        }
        val bytes = ByteBuffer.allocate(samples.size * BYTES_PER_SAMPLE).order(ByteOrder.LITTLE_ENDIAN)
        bytes.asShortBuffer().put(samples)
        if (ws.send(bytes.array().toByteString())) framesSent++ else framesDropped++
    }

    private fun releaseAudio() {
        val record = audioRecord ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            recordingCallback?.let { record.unregisterAudioRecordingCallback(it) }
        }
        recordingCallback = null
        echoCanceler?.release()
        echoCanceler = null
        record.release()
        audioRecord = null
    }

    // ------------------------------------------------------------------------------------------
    // WebSocket
    // ------------------------------------------------------------------------------------------

    private fun connect() {
        if (!running.get()) return
        val id = sessionId ?: return
        dispatchState(if (reconnectAttempts == 0) ConnectionState.CONNECTING else ConnectionState.RECONNECTING)
        val generation = connectionGeneration.incrementAndGet()
        val request = Request.Builder().url("$baseUrl/$id").build()
        val socket = httpClient.newWebSocket(request, SocketListener(generation))
        if (generation == connectionGeneration.get()) webSocket = socket else socket.cancel()
    }

    private fun handleDisconnect(generation: Int, cause: Throwable?) {
        if (generation != connectionGeneration.get()) return // stale socket or deliberate stop()
        socketOpen = false
        rttJob?.cancel()
        pingSentAtMs.set(0L)

        if (!running.get()) {
            dispatchState(ConnectionState.CLOSED)
            return
        }
        if (cause != null) Log.w(TAG, "WebSocket failure: ${cause.message}")

        reconnectAttempts++
        val exponential = config.reconnectBaseDelayMs shl min(reconnectAttempts - 1, 6)
        val delayMs = min(config.reconnectMaxDelayMs, exponential) + Random.nextLong(0, 250)
        dispatchState(ConnectionState.RECONNECTING)
        scope?.launch {
            delay(delayMs)
            connect()
        }
    }

    private fun startRttProbe(ws: WebSocket, generation: Int) {
        rttJob?.cancel()
        rttJob = scope?.launch {
            while (isActive && socketOpen && generation == connectionGeneration.get()) {
                val now = SystemClock.elapsedRealtime()
                val outstanding = pingSentAtMs.get()
                // One probe in flight at a time; abandon a probe that never came back.
                if (outstanding == 0L || now - outstanding > PING_TIMEOUT_MS) {
                    pingSentAtMs.set(now)
                    ws.send(PING_MESSAGE)
                }
                delay(config.rttProbeIntervalMs)
            }
        }
    }

    private inner class SocketListener(private val generation: Int) : WebSocketListener() {
        private fun isCurrent() = running.get() && generation == connectionGeneration.get()

        override fun onOpen(webSocket: WebSocket, response: Response) {
            if (!isCurrent()) {
                webSocket.cancel()
                return
            }
            // onOpen can race the assignment in connect(); make sure the field points at this socket.
            this@AudioStreamManager.webSocket = webSocket
            socketOpen = true
            reconnectAttempts = 0
            config.openingMessages.forEach { webSocket.send(it) }
            dispatchState(ConnectionState.CONNECTED)
            startRttProbe(webSocket, generation)
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            if (!isCurrent()) return
            val json = try {
                JSONObject(text)
            } catch (e: JSONException) {
                Log.w(TAG, "Ignoring non-JSON message")
                return
            }
            when (json.optString("event")) {
                "pong" -> {
                    val sentAt = pingSentAtMs.getAndSet(0L)
                    if (sentAt > 0L) {
                        val rtt = SystemClock.elapsedRealtime() - sentAt
                        mainHandler.post { if (running.get()) listener.onNetworkRtt(rtt) }
                    }
                }
                "session_terminated" -> Unit
                else -> RiskTelemetry.fromJson(json)?.let { telemetry ->
                    mainHandler.post { if (running.get()) listener.onTelemetry(telemetry) }
                }
            }
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) = Unit

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(NORMAL_CLOSURE, null)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            handleDisconnect(generation, null)
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            handleDisconnect(generation, t)
        }
    }

    // ------------------------------------------------------------------------------------------

    private fun dispatchState(state: ConnectionState) {
        mainHandler.post { listener.onConnectionStateChanged(state) }
    }

    private fun dispatchFatal(message: String, cause: Throwable?) {
        Log.e(TAG, message, cause)
        mainHandler.post { listener.onFatalError(message, cause) }
    }

    companion object {
        private const val TAG = "AudioStreamManager"
        private const val BYTES_PER_SAMPLE = 2
        private const val NORMAL_CLOSURE = 1000
        private const val PING_TIMEOUT_MS = 10_000L
        private const val PING_MESSAGE = """{"action":"ping"}"""
        private const val TERMINATE_MESSAGE = """{"action":"terminate"}"""

        /** Sources that carry the call itself (both parties) rather than the microphone. */
        val CALL_SOURCES: Set<Int> = setOf(
            MediaRecorder.AudioSource.VOICE_CALL,
            MediaRecorder.AudioSource.VOICE_DOWNLINK,
            MediaRecorder.AudioSource.VOICE_UPLINK,
        )

        fun sourceName(source: Int): String = when (source) {
            MediaRecorder.AudioSource.MIC -> "MIC"
            MediaRecorder.AudioSource.VOICE_CALL -> "VOICE_CALL"
            MediaRecorder.AudioSource.VOICE_DOWNLINK -> "VOICE_DOWNLINK"
            MediaRecorder.AudioSource.VOICE_UPLINK -> "VOICE_UPLINK"
            MediaRecorder.AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
            MediaRecorder.AudioSource.VOICE_COMMUNICATION -> "VOICE_COMMUNICATION"
            MediaRecorder.AudioSource.UNPROCESSED -> "UNPROCESSED"
            else -> "source $source"
        }

        fun defaultHttpClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS) // long-lived socket
            .pingInterval(15, TimeUnit.SECONDS)    // protocol-level keep-alive / dead-peer detection
            .build()
    }
}
