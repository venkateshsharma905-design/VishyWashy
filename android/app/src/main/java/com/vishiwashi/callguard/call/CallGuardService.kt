package com.vishiwashi.callguard.call

import android.Manifest
import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Path
import android.graphics.Rect
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.telecom.TelecomManager
import android.telephony.PhoneStateListener
import android.telephony.TelephonyManager
import android.util.Log
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.vishiwashi.callguard.audio.AudioStreamManager
import com.vishiwashi.callguard.audio.AudioStreamManager.ConnectionState
import com.vishiwashi.callguard.audio.RiskTelemetry
import com.vishiwashi.callguard.consent.DpdpConsentManager
import com.vishiwashi.callguard.hud.CallHud
import com.vishiwashi.callguard.hud.HudState
import com.vishiwashi.callguard.hud.RiskLevel
import com.vishiwashi.callguard.hud.RiskLevelTracker
import com.vishiwashi.callguard.settings.CallHistory
import com.vishiwashi.callguard.settings.EdgeEndpoint
import com.vishiwashi.callguard.settings.ProtectionSettings
import com.vishiwashi.callguard.settings.Whitelist
import java.util.UUID
import kotlin.math.max
import kotlin.math.roundToInt
import org.json.JSONArray
import org.json.JSONObject

/**
 * Call-driven protection, hosted in an accessibility service.
 *
 * Why an accessibility service: during a voice call Android silences every ordinary app's audio
 * capture — only the call itself, accessibility services and privileged system apps receive
 * audio (developer.android.com/media/platform/sharing-audio-input). An accessibility service can
 * also draw `TYPE_ACCESSIBILITY_OVERLAY` windows above the dialer without the overlay permission,
 * and it is bound by the system, so nothing has to run in the foreground between calls.
 *
 * Lifecycle: idle while no call is active. When a call connects (answered, dialled, or a VoIP
 * call — detected from the audio mode), and the user has armed protection with valid consent,
 * the HUD appears and audio streams to the edge server. Both stop when the call ends.
 *
 * Capturing both parties: the call-audio source (VOICE_CALL) is tried first; most phones only
 * expose it to system apps, so the microphone is used instead and — if enabled — the call is
 * switched to speakerphone so the microphone hears the caller as well as the user.
 */
class CallGuardService : AccessibilityService(), AudioStreamManager.Listener, CallHud.Actions {

    private val main = Handler(Looper.getMainLooper())
    private lateinit var audioManager: AudioManager
    private lateinit var settings: ProtectionSettings
    private lateinit var consentPrefs: SharedPreferences

    private var modeListener: AudioManager.OnModeChangedListener? = null
    private var phoneStateListener: PhoneStateListener? = null
    private var callActive = false

    // ---- Per-call session state --------------------------------------------------------------
    private var hud: CallHud? = null
    private var stream: AudioStreamManager? = null
    private var riskTracker = RiskLevelTracker()
    private var connectionState = ConnectionState.IDLE
    private var lastTelemetry: RiskTelemetry? = null
    private var networkRttMs: Long? = null
    private var captureSilenced = false
    private var callAudioCaptured = false
    private var speakerRequestedByUs = false
    private var seniorMode = false
    private var seniorAutoHangupDone = false
    private var fraudHoldDone = false
    private var holdAttempt = 0
    private var moreMenuTried = false
    private var micMutedByUs = false
    /** Set while the call is paused for a fraud alert, waiting for the user's decision. */
    private var pause: HudState.OnHold.Pause? = null

    // Per-call reporting state
    private var sessionId: String? = null
    private var reportToken: String? = null
    private var peerLabel: String = ""
    private var callStartedAtMs: Long = 0L
    private var peakRisk: Double = 0.0

    private val endCallDebounce = Runnable { onCallStateChanged(false, debounced = true) }
    private val prefListener = SharedPreferences.OnSharedPreferenceChangeListener { _, _ -> reconcile() }

    // ------------------------------------------------------------------------------------------
    // AccessibilityService lifecycle
    // ------------------------------------------------------------------------------------------

    override fun onServiceConnected() {
        super.onServiceConnected()
        audioManager = getSystemService(AudioManager::class.java)
        settings = ProtectionSettings(this)
        consentPrefs = getSharedPreferences(DpdpConsentManager.PREFS_NAME, Context.MODE_PRIVATE)
        settings.prefs.registerOnSharedPreferenceChangeListener(prefListener)
        consentPrefs.registerOnSharedPreferenceChangeListener(prefListener)
        registerCallDetection()
        connected = true
        Log.i(TAG, "VishyWashy accessibility service connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    override fun onUnbind(intent: Intent?): Boolean {
        shutdown()
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        shutdown()
        super.onDestroy()
    }

    private fun shutdown() {
        if (!connected) return
        connected = false
        main.removeCallbacksAndMessages(null)
        endSession()
        unregisterCallDetection()
        settings.prefs.unregisterOnSharedPreferenceChangeListener(prefListener)
        consentPrefs.unregisterOnSharedPreferenceChangeListener(prefListener)
    }

    // ------------------------------------------------------------------------------------------
    // Call detection
    // ------------------------------------------------------------------------------------------

    private fun registerCallDetection() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Audio mode needs no permission and covers cellular (IN_CALL) and VoIP (IN_COMMUNICATION).
            val listener = AudioManager.OnModeChangedListener { mode -> onCallStateChanged(isCallMode(mode)) }
            audioManager.addOnModeChangedListener(mainExecutor, listener)
            modeListener = listener
        } else {
            val telephony = getSystemService(TelephonyManager::class.java)
            @Suppress("DEPRECATION")
            val listener = object : PhoneStateListener() {
                @Deprecated("Deprecated in Java")
                override fun onCallStateChanged(state: Int, phoneNumber: String?) {
                    onCallStateChanged(state == TelephonyManager.CALL_STATE_OFFHOOK || isCallMode(audioManager.mode))
                }
            }
            @Suppress("DEPRECATION")
            telephony?.listen(listener, PhoneStateListener.LISTEN_CALL_STATE)
            phoneStateListener = listener
        }
        onCallStateChanged(isCallMode(audioManager.mode))
    }

    private fun unregisterCallDetection() {
        modeListener?.let { if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) audioManager.removeOnModeChangedListener(it) }
        modeListener = null
        phoneStateListener?.let {
            @Suppress("DEPRECATION")
            getSystemService(TelephonyManager::class.java)?.listen(it, PhoneStateListener.LISTEN_NONE)
        }
        phoneStateListener = null
    }

    private fun isCallMode(mode: Int) = mode == AudioManager.MODE_IN_CALL || mode == AudioManager.MODE_IN_COMMUNICATION

    private fun onCallStateChanged(active: Boolean, debounced: Boolean = false) {
        if (active) {
            main.removeCallbacks(endCallDebounce)
            if (!callActive) {
                callActive = true
                Log.i(TAG, "Call connected")
                reconcile()
            }
        } else if (callActive) {
            // The mode can flicker during call setup or when audio routes change.
            if (!debounced) {
                main.removeCallbacks(endCallDebounce)
                main.postDelayed(endCallDebounce, CALL_END_DEBOUNCE_MS)
                return
            }
            callActive = false
            Log.i(TAG, "Call ended")
            reconcile()
        }
    }

    /** Starts or stops the per-call session to match call state, settings and consent. */
    private fun reconcile() {
        if (!connected) return
        val wanted = callActive && settings.armed && prerequisitesMet(this)
        when {
            wanted && hud == null -> startSession()
            !wanted && hud != null -> endSession()
        }
    }

    // ------------------------------------------------------------------------------------------
    // Session
    // ------------------------------------------------------------------------------------------

    /** Reads the number and visible text of the in-call screen (for whitelist + the peer label). */
    private fun describeCall(): Pair<String?, String> {
        val texts = ArrayList<String>(32)
        // Only the dialer / in-call application window — never the status bar or system UI, whose
        // text ("Wi-Fi signal full", "battery 77%") would otherwise be mistaken for the caller.
        val appWindows = windows?.filter {
            it.type == AccessibilityWindowInfo.TYPE_APPLICATION &&
                it.root?.packageName?.toString().let { pkg -> pkg != packageName && pkg !in SYSTEM_PACKAGES }
        }.orEmpty()
        for (window in appWindows) {
            window.root?.let { collectText(it, texts, 0) }
        }
        val number = texts.asSequence().mapNotNull { NUMBER_RE.find(it)?.value?.trim() }
            .firstOrNull { it.filter(Char::isDigit).length >= 7 }
        val name = texts.firstOrNull {
            it.length in 2..40 && it.any(Char::isLetter) && it.none(Char::isDigit) &&
                NON_PEER_WORDS.none { w -> it.contains(w, ignoreCase = true) }
        }
        return number to (number ?: name ?: "Unknown caller")
    }

    private fun collectText(node: AccessibilityNodeInfo, out: MutableList<String>, depth: Int) {
        if (depth > 6 || out.size > 60) return
        (node.text ?: node.contentDescription)?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let(out::add)
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let { collectText(it, out, depth + 1) }
        }
    }

    @SuppressLint("MissingPermission") // RECORD_AUDIO verified by prerequisitesMet().
    private fun startSession() {
        val endpoint = EdgeEndpoint.load(this) ?: return

        val (number, label) = describeCall()
        if (Whitelist(this).matches(number, number ?: label)) {
            Log.i(TAG, "Whitelisted caller — not monitoring this call")
            callActive = false   // ignore until the next (non-whitelisted) call
            return
        }
        peerLabel = label
        sessionId = UUID.randomUUID().toString()
        reportToken = UUID.randomUUID().toString().replace("-", "")
        callStartedAtMs = System.currentTimeMillis()
        peakRisk = 0.0
        seniorMode = settings.seniorMode
        seniorAutoHangupDone = false
        fraudHoldDone = false
        pause = null

        val controller = CallHud(
            host = this,
            windowManager = getSystemService(WindowManager::class.java),
            windowType = WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            seniorMode = settings.seniorMode,
            actions = this,
        )
        if (!controller.attach()) return
        hud = controller

        riskTracker = RiskLevelTracker()
        connectionState = ConnectionState.IDLE
        lastTelemetry = null
        networkRttMs = null
        captureSilenced = false
        callAudioCaptured = false
        render()

        val manager = try {
            AudioStreamManager(
                this,
                endpoint.webSocketBase,
                this,
                AudioStreamManager.Config(
                    enableEchoCanceler = false,
                    audioSources = CALL_AUDIO_SOURCES,
                    sessionId = sessionId,
                    openingMessages = listOfNotNull(
                        settings.voiceprint?.let(::voiceprintMessage),
                        callMetaMessage(),
                    ),
                ),
            )
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "Invalid endpoint", e)
            controller.render(HudState.Unavailable("Server address invalid"))
            return
        }
        stream = manager
        manager.start()
    }

    private fun endSession() {
        resetSpeaker()
        restoreCallVolume()
        lastSpeakerOn = null
        main.removeCallbacks(pressHold)
        main.removeCallbacks(pressHoldInMenu)
        restoreMicrophone()
        pause = null
        val sid = sessionId
        val startedAt = callStartedAtMs
        stream?.stop()
        stream = null
        hud?.detach()
        hud = null
        if (speakerRequestedByUs && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            audioManager.clearCommunicationDevice()
        }
        speakerRequestedByUs = false
        if (sid != null && startedAt > 0) finishCall(sid, startedAt)
    }

    private fun finishCall(sid: String, startedAt: Long) {
        val durationS = ((System.currentTimeMillis() - startedAt) / 1000L).toInt()
        val flagged = peakRisk >= CallHud.CAUTION_THRESHOLD
        CallHistory(this).put(
            CallHistory.Record(
                id = sid, startedAtMs = startedAt, durationS = durationS, peer = peerLabel,
                peakRisk = peakRisk,
                outcome = if (flagged) CallHistory.Outcome.FLAGGED else CallHistory.Outcome.MONITORED,
            ),
        )
        val token = reportToken
        if (flagged && token != null) {
            // Offer to report, for the two-minute window the recording survives server-side.
            lastReport = ReportHandle(sid, token, peerLabel, EdgeEndpoint.load(this)?.display.orEmpty(),
                System.currentTimeMillis() + REPORT_WINDOW_MS)
            ReportNotifier.show(this, sid, token, peerLabel)
        }
        sessionId = null
        reportToken = null
    }

    // ------------------------------------------------------------------------------------------
    // AudioStreamManager.Listener (main thread)
    // ------------------------------------------------------------------------------------------

    override fun onTelemetry(telemetry: RiskTelemetry) {
        if (hud == null) return
        lastTelemetry = telemetry
        peakRisk = max(peakRisk, telemetry.riskScore)
        riskTracker.update(telemetry.riskScore)
        render()
        maybeSeniorAutoHangup(telemetry.riskScore)
        maybeHoldOnFraud()
    }

    /**
     * "Hold call on fraud": the first time a call reaches HIGH risk, pause it — the dialer's Hold
     * button, or its Mute button where there is none (WhatsApp) — and ask the user to continue or
     * hang up. Fires once per call; senior mode hangs up instead.
     */
    private fun maybeHoldOnFraud() {
        if (fraudHoldDone || seniorMode || !settings.holdOnFraud) return
        val score = lastTelemetry?.riskScore ?: return
        if (score < RiskLevel.THRESHOLD_ALERT && riskTracker.level != RiskLevel.ALERT) return
        fraudHoldDone = true
        holdAttempt = 0
        moreMenuTried = false
        Log.i(TAG, "Fraud risk ${(score * 100).toInt()}%: putting the call on hold")
        showInCallScreen()
        main.postDelayed(pressHold, HOLD_PRESS_DELAY_MS)
    }

    /**
     * Presses the dialer's Hold button. The call screen may take a moment to come up (or the
     * user may be in another app), so this retries for ~[HOLD_MAX_ATTEMPTS] × [HOLD_RETRY_MS],
     * bringing the call screen forward each time, before falling back to muting.
     */
    private val pressHold: Runnable = object : Runnable {
        override fun run() {
            if (hud == null || pause != null) return
            holdAttempt++
            Log.i(TAG, "Hold attempt $holdAttempt, windows: " +
                (windows?.mapNotNull { it.root?.packageName } ?: emptyList()).joinToString())
            when {
                isOnHold() -> paused(HudState.OnHold.Pause.HELD)
                clickNodeWithLabel(HOLD_LABELS, skipIfChecked = true) -> paused(HudState.OnHold.Pause.HELD)
                // Some dialers (Samsung, several Xiaomi/Oppo builds) keep Hold in the ⋮ / More menu.
                !moreMenuTried && clickNodeWithLabel(MORE_LABELS, exact = true) -> {
                    moreMenuTried = true
                    main.postDelayed(pressHoldInMenu, MENU_OPEN_DELAY_MS)
                }
                holdAttempt < HOLD_MAX_ATTEMPTS -> {
                    showInCallScreen()
                    main.postDelayed(this, HOLD_RETRY_MS)
                }
                else -> pauseByMuting()
            }
        }
    }

    private val pressHoldInMenu = Runnable {
        if (hud == null || pause != null) return@Runnable
        if (clickNodeWithLabel(HOLD_LABELS, skipIfChecked = true)) {
            paused(HudState.OnHold.Pause.HELD)
        } else {
            performGlobalAction(GLOBAL_ACTION_BACK)     // close the menu we opened
            main.postDelayed(pressHold, MENU_OPEN_DELAY_MS)
        }
    }

    /** No reachable Hold button: mute with the dialer's button, else mute the microphone directly. */
    private fun pauseByMuting() {
        if (clickNodeWithLabel(MUTE_LABELS, skipIfChecked = true)) {
            paused(HudState.OnHold.Pause.MUTED)
            return
        }
        val muted = try {
            audioManager.isMicrophoneMute = true
            audioManager.isMicrophoneMute
        } catch (e: RuntimeException) {
            Log.w(TAG, "Microphone mute rejected", e)
            false
        }
        micMutedByUs = muted
        paused(if (muted) HudState.OnHold.Pause.MUTED else HudState.OnHold.Pause.NOT_PAUSED)
    }

    /** The dialer shows the call as held: a Resume / Unhold button, a checked Hold toggle, or "On hold". */
    private fun isOnHold(): Boolean =
        findDialerNode(RESUME_LABELS, exact = true) != null ||
            findDialerNode(ON_HOLD_LABELS, exact = false) != null ||
            findDialerNode(HOLD_LABELS, exact = false)?.let { it.isChecked || it.isSelected } == true

    private fun restoreMicrophone() {
        if (!micMutedByUs) return
        micMutedByUs = false
        try {
            audioManager.isMicrophoneMute = false
        } catch (e: RuntimeException) {
            Log.w(TAG, "Microphone unmute rejected", e)
        }
    }

    private fun paused(how: HudState.OnHold.Pause) {
        pause = how
        Log.i(TAG, "Fraud alert: call paused ($how)")
        render()
    }

    override fun onContinueCall() {
        when (pause) {
            HudState.OnHold.Pause.HELD -> if (!clickNodeWithLabel(RESUME_LABELS, exact = true) &&
                !clickNodeWithLabel(HOLD_LABELS, exact = true, onlyIfChecked = true)
            ) {
                showInCallScreen()
                toast("Tap Resume / Unhold on the call screen to continue.")
            }
            HudState.OnHold.Pause.MUTED -> if (micMutedByUs) {
                restoreMicrophone()
            } else if (!clickNodeWithLabel(UNMUTE_LABELS, exact = true) &&
                !clickNodeWithLabel(MUTE_LABELS, exact = true, onlyIfChecked = true)
            ) {
                showInCallScreen()
                toast("Tap Unmute on the call screen to continue.")
            }
            else -> Unit
        }
        pause = null
        render()
    }

    /**
     * Senior mode: once the risk crosses the HIGH threshold, show the alert for a moment, then end
     * the call automatically and start the report. Fires once per call.
     */
    private fun maybeSeniorAutoHangup(riskScore: Double) {
        if (!seniorMode || seniorAutoHangupDone || riskScore < CallHud.ALERT_THRESHOLD) return
        seniorAutoHangupDone = true
        toast("High-risk call — VishyWashy is ending it for you.")
        main.postDelayed({
            val sid = sessionId
            val token = reportToken
            val peer = peerLabel
            onHangUp()
            if (sid != null && token != null) {
                main.postDelayed({ ReportActivity.launch(this, sid, token, peer) }, 1200)
            }
        }, SENIOR_HANGUP_DELAY_MS)
    }

    override fun onConnectionStateChanged(state: ConnectionState) {
        if (hud == null) return
        connectionState = state
        if (state != ConnectionState.CONNECTED) networkRttMs = null
        render()
    }

    override fun onNetworkRtt(rttMs: Long) {
        if (hud == null) return
        networkRttMs = rttMs
        render()
    }

    override fun onCaptureSilencedChanged(silenced: Boolean) {
        if (hud == null) return
        captureSilenced = silenced
        render()
    }

    override fun onAudioSourceChanged(source: Int) {
        if (hud == null) return
        callAudioCaptured = source in AudioStreamManager.CALL_SOURCES
        Log.i(TAG, "Audio source: ${AudioStreamManager.sourceName(source)}")
        if (!callAudioCaptured && settings.autoSpeaker) {
            resetSpeaker()
            main.post(speakerRetry)
        }
        render()
    }

    override fun onFatalError(message: String, cause: Throwable?) {
        Log.e(TAG, message, cause)
        hud?.render(HudState.Unavailable(message.take(40)))
        stream?.stop()
        stream = null
    }

    private fun render() {
        val controller = hud ?: return
        val telemetry = lastTelemetry
        val level = riskTracker.level
        val online = connectionState == ConnectionState.CONNECTED
        val callerAudible = callAudioCaptured || isSpeakerOn()

        val paused = pause
        val state: HudState = when {
            telemetry != null && paused != null -> HudState.OnHold(
                score = telemetry.riskScore,
                signals = telemetry.indicators,
                paused = paused,
            )
            telemetry != null && level == RiskLevel.ALERT -> HudState.Alert(
                score = telemetry.riskScore,
                signals = telemetry.indicators,
                guidance = telemetry.guidance,
            )
            captureSilenced -> HudState.Unavailable("Call audio blocked")
            !online -> HudState.Unavailable(
                if (connectionState == ConnectionState.RECONNECTING) "Reconnecting…" else "Connecting…",
            )
            telemetry == null -> HudState.Unavailable("Listening…")
            level == RiskLevel.CAUTION -> HudState.Caution(telemetry.riskScore, callerAudible)
            else -> HudState.Normal(
                latencyMs = (telemetry.inferenceLatencyMs + (networkRttMs ?: 0L)).roundToInt(),
                callerAudible = callerAudible,
            )
        }
        controller.render(state)
    }

    // ------------------------------------------------------------------------------------------
    // CallHud.Actions
    // ------------------------------------------------------------------------------------------

    override fun onHangUp() {
        // endCall() only ends the foreground call, so a call we put on hold is ended from its screen.
        val ended = if (pause == HudState.OnHold.Pause.HELD) {
            clickNodeWithLabel(END_CALL_LABELS) || endCallViaTelecom()
        } else {
            endCallViaTelecom() || clickNodeWithLabel(END_CALL_LABELS)
        }
        if (ended) return
        toast("Couldn't end the call automatically. Please hang up now.")
    }

    override fun onReport() {
        val dial = Intent(Intent.ACTION_DIAL, Uri.parse("tel:${CallHud.CYBER_HELPLINE}"))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        try {
            startActivity(dial)
        } catch (e: ActivityNotFoundException) {
            toast("No dialer found. Call ${CallHud.CYBER_HELPLINE} to report cyber fraud.")
        }
    }

    override fun onEnableSpeaker() {
        if (speakerGaveUp) {
            // Automatic switching doesn't work on this phone: show the call screen and say what to do.
            showInCallScreen()
            toast("Tap Speaker on the call screen so VishyWashy can hear the caller.")
        }
        resetSpeaker()
        main.postDelayed(speakerRetry, 600)
    }

    override fun onHangUpAndReport() {
        // Senior mode: hang up now, then open the report flow once the recording is ready.
        val sid = sessionId
        val token = reportToken
        val peer = peerLabel
        onHangUp()
        if (sid != null && token != null) {
            main.postDelayed({ ReportActivity.launch(this, sid, token, peer) }, 800)
        }
    }

    // ------------------------------------------------------------------------------------------
    // Speakerphone (so the microphone hears the caller)
    // ------------------------------------------------------------------------------------------

    private var speakerAttempt = 0
    private var speakerTarget = 0          // which "Speaker" element on the call screen we're trying
    private var speakerTapNext = false     // tried an accessibility click on it; a real tap is next
    private var speakerCallScreenShown = false
    private var speakerGaveUp = false      // automatic switching failed: the HUD asks the user
    private var lastSpeakerOn: Boolean? = null
    private var savedCallVolume: Int? = null

    private fun resetSpeaker() {
        main.removeCallbacks(speakerRetry)
        speakerAttempt = 0
        speakerTarget = 0
        speakerTapNext = false
        speakerCallScreenShown = false
        speakerGaveUp = false
    }

    /**
     * Turns the loudspeaker on, so the microphone hears the caller on a normal SIM call.
     *
     * The dialer's Speaker button is a toggle, so the state is checked before every press and the
     * attempts stop the moment the speaker is on — it is never pressed while already on. Dialers
     * differ: some ignore accessibility clicks (a real tap works), and some (OnePlus) show more
     * than one element called "Speaker", so each is tried in turn. If nothing works, the HUD asks
     * the user to tap Speaker themselves.
     */
    private val speakerRetry = object : Runnable {
        override fun run() {
            if (hud == null) return
            if (isSpeakerOn()) {
                speakerGaveUp = false
                render()
                return
            }
            if (!speakerCallScreenShown) {
                speakerCallScreenShown = true
                showInCallScreen()                         // the buttons must be on screen
                main.postDelayed(this, SPEAKER_SCREEN_MS)
                return
            }
            val targets = speakerTargets()
            if (targets.isEmpty()) {
                if (++speakerAttempt < SPEAKER_MAX_ATTEMPTS) {
                    main.postDelayed(this, SPEAKER_RETRY_MS)
                } else {
                    giveUpOnSpeaker()
                }
                return
            }
            if (speakerTarget >= targets.size) {
                giveUpOnSpeaker()
                return
            }
            val target = targets[speakerTarget]
            if (!speakerTapNext) {
                speakerTapNext = true
                val clicked = target.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                Log.i(TAG, "Speaker: click on element ${speakerTarget + 1}/${targets.size} (accepted=$clicked)")
            } else {
                speakerTapNext = false
                speakerTarget++
                val tapped = tapNode(target)
                Log.i(TAG, "Speaker: tap on element ${speakerTarget}/${targets.size} (dispatched=$tapped)")
            }
            main.postDelayed(this, SPEAKER_VERIFY_MS)
        }
    }

    private fun giveUpOnSpeaker() {
        Log.i(TAG, "Speaker: could not switch it on automatically; asking the user")
        requestSpeakerViaAudioManager()           // harmless where it doesn't apply
        speakerGaveUp = true
        render()
    }

    /** Clickable "Speaker" buttons on the call screen, top to bottom, without duplicates. */
    private fun speakerTargets(): List<AccessibilityNodeInfo> {
        val seen = mutableSetOf<String>()
        return dialerNodes(SPEAKER_LABELS, exact = true).mapNotNull { node ->
            var target: AccessibilityNodeInfo? = node
            var depth = 0
            while (target != null && !target.isClickable && depth < 4) {
                target = target.parent
                depth++
            }
            target?.takeIf { it.isClickable && it.isEnabled }
        }.filter { node ->
            val r = Rect().also(node::getBoundsInScreen)
            !r.isEmpty && seen.add(r.flattenToString())
        }
    }

    private fun isSpeakerOn(): Boolean {
        @Suppress("DEPRECATION")
        val on = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            audioManager.communicationDevice?.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER) ||
            audioManager.isSpeakerphoneOn || speakerButtonOn()
        if (on != lastSpeakerOn) {
            lastSpeakerOn = on
            Log.i(TAG, "Loudspeaker is ${if (on) "on" else "off"}")
            if (on) boostCallVolume()
        }
        return on
    }

    /** Some "Speaker" element on the call screen shows as on (checked / selected / "On"). */
    private fun speakerButtonOn(): Boolean =
        dialerNodes(SPEAKER_LABELS, exact = true).any { node ->
            var n: AccessibilityNodeInfo? = node
            var depth = 0
            var on = false
            while (n != null && depth <= 4 && !on) {
                val state = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    n.stateDescription?.toString()?.lowercase().orEmpty()
                } else ""
                // OnePlus says "Selected" / "Not selected", others "On" / "Off".
                on = (n.isCheckable && n.isChecked) || state == "on" || state == "selected" ||
                    (n.isClickable && n.isSelected)
                if (n.isClickable) break                // the button itself; don't look past it
                n = n.parent
                depth++
            }
            on
        }

    /** A louder loudspeaker is heard better by the microphone: max call volume while protected. */
    private fun boostCallVolume() {
        if (savedCallVolume != null) return
        try {
            val stream = AudioManager.STREAM_VOICE_CALL
            savedCallVolume = audioManager.getStreamVolume(stream)
            audioManager.setStreamVolume(stream, audioManager.getStreamMaxVolume(stream), 0)
        } catch (e: RuntimeException) {
            Log.w(TAG, "Could not raise the call volume", e)
        }
    }

    private fun restoreCallVolume() {
        val volume = savedCallVolume ?: return
        savedCallVolume = null
        try {
            audioManager.setStreamVolume(AudioManager.STREAM_VOICE_CALL, volume, 0)
        } catch (e: RuntimeException) {
            Log.w(TAG, "Could not restore the call volume", e)
        }
    }

    private fun requestSpeakerViaAudioManager() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val speaker = audioManager.availableCommunicationDevices
                .firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
            if (speaker != null && audioManager.setCommunicationDevice(speaker)) speakerRequestedByUs = true
        } else {
            @Suppress("DEPRECATION")
            audioManager.isSpeakerphoneOn = true
        }
    }

    @SuppressLint("MissingPermission") // Checked explicitly.
    private fun showInCallScreen() {
        if (!granted(Manifest.permission.READ_PHONE_STATE)) return
        try {
            getSystemService(TelecomManager::class.java)?.showInCallScreen(false)
        } catch (e: SecurityException) {
            Log.w(TAG, "showInCallScreen rejected", e)
        }
    }

    @SuppressLint("MissingPermission") // Checked explicitly.
    private fun endCallViaTelecom(): Boolean {
        if (!granted(Manifest.permission.ANSWER_PHONE_CALLS)) return false
        return try {
            @Suppress("DEPRECATION")
            getSystemService(TelecomManager::class.java)?.endCall() == true
        } catch (e: SecurityException) {
            Log.w(TAG, "endCall rejected", e)
            false
        }
    }

    /**
     * Clicks the first clickable control in another app's window whose text or content
     * description matches one of [labels] (e.g. the dialer's "Speaker" toggle). [exact] requires
     * the whole label to match; [onlyIfChecked] clicks only a toggle that is currently on
     * (to switch it back off); [skipIfChecked] treats a toggle that is already on as done.
     */
    /**
     * Nodes in other apps' windows whose text or description is one of [labels], best label
     * first. The tree is walked directly: findAccessibilityNodeInfosByText does not reach
     * Jetpack Compose UIs such as the Google Phone dialer.
     */
    private fun dialerNodes(labels: List<String>, exact: Boolean): List<AccessibilityNodeInfo> {
        val wanted = labels.map { it.lowercase() }
        fun rank(node: AccessibilityNodeInfo): Int {
            for (raw in listOf(node.contentDescription, node.text)) {
                val shown = raw?.toString()?.trim()?.lowercase() ?: continue
                val i = wanted.indexOfFirst { shown == it || (!exact && shown.startsWith("$it ")) }
                if (i >= 0) return i
            }
            return -1
        }
        val roots = buildList {
            windows?.forEach { window -> window.root?.let(::add) }
            rootInActiveWindow?.let(::add)
        }.filter { it.packageName?.toString() != packageName }.distinctBy { it.windowId }
        val found = mutableListOf<Pair<Int, AccessibilityNodeInfo>>()
        for (root in roots) {
            val stack = ArrayDeque<AccessibilityNodeInfo>().apply { add(root) }
            var visited = 0
            while (stack.isNotEmpty() && visited < MAX_NODES_PER_WINDOW) {
                val node = stack.removeLast()
                visited++
                if (!node.isVisibleToUser) continue
                val r = rank(node)
                if (r >= 0) found += r to node
                for (i in node.childCount - 1 downTo 0) node.getChild(i)?.let(stack::add)
            }
        }
        return found.sortedBy { it.first }.map { it.second }
    }

    private fun findDialerNode(labels: List<String>, exact: Boolean): AccessibilityNodeInfo? =
        dialerNodes(labels, exact).firstOrNull()

    /** Taps the centre of [node] with a real touch gesture (for buttons that ignore
     *  accessibility clicks). */
    private fun tapNode(node: AccessibilityNodeInfo): Boolean {
        val bounds = Rect().also(node::getBoundsInScreen)
        if (bounds.isEmpty) return false
        val path = Path().apply { moveTo(bounds.exactCenterX(), bounds.exactCenterY()) }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, TAP_MS))
            .build()
        return dispatchGesture(gesture, null, null)
    }

    private fun clickNodeWithLabel(
        labels: List<String>,
        skipIfChecked: Boolean = false,
        exact: Boolean = false,
        onlyIfChecked: Boolean = false,
    ): Boolean {
        for (node in dialerNodes(labels, exact)) {
            val shown = (node.contentDescription ?: node.text)?.toString()?.trim()
            var target: AccessibilityNodeInfo? = node
            var depth = 0
            while (target != null && !target.isClickable && depth < 4) {
                target = target.parent
                depth++
            }
            if (target == null || !target.isClickable || !target.isEnabled) {
                Log.i(TAG, "UI: '$shown' found but nothing clickable to press")
                continue
            }
            val on = target.isChecked || node.isChecked || target.isSelected
            if (skipIfChecked && on) {
                Log.i(TAG, "UI: '$shown' already on")
                return true
            }
            if (onlyIfChecked && !on) continue
            if (target.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                Log.i(TAG, "UI: pressed '$shown'")
                return true
            }
            Log.i(TAG, "UI: press on '$shown' was rejected")
        }
        return false
    }

    // ------------------------------------------------------------------------------------------

    private fun voiceprintMessage(embedding: FloatArray): String =
        JSONObject().put("action", "owner_voiceprint")
            .put("embedding", JSONArray().apply { embedding.forEach { put(it.toDouble()) } })
            .toString()

    private fun callMetaMessage(): String =
        JSONObject().put("action", "call_meta")
            .put("peer", peerLabel)
            .put("report_token", reportToken)
            .put("record", true)
            .toString()

    private fun granted(permission: String) =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun toast(message: String) {
        Toast.makeText(applicationContext, message, Toast.LENGTH_LONG).show()
    }

    companion object {
        private const val TAG = "CallGuardService"
        private const val CALL_END_DEBOUNCE_MS = 1_500L
        private const val SPEAKER_RETRY_MS = 1_500L
        private const val SPEAKER_VERIFY_MS = 2_000L
        private const val SPEAKER_SCREEN_MS = 800L
        private const val TAP_MS = 60L
        private const val SPEAKER_MAX_ATTEMPTS = 4
        private const val SENIOR_HANGUP_DELAY_MS = 2_000L
        private const val HOLD_PRESS_DELAY_MS = 600L
        private const val MENU_OPEN_DELAY_MS = 500L
        private const val HOLD_RETRY_MS = 500L
        private const val HOLD_MAX_ATTEMPTS = 8
        private const val MAX_NODES_PER_WINDOW = 3000

        /** How long after a call the report (recording) stays available server-side. */
        const val REPORT_WINDOW_MS = 2 * 60 * 1000L

        private val NUMBER_RE = Regex("""[+(]?\d[\d\s().-]{6,16}\d""")
        private val SYSTEM_PACKAGES = setOf("com.android.systemui", "android")
        private val NON_PEER_WORDS = listOf(
            "call", "speaker", "mute", "keypad", "hold", "video", "add", "end", "dialing",
            "calling", "incoming", "connected", "min", "sec", "hd", "signal", "battery",
            "wi-fi", "wifi", "bluetooth", "charging", "silence", "message", "contacts", "merge",
            "record", "unknown", "mobile", "india", "pause",
            // The app's own name shows up in other windows (accessibility shortcut, launcher).
            "vishywashy", "callguard", "button",
        )

        data class ReportHandle(
            val sessionId: String, val token: String, val peer: String,
            val endpointDisplay: String, val expiresAtMs: Long,
        ) { val valid: Boolean get() = System.currentTimeMillis() < expiresAtMs }

        /** The most recent flagged call still within its report window, for the app UI. */
        @Volatile
        var lastReport: ReportHandle? = null

        /** Both parties if the device allows it; otherwise the microphone, unprocessed for ASR. */
        private val CALL_AUDIO_SOURCES = listOf(
            MediaRecorder.AudioSource.VOICE_CALL,
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC,
        )

        private val SPEAKER_LABELS = listOf("Speaker", "Speakerphone", "Loudspeaker", "स्पीकर")
        private val END_CALL_LABELS = listOf("End call", "Hang up", "End", "कॉल समाप्त करें")
        private val HOLD_LABELS = listOf("Hold", "Hold call", "Put on hold", "Put call on hold", "Hold the call",
            "होल्ड", "होल्ड करें", "कॉल होल्ड करें", "कॉल होल्ड", "होल्ड पर रखें")
        private val ON_HOLD_LABELS = listOf("On hold", "Call on hold", "Held", "होल्ड पर", "कॉल होल्ड पर है")
        private val RESUME_LABELS = listOf("Resume", "Resume call", "Unhold", "Unhold call", "Un-hold", "Take off hold",
            "रिज़्यूम", "होल्ड हटाएं")
        private val MORE_LABELS = listOf("Show more", "More", "More options", "More call options", "More actions",
            "Options", "और दिखाएं", "अधिक", "और")
        private val MUTE_LABELS = listOf("Mute", "Mute call", "Mute microphone", "म्यूट", "म्यूट करें")
        private val UNMUTE_LABELS = listOf("Unmute", "Unmute call", "Unmute microphone", "अनम्यूट")

        /** True while the system has the service bound (the user enabled it in Accessibility settings). */
        @Volatile
        var connected: Boolean = false
            private set

        /** Whether the user has switched the service on in Accessibility settings. */
        fun isEnabled(context: Context): Boolean {
            val expected = ComponentName(context, CallGuardService::class.java)
            val enabled = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
            ) ?: return false
            return enabled.split(':').any { ComponentName.unflattenFromString(it) == expected }
        }

        /** Consent and microphone permission; checked again at every call. */
        fun prerequisitesMet(context: Context): Boolean =
            DpdpConsentManager(context).hasValidConsent() &&
                ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED &&
                EdgeEndpoint.load(context) != null
    }
}
