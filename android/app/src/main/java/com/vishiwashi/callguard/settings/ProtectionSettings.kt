package com.vishiwashi.callguard.settings

import android.content.Context
import android.content.SharedPreferences

/**
 * User switches for call protection. [CallGuardService][com.vishiwashi.callguard.call.CallGuardService]
 * observes these preferences, so changes take effect immediately, even mid-call.
 */
class ProtectionSettings(context: Context) {

    val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** When true, CallGuard starts automatically whenever a call connects. */
    var armed: Boolean
        get() = prefs.getBoolean(KEY_ARMED, false)
        set(value) = prefs.edit().putBoolean(KEY_ARMED, value).apply()

    /**
     * Switch the call to speakerphone when the phone does not expose call audio directly, so the
     * microphone can hear the other party as well as the user.
     */
    var autoSpeaker: Boolean
        get() = prefs.getBoolean(KEY_AUTO_SPEAKER, true)
        set(value) = prefs.edit().putBoolean(KEY_AUTO_SPEAKER, value).apply()

    /**
     * The owner's voiceprint (192 numbers from the server's speaker model), kept only on this
     * phone and sent at the start of each call so the server can tell "You" from the caller.
     */
    var voiceprint: FloatArray?
        get() = prefs.getString(KEY_VOICEPRINT, null)
            ?.split(',')
            ?.mapNotNull { it.toFloatOrNull() }
            ?.takeIf { it.size == VOICEPRINT_DIMS }
            ?.toFloatArray()
        set(value) = prefs.edit().apply {
            if (value == null) remove(KEY_VOICEPRINT) else putString(KEY_VOICEPRINT, value.joinToString(","))
        }.apply()

    /**
     * When a call reaches HIGH risk, put it on hold (or mute it where there is no Hold button)
     * and let the user choose to continue or hang up.
     */
    var holdOnFraud: Boolean
        get() = prefs.getBoolean(KEY_HOLD_ON_FRAUD, true)
        set(value) = prefs.edit().putBoolean(KEY_HOLD_ON_FRAUD, value).apply()

    /** "system" | "light" | "dark" — the app and HUD theme. */
    var themeMode: String
        get() = prefs.getString(KEY_THEME, THEME_SYSTEM) ?: THEME_SYSTEM
        set(value) = prefs.edit().putString(KEY_THEME, value).apply()

    /**
     * Senior-citizen mode: the HUD opens straight to a large "Hang up & report" button on any
     * risk, and taps are bigger. Designed for one-tap action.
     */
    var seniorMode: Boolean
        get() = prefs.getBoolean(KEY_SENIOR, false)
        set(value) = prefs.edit().putBoolean(KEY_SENIOR, value).apply()

    /** Where "Report to cyber cell" is addressed. Fixed: it cannot be changed in the app, so a
     *  scammer can't talk the user into redirecting reports (a number saved by older versions
     *  is ignored). */
    val reportNumber: String
        get() = DEFAULT_REPORT_NUMBER

    companion object {
        const val VOICEPRINT_DIMS = 192
        const val THEME_SYSTEM = "system"
        const val THEME_LIGHT = "light"
        const val THEME_DARK = "dark"
        /** Cyber-cell report destination (fixed). */
        const val DEFAULT_REPORT_NUMBER = "+91 8383853196"
        private const val KEY_THEME = "theme_mode"
        private const val KEY_SENIOR = "senior_mode"
        private const val KEY_VOICEPRINT = "owner_voiceprint"
        private const val KEY_HOLD_ON_FRAUD = "hold_on_fraud"

        /** Shared with [EdgeEndpoint], which stores the server address in the same file. */
        const val PREFS = "callguard_settings"
        const val KEY_ARMED = "armed"
        const val KEY_AUTO_SPEAKER = "auto_speaker"
    }
}
