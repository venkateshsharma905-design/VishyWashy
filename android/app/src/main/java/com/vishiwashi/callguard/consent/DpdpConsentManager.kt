package com.vishiwashi.callguard.consent

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.DialogInterface
import android.graphics.Typeface
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.StyleSpan
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Records and enforces consent under India's Digital Personal Data Protection Act, 2023.
 *
 * Design points:
 *  - Notice before consent (s.5): what is processed, why, how to withdraw, how to complain.
 *  - Consent must be a clear affirmative action (s.6): an unticked box gates the positive button.
 *  - Purpose limitation: consent is stored per purpose id and bound to [NOTICE_VERSION]; changing
 *    the notice text requires bumping the version, which invalidates prior consent.
 *  - Withdrawal is as easy as giving consent: one call here, one tap in the HUD notification,
 *    and it stops processing immediately.
 *  - An append-only local ledger of grant/withdraw events supports accountability.
 *
 * This is an engineering implementation, not legal advice — have counsel review the notice text,
 * especially regarding the *other* party on the call, whose voice is also processed.
 */
class DpdpConsentManager(context: Context) {

    data class ConsentRecord(
        val noticeVersion: Int,
        val purposes: Set<String>,
        val grantedAtEpochMs: Long,
        val noticeLocale: String,
    )

    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun currentConsent(): ConsentRecord? {
        val version = prefs.getInt(KEY_NOTICE_VERSION, -1)
        if (version < 0) return null
        return ConsentRecord(
            noticeVersion = version,
            purposes = prefs.getStringSet(KEY_PURPOSES, emptySet()).orEmpty().toSet(),
            grantedAtEpochMs = prefs.getLong(KEY_GRANTED_AT, 0L),
            noticeLocale = prefs.getString(KEY_LOCALE, "").orEmpty(),
        )
    }

    fun hasValidConsent(): Boolean {
        val record = currentConsent() ?: return false
        return record.noticeVersion == NOTICE_VERSION && record.purposes.containsAll(REQUIRED_PURPOSES)
    }

    fun recordConsent() {
        val now = System.currentTimeMillis()
        prefs.edit()
            .putInt(KEY_NOTICE_VERSION, NOTICE_VERSION)
            .putStringSet(KEY_PURPOSES, REQUIRED_PURPOSES)
            .putLong(KEY_GRANTED_AT, now)
            .putString(KEY_LOCALE, Locale.getDefault().toLanguageTag())
            .putString(KEY_LEDGER, appendToLedger(EVENT_GRANTED, now))
            .commit()
    }

    /**
     * Withdraws consent. The call service observes these preferences and stops any running
     * analysis immediately.
     */
    fun withdrawConsent() {
        val now = System.currentTimeMillis()
        prefs.edit()
            .remove(KEY_NOTICE_VERSION)
            .remove(KEY_PURPOSES)
            .remove(KEY_GRANTED_AT)
            .remove(KEY_LOCALE)
            .putString(KEY_LEDGER, appendToLedger(EVENT_WITHDRAWN, now))
            .commit()
    }

    /** Entries formatted as `epochMs|EVENT|vN`, oldest first. */
    fun ledger(): List<String> =
        prefs.getString(KEY_LEDGER, null)?.lines()?.filter { it.isNotBlank() }.orEmpty()

    /**
     * Shows the consent notice. [onResult] is invoked exactly once: `true` only after the user
     * ticks the acknowledgement box and taps "I consent" (consent is recorded before the callback).
     */
    fun showConsentDialog(activity: Activity, onResult: (granted: Boolean) -> Unit) {
        if (activity.isFinishing || activity.isDestroyed) {
            onResult(false)
            return
        }

        var delivered = false
        fun deliver(granted: Boolean) {
            if (!delivered) {
                delivered = true
                onResult(granted)
            }
        }

        val padding = activity.dp(20)
        val notice = TextView(activity).apply {
            text = buildNotice()
            textSize = 14f
            setLineSpacing(0f, 1.15f)
        }
        val acknowledgement = CheckBox(activity).apply {
            text = "I have read this notice and I consent to real-time analysis of call audio " +
                "for scam detection."
            textSize = 14f
            isChecked = false
            setPadding(paddingLeft, activity.dp(12), paddingRight, activity.dp(12))
        }
        val content = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding / 2, padding, 0)
            addView(notice)
            addView(acknowledgement)
        }

        val dialog = AlertDialog.Builder(activity)
            .setTitle("Consent for real-time call protection")
            .setView(ScrollView(activity).apply { addView(content) })
            .setPositiveButton("I consent") { _, _ ->
                recordConsent()
                deliver(true)
            }
            .setNegativeButton("Not now") { _, _ -> deliver(false) }
            .setOnCancelListener { deliver(false) }
            .create()

        dialog.setCanceledOnTouchOutside(false)
        dialog.setOnShowListener {
            val positive = dialog.getButton(DialogInterface.BUTTON_POSITIVE)
            positive.isEnabled = false
            acknowledgement.setOnCheckedChangeListener { _, checked -> positive.isEnabled = checked }
        }
        dialog.show()
    }

    private fun buildNotice(): CharSequence {
        val text = SpannableStringBuilder()
        fun heading(value: String) {
            val start = text.length
            text.append(value).append('\n')
            text.setSpan(StyleSpan(Typeface.BOLD), start, start + value.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        fun paragraph(value: String) {
            text.append(value).append("\n\n")
        }

        paragraph(
            "VishyWashy can warn you about likely scam calls while they happen. " +
                "This needs your explicit permission to process audio during calls.",
        )
        heading("When it runs")
        paragraph(
            "Only while a phone or internet call is connected, and only while protection is armed. " +
                "Nothing is captured between calls.",
        )
        heading("What is processed")
        paragraph(
            "Call audio: your voice and the voice of the person you are speaking with. Where the " +
                "phone does not give apps the call audio directly, VishyWashy switches the call to " +
                "loudspeaker so the microphone can hear both sides (you can turn this off). " +
                "VishyWashy uses Android's accessibility service to detect calls, show its warning " +
                "above the dialer and press the loudspeaker button; it does not read your screen " +
                "for any other purpose.",
        )
        heading("Purpose")
        paragraph(
            "Only to compute a live scam-risk score and show warnings on your screen. " +
                "It is not used for advertising, profiling or any other purpose.",
        )
        heading("Your voiceprint (optional)")
        paragraph(
            "If you record a short voice sample, the server turns it into a voiceprint (a list of " +
                "numbers describing your voice) so the transcript can show which words are yours and " +
                "which are the caller's. The voiceprint is stored only on this phone and sent with " +
                "each call; the server does not keep it or the recording. Delete it any time in the app.",
        )
        heading("How it is handled")
        paragraph(
            "Audio is sent in 2.5-second segments to your configured analysis " +
                "server and processed in memory. Raw audio and transcripts are not stored. Risk " +
                "scores, timing and matched warning keywords are kept only for the active session.",
        )
        heading("Your choices and rights")
        paragraph(
            "Nothing is recorded unless you agree below. You can withdraw consent at any time from " +
                "this app or from the VishyWashy notification; analysis stops immediately. " +
                "Grievances: $GRIEVANCE_CONTACT. You may also complain to the Data Protection " +
                "Board of India.",
        )
        return text.trimEnd()
    }

    private fun appendToLedger(event: String, atEpochMs: Long): String =
        (ledger() + "$atEpochMs|$event|v$NOTICE_VERSION")
            .takeLast(MAX_LEDGER_ENTRIES)
            .joinToString("\n")

    private fun Context.dp(value: Int): Int = (value * resources.displayMetrics.density).roundToInt()

    companion object {
        /** Bump whenever the notice text or purposes change; previously given consent becomes invalid. */
        const val NOTICE_VERSION = 3

        const val PURPOSE_REALTIME_SCAM_ANALYSIS = "realtime_call_audio_scam_analysis"
        val REQUIRED_PURPOSES: Set<String> = setOf(PURPOSE_REALTIME_SCAM_ANALYSIS)

        // TODO: replace with your Data Fiduciary's grievance officer contact before release.
        const val GRIEVANCE_CONTACT = "grievance@vishiwashi.example"

        /** Observed by the call service so withdrawal stops analysis immediately. */
        const val PREFS_NAME = "dpdp_consent"
        private const val KEY_NOTICE_VERSION = "notice_version"
        private const val KEY_PURPOSES = "purposes"
        private const val KEY_GRANTED_AT = "granted_at"
        private const val KEY_LOCALE = "notice_locale"
        private const val KEY_LEDGER = "ledger"
        private const val EVENT_GRANTED = "GRANTED"
        private const val EVENT_WITHDRAWN = "WITHDRAWN"
        private const val MAX_LEDGER_ENTRIES = 50
    }
}
