package com.vishiwashi.callguard.call

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Handler
import android.os.Looper
import com.vishiwashi.callguard.settings.EdgeEndpoint
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.IOException

/**
 * Hands a just-finished scam call to the cyber cell. By policy the call recording is never
 * exported to the user as a file they could keep: it stays on the server for two minutes and is
 * delivered to the cyber cell only, as a secure link inside a message addressed to the
 * user-configured cyber-cell number. This builds that message (addressed, pre-filled) for the user to send; no
 * audio file is downloaded to the phone and no general share sheet is offered.
 */
class CallReporter(
    private val appContext: Context,
    private val httpClient: OkHttpClient,
) {
    interface Listener {
        /** A messaging intent addressed only to the cyber-cell number, pre-filled. */
        fun onReady(messageIntent: Intent)
        fun onUnavailable(message: String)
    }

    private val main = Handler(Looper.getMainLooper())

    fun prepare(endpoint: EdgeEndpoint, sessionId: String, token: String, reportNumber: String, listener: Listener) {
        Thread({ run(endpoint, sessionId, token, reportNumber, listener) }, "CallGuard-Report").start()
    }

    private fun run(endpoint: EdgeEndpoint, sessionId: String, token: String, reportNumber: String, listener: Listener) {
        val base = "${endpoint.httpBase}/api/report/$sessionId"
        val meta = try {
            httpClient.newCall(Request.Builder().url("$base?token=$token").build()).execute().use { response ->
                if (response.code == 404) return fail(listener, "This report has expired. Recordings are kept for only two minutes after a call ends.")
                if (!response.isSuccessful) return fail(listener, "Couldn't fetch the report (HTTP ${response.code}).")
                JSONObject(response.body?.string().orEmpty().ifBlank { "{}" })
            }
        } catch (e: IOException) {
            return fail(listener, "Can't reach the server: ${e.message ?: e.javaClass.simpleName}")
        }
        if (!meta.optBoolean("ok")) return fail(listener, meta.optString("detail").ifBlank { "Report unavailable." })

        val viewUrl = "${endpoint.httpBase}${meta.optString("view_url")}"
        val helpline = meta.optString("helpline", "1930")
        val peer = meta.optString("peer_label").ifBlank { "unknown number" }
        val durationS = meta.optDouble("duration_s", 0.0)
        val risk = (meta.optDouble("peak_risk", 0.0) * 100).toInt()
        val body = buildString {
            append("VishyWashy scam report (national helpline $helpline)\n")
            append("From: ").append(peer).append('\n')
            append("Duration: ").append(durationS.toInt()).append("s · peak risk ").append(risk).append("%\n")
            append("Transcript & recording (secure, expires in 2 min):\n").append(viewUrl)
        }
        // smsto: addresses the message only to the cyber-cell number; no file is attached, so the
        // user cannot save or redirect the recording.
        val digits = reportNumber.filter { it.isDigit() || it == '+' }
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$digits")).apply {
            putExtra("sms_body", body)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        if (intent.resolveActivity(appContext.packageManager) == null) {
            return fail(listener, "No messaging app found to send the report.")
        }
        main.post { listener.onReady(intent) }
    }

    private fun fail(listener: Listener, message: String) {
        main.post { listener.onUnavailable(message) }
    }
}
