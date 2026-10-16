package com.vishiwashi.callguard.call

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.vishiwashi.callguard.audio.AudioStreamManager
import com.vishiwashi.callguard.settings.CallHistory
import com.vishiwashi.callguard.settings.EdgeEndpoint
import com.vishiwashi.callguard.settings.ProtectionSettings

/**
 * Invisible launcher for the report share sheet. Started from the report notification, the senior-
 * mode button, or the app's History. It fetches the just-ended call's transcript and recording
 * from the server and hands the user Android's share sheet, addressed to the configured cyber-cell
 * number. The user chooses the app and sends; nothing is sent automatically.
 */
class ReportActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val sessionId = intent.getStringExtra(EXTRA_SESSION)
        val token = intent.getStringExtra(EXTRA_TOKEN)
        val peer = intent.getStringExtra(EXTRA_PEER).orEmpty()
        val endpoint = EdgeEndpoint.load(this)
        if (sessionId == null || token == null || endpoint == null) {
            Toast.makeText(this, "Nothing to report.", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        Toast.makeText(this, "Preparing report…", Toast.LENGTH_SHORT).show()
        val reporter = CallReporter(applicationContext, AudioStreamManager.defaultHttpClient())
        reporter.prepare(endpoint, sessionId, token, ProtectionSettings(this).reportNumber, object : CallReporter.Listener {
            override fun onReady(shareIntent: Intent) {
                CallHistory(this@ReportActivity).setOutcome(sessionId, CallHistory.Outcome.REPORTED)
                startActivity(shareIntent)
                finish()
            }

            override fun onUnavailable(message: String) {
                Toast.makeText(this@ReportActivity, message, Toast.LENGTH_LONG).show()
                finish()
            }
        })
    }

    companion object {
        private const val EXTRA_SESSION = "session_id"
        private const val EXTRA_TOKEN = "token"
        private const val EXTRA_PEER = "peer"

        fun intent(context: Context, sessionId: String, token: String, peer: String): Intent =
            Intent(context, ReportActivity::class.java)
                .putExtra(EXTRA_SESSION, sessionId)
                .putExtra(EXTRA_TOKEN, token)
                .putExtra(EXTRA_PEER, peer)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

        fun launch(context: Context, sessionId: String, token: String, peer: String) {
            context.startActivity(intent(context, sessionId, token, peer))
        }
    }
}
