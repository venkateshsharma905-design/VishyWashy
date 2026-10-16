package com.vishiwashi.callguard.call

import android.app.Notification
import com.vishiwashi.callguard.R
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import androidx.core.app.NotificationCompat
import com.vishiwashi.callguard.call.CallGuardService.Companion.REPORT_WINDOW_MS

/**
 * After a flagged call ends, offers "Report to cyber cell" as a notification for the two minutes
 * the recording stays available on the server. Tapping it opens [ReportActivity], which lets the
 * user send the transcript and recording themselves.
 */
object ReportNotifier {

    private const val CHANNEL_ID = "callguard_report"
    private const val NOTIFICATION_ID = 0x1931

    fun show(context: Context, sessionId: String, token: String, peer: String) {
        val manager = context.getSystemService(NotificationManager::class.java)
        ensureChannel(manager)
        if (!manager.areNotificationsEnabled()) return

        val intent = ReportActivity.intent(context, sessionId, token, peer)
        val pending = PendingIntent.getActivity(
            context, sessionId.hashCode(), intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val who = peer.ifBlank { "this caller" }
        val notification: Notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_logo)
            .setContentTitle("Suspicious call flagged")
            .setContentText("Report $who to the cyber cell")
            .setStyle(
                NotificationCompat.BigTextStyle().bigText(
                    "VishyWashy flagged $who. You can send the transcript and recording to the cyber " +
                        "cell. Available for 2 minutes, then the recording is deleted.",
                ),
            )
            .setCategory(NotificationCompat.CATEGORY_RECOMMENDATION)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setTimeoutAfter(REPORT_WINDOW_MS)
            .setContentIntent(pending)
            .addAction(0, "Report to cyber cell", pending)
            .build()
        manager.notify(NOTIFICATION_ID, notification)
    }

    private fun ensureChannel(manager: NotificationManager) {
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Scam call reports", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "Offers to report a flagged call to the cyber cell"
            },
        )
    }
}
