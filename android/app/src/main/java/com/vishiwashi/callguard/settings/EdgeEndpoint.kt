package com.vishiwashi.callguard.settings

import android.content.Context
import com.vishiwashi.callguard.BuildConfig
import java.net.URI
import java.net.URISyntaxException

/**
 * The edge server the app streams to, entered by the user as anything from `192.168.1.20` or
 * `192.168.1.20:8000` to `https://callguard.example.com` or a full `ws://…/ws/telemetry` URL.
 *
 * Plain hosts default to port [DEFAULT_PORT] (what `python server.py` binds), so the address
 * printed in the server console can be typed in as-is.
 */
data class EdgeEndpoint(
    /** Telemetry socket base; [com.vishiwashi.callguard.audio.AudioStreamManager] appends `/{session}`. */
    val webSocketBase: String,
    /** HTTP origin for REST calls such as `/api/health`. */
    val httpBase: String,
    /** Normalised form shown back to the user. */
    val display: String,
) {
    val healthUrl: String get() = "$httpBase/api/health"

    companion object {
        const val DEFAULT_PORT = 8000
        private const val TELEMETRY_PATH = "/ws/telemetry"
        private const val PREFS = "callguard_settings"
        private const val KEY_ADDRESS = "edge_address"

        /** Returns null when [input] cannot be a server address. */
        fun parse(input: String): EdgeEndpoint? {
            val raw = input.trim().trimEnd('/')
            if (raw.isEmpty() || raw.any { it.isWhitespace() }) return null
            val withScheme = if ("://" in raw) raw else "http://$raw"
            val uri = try {
                URI(withScheme)
            } catch (e: URISyntaxException) {
                return null
            }
            val host = uri.host?.takeIf { it.isNotBlank() } ?: return null

            val secure = when (uri.scheme?.lowercase()) {
                "http", "ws" -> false
                "https", "wss" -> true
                else -> return null
            }
            val port = when {
                uri.port in 1..65535 -> uri.port
                secure -> -1                     // implicit 443
                else -> DEFAULT_PORT
            }
            val hostPart = if (':' in host && !host.startsWith("[")) "[$host]" else host
            val authority = if (port > 0) "$hostPart:$port" else hostPart
            return EdgeEndpoint(
                webSocketBase = "${if (secure) "wss" else "ws"}://$authority$TELEMETRY_PATH",
                httpBase = "${if (secure) "https" else "http"}://$authority",
                display = if (secure) "https://$authority" else authority,
            )
        }

        /** The saved endpoint, else the build-time default ([BuildConfig.EDGE_WS_URL]). */
        fun load(context: Context): EdgeEndpoint? {
            val saved = context.prefs().getString(KEY_ADDRESS, null)
            return saved?.let(::parse) ?: parse(BuildConfig.EDGE_WS_URL)
        }

        fun save(context: Context, endpoint: EdgeEndpoint) {
            context.prefs().edit().putString(KEY_ADDRESS, endpoint.display).apply()
        }

        private fun Context.prefs() = applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    }
}
