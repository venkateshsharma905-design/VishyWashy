package com.vishiwashi.callguard.audio

import org.json.JSONObject

/**
 * One telemetry frame broadcast by the edge server (`FusionResult.to_dict()` plus
 * session metadata) roughly every 500 ms on `/ws/telemetry/{session_id}`.
 */
data class RiskTelemetry(
    /** Risk R in [0, 1]. */
    val riskScore: Double,
    /** Server-side label, e.g. "LOW" / "MEDIUM" / "HIGH". Informational only; the HUD uses [riskScore]. */
    val riskLevel: String,
    /** Server-side end-to-end inference time for the last 2.5 s window. */
    val inferenceLatencyMs: Double,
    val pIntent: Double,
    val keywordMultiplier: Double,
    /** Matched lexical indicator tags (e.g. "otp_request", "urgency"). */
    val indicators: List<String>,
    /** Human-readable guidance from the server, may be empty. */
    val guidance: String,
    val frameCount: Int,
) {
    companion object {
        /** Returns null when [json] is not a risk telemetry frame (e.g. a control event). */
        fun fromJson(json: JSONObject): RiskTelemetry? {
            if (!json.has("risk_score")) return null
            val score = json.optDouble("risk_score", Double.NaN)
            if (score.isNaN()) return null

            val indicators = json.optJSONArray("indicators")?.let { array ->
                (0 until array.length())
                    .mapNotNull { i -> array.optString(i).takeIf { it.isNotBlank() && it != "null" } }
            }.orEmpty()

            return RiskTelemetry(
                riskScore = score.coerceIn(0.0, 1.0),
                riskLevel = json.stringOrEmpty("risk_level"),
                inferenceLatencyMs = json.finiteDouble("latency_ms"),
                pIntent = json.finiteDouble("p_intent"),
                keywordMultiplier = json.finiteDouble("keyword_multiplier"),
                indicators = indicators,
                guidance = json.stringOrEmpty("guidance"),
                frameCount = json.optInt("frame_count", 0),
            )
        }

        private fun JSONObject.stringOrEmpty(key: String): String =
            if (isNull(key)) "" else optString(key, "")

        private fun JSONObject.finiteDouble(key: String): Double =
            optDouble(key, 0.0).takeIf { it.isFinite() } ?: 0.0
    }
}
