package com.vishiwashi.callguard.settings

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * A local log of calls CallGuard watched: when, who, how risky, and what happened. Stored only on
 * this phone, and never includes audio — the recording itself lives on the server for two minutes
 * only (see the report flow). The History tab reads this.
 */
class CallHistory(context: Context) {

    enum class Outcome { MONITORED, FLAGGED, HUNG_UP, REPORTED }

    data class Record(
        val id: String,
        val startedAtMs: Long,
        val durationS: Int,
        val peer: String,
        val peakRisk: Double,
        val outcome: Outcome,
    ) {
        val flagged: Boolean get() = outcome != Outcome.MONITORED
    }

    private val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun records(): List<Record> {
        val array = JSONArray(prefs.getString(KEY, "[]"))
        return (0 until array.length()).mapNotNull { read(array.optJSONObject(it)) }
            .sortedByDescending { it.startedAtMs }
    }

    /** Insert or update a record (same [Record.id] overwrites — e.g. when a call is later reported). */
    fun put(record: Record) {
        val kept = records().filter { it.id != record.id }.take(MAX - 1)
        val array = JSONArray()
        (listOf(record) + kept).forEach { array.put(write(it)) }
        prefs.edit().putString(KEY, array.toString()).apply()
    }

    fun setOutcome(id: String, outcome: Outcome) {
        records().firstOrNull { it.id == id }?.let { put(it.copy(outcome = outcome)) }
    }

    fun clear() = prefs.edit().remove(KEY).apply()

    private fun write(r: Record) = JSONObject().apply {
        put("id", r.id); put("t", r.startedAtMs); put("dur", r.durationS)
        put("peer", r.peer); put("risk", r.peakRisk); put("outcome", r.outcome.name)
    }

    private fun read(o: JSONObject?): Record? {
        o ?: return null
        return runCatching {
            Record(
                id = o.getString("id"),
                startedAtMs = o.getLong("t"),
                durationS = o.optInt("dur"),
                peer = o.optString("peer"),
                peakRisk = o.optDouble("risk", 0.0),
                outcome = runCatching { Outcome.valueOf(o.optString("outcome")) }.getOrDefault(Outcome.MONITORED),
            )
        }.getOrNull()
    }

    companion object {
        private const val PREFS = "callguard_history"
        private const val KEY = "records"
        private const val MAX = 200
    }
}
