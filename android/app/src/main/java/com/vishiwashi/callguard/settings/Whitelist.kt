package com.vishiwashi.callguard.settings

import android.content.Context

/**
 * Numbers (or saved-contact names) the user never wants monitored. A call from a whitelisted
 * entry does not start [com.vishiwashi.callguard.call.CallGuardService] at all — no audio is
 * captured, nothing is streamed, no recording is kept.
 *
 * Matching is lenient: a stored number matches when the last [MATCH_DIGITS] digits agree (so
 * +91 98765 43210, 098765 43210 and 9876543210 are the same), and a stored name matches when it
 * appears in the text shown on the call screen.
 */
class Whitelist(context: Context) {

    data class Entry(val label: String, val digits: String) {
        val isNumber: Boolean get() = digits.isNotEmpty()
    }

    private val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun entries(): List<Entry> =
        prefs.getStringSet(KEY, emptySet()).orEmpty()
            .mapNotNull { decode(it) }
            .sortedBy { it.label.lowercase() }

    /** Adds a contact picked from the phone book: stored by number, shown by name. */
    fun addContact(name: String, number: String): Boolean {
        val digits = national(number)
        if (digits.length < 4) return false
        val set = prefs.getStringSet(KEY, emptySet()).orEmpty().toMutableSet()
        if (!set.add(encode(Entry(label = name.ifBlank { number }, digits = digits)))) return false
        prefs.edit().putStringSet(KEY, set).apply()
        return true
    }

    fun add(raw: String): Boolean {
        val entry = parse(raw) ?: return false
        val set = prefs.getStringSet(KEY, emptySet()).orEmpty().toMutableSet()
        if (!set.add(encode(entry))) return false
        prefs.edit().putStringSet(KEY, set).apply()
        return true
    }

    /** Puts back an entry removed with [remove] (the "Undo" of a removal). */
    fun restore(entry: Entry) {
        val set = prefs.getStringSet(KEY, emptySet()).orEmpty().toMutableSet()
        if (set.add(encode(entry))) prefs.edit().putStringSet(KEY, set).apply()
    }

    fun remove(entry: Entry) {
        val set = prefs.getStringSet(KEY, emptySet()).orEmpty().toMutableSet()
        set.remove(encode(entry))
        prefs.edit().putStringSet(KEY, set).apply()
    }

    /** True when any whitelisted entry matches the number and/or on-screen text of a call. */
    fun matches(number: String?, screenText: String?): Boolean {
        val digits = normalize(number)
        val haystack = screenText?.lowercase().orEmpty()
        return entries().any { entry ->
            when {
                entry.isNumber && digits.isNotEmpty() && digitsMatch(entry.digits, digits) -> true
                entry.isNumber && haystack.isNotEmpty() -> digitsMatch(entry.digits, normalize(haystack))
                else -> entry.label.length >= 2 && haystack.contains(entry.label.lowercase())
            }
        }
    }

    companion object {
        const val MATCH_DIGITS = 10
        private const val PREFS = "callguard_whitelist"
        private const val KEY = "entries"

        private fun normalize(value: String?): String = value?.filter { it.isDigit() }.orEmpty()

        /**
         * The national 10-digit number: strips a leading country code (91) or trunk 0, so a bare
         * number, a 0-prefixed one and a +91 one all compare equal. No prefix needs to be typed.
         */
        private fun national(value: String?): String {
            var d = normalize(value)
            if (d.length > 10 && d.startsWith("91")) d = d.substring(2)
            if (d.length == 11 && d.startsWith("0")) d = d.substring(1)
            return d.takeLast(MATCH_DIGITS)
        }

        private fun digitsMatch(a: String, b: String): Boolean {
            val na = national(a); val nb = national(b)
            return na.isNotEmpty() && na == nb
        }

        /** A pure-digit entry becomes a number; anything else is kept as a name to match on screen. */
        private fun parse(raw: String): Entry? {
            val trimmed = raw.trim()
            if (trimmed.isEmpty()) return null
            val digits = national(trimmed)
            return if (digits.length >= 4 && trimmed.all { it.isDigit() || it in "+-() " }) {
                Entry(label = trimmed, digits = digits)
            } else {
                Entry(label = trimmed, digits = "")
            }
        }

        private fun encode(entry: Entry): String = "${entry.digits}\u0001${entry.label}"

        private fun decode(stored: String): Entry? {
            val parts = stored.split('\u0001', limit = 2)
            return if (parts.size == 2) Entry(label = parts[1], digits = parts[0]) else null
        }
    }
}
