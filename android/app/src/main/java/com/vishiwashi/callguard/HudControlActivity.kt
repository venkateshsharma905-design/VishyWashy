package com.vishiwashi.callguard

import android.Manifest
import android.animation.ValueAnimator
import android.app.UiModeManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.ContactsContract
import android.provider.Settings
import android.text.InputType
import android.text.format.DateUtils
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.animation.doOnEnd
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.doOnPreDraw
import androidx.core.view.updatePadding
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import com.vishiwashi.callguard.audio.AudioStreamManager
import com.vishiwashi.callguard.call.CallGuardService
import com.vishiwashi.callguard.call.ReportActivity
import com.vishiwashi.callguard.consent.HudLaunchCoordinator
import com.vishiwashi.callguard.settings.CallHistory
import com.vishiwashi.callguard.settings.EdgeEndpoint
import com.vishiwashi.callguard.settings.ProtectionSettings
import com.vishiwashi.callguard.settings.VoiceEnrollment
import com.vishiwashi.callguard.settings.Whitelist
import com.vishiwashi.callguard.ui.Neu
import com.vishiwashi.callguard.ui.NeuPalette
import java.io.IOException
import java.util.concurrent.TimeUnit
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONObject

/**
 * CallGuard home: a dark-neumorphism UI with four tabs — Protect (arm / status / report), History
 * (watched calls), Trusted (numbers to skip), and Settings (server, voice, theme, senior mode).
 */
class HudControlActivity : ComponentActivity() {

    private lateinit var settings: ProtectionSettings
    private lateinit var coordinator: HudLaunchCoordinator
    private lateinit var neu: Neu
    private lateinit var palette: NeuPalette
    private lateinit var content: FrameLayout
    private lateinit var tabBar: LinearLayout

    private var currentTab = Tab.PROTECT
    private var firstResume = true
    private var themeOnCreate = ""

    private val httpClient: OkHttpClient by lazy {
        AudioStreamManager.defaultHttpClient().newBuilder().callTimeout(6, TimeUnit.SECONDS).build()
    }
    private val enrollment by lazy { VoiceEnrollment(httpClient) }
    private var pendingTest: Call? = null

    private val micPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    private val pickContact = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val uri = result.data?.data ?: return@registerForActivityResult
        val picked = readPickedContact(uri) ?: run { toast("Couldn't read that contact's number."); return@registerForActivityResult }
        if (Whitelist(this).addContact(picked.first, picked.second)) rebuild() else toast("Already trusted.")
    }

    private enum class Tab(val label: String, val icon: String) {
        PROTECT("Protect", "🛡"), HISTORY("History", "🕒"),
        WHITELIST("Trusted", "✓"), SETTINGS("Settings", "⚙")
    }

    private companion object {
        const val UNDO_MS = 5_000
        val OWN_NAMES = listOf("VishyWashy", "CallGuard")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        settings = ProtectionSettings(this)
        applyNightMode(settings.themeMode)
        themeOnCreate = settings.themeMode
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        palette = NeuPalette.resolve(this, settings.themeMode)
        neu = Neu(this, palette)
        coordinator = HudLaunchCoordinator(
            activity = this,
            endpointProvider = { EdgeEndpoint.load(this)?.webSocketBase },
            onStatus = { if (currentTab == Tab.PROTECT) rebuild() },
        )
        setContentView(buildRoot())
        selectTab(currentTab)
    }

    override fun onResume() {
        super.onResume()
        if (settings.themeMode != themeOnCreate) { recreate(); return }
        coordinator.onResume()
        // The first resume follows onCreate, which has just built (and is animating) the tab.
        if (firstResume) firstResume = false else rebuild()
    }

    override fun onDestroy() {
        enrollment.cancel()
        pendingTest?.cancel()
        super.onDestroy()
    }

    // ------------------------------------------------------------------------------------------
    // Shell
    // ------------------------------------------------------------------------------------------

    private fun buildRoot(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = neu.backgroundDrawable()
        }
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(neu.dp(22), neu.dp(18), neu.dp(22), neu.dp(6))
            addView(LinearLayout(this@HudControlActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                addView(ImageView(this@HudControlActivity).apply {
                    setImageResource(R.drawable.ic_logo)
                    contentDescription = null
                    scaleType = ImageView.ScaleType.FIT_CENTER
                }, LinearLayout.LayoutParams(neu.dp(40), neu.dp(33)).apply { marginEnd = neu.dp(12) })
                addView(neu.title("VishyWashy", 26f))
            })
            addView(neu.body("Real-time scam-call protection").apply { setPadding(0, neu.dp(4), 0, 0) })
        })

        content = FrameLayout(this)
        root.addView(content, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        tabBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(neu.dp(14), neu.dp(8), neu.dp(14), neu.dp(8))
        }
        Tab.values().forEach { tab ->
            tabBar.addView(tabButton(tab), LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        root.addView(tabBar)

        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.updatePadding(top = bars.top, bottom = bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        return root
    }

    private fun tabButton(tab: Tab): View = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER
        setPadding(neu.dp(4), neu.dp(8), neu.dp(4), neu.dp(8))
        tag = tab
        addView(TextView(context).apply { text = tab.icon; textSize = 18f; gravity = Gravity.CENTER })
        addView(TextView(context).apply {
            text = tab.label; textSize = 11f; gravity = Gravity.CENTER
            setPadding(0, neu.dp(2), 0, 0)
        })
        isClickable = true
        neu.pressable(this, 0.9f)
        setOnClickListener { neu.haptic(it); selectTab(tab) }
    }

    private fun selectTab(tab: Tab) {
        val shown = content.getChildAt(content.childCount - 1) as? ScrollView
        if (tab == currentTab && shown != null) {
            shown.smoothScrollTo(0, 0)                  // tapping the open tab scrolls back to the top
            return
        }
        val direction = Integer.signum(tab.ordinal - currentTab.ordinal)
        currentTab = tab
        for (i in 0 until tabBar.childCount) {
            val view = tabBar.getChildAt(i) as LinearLayout
            val selected = view.tag == tab
            view.background = if (selected) neu.tabChip() else null
            val tint = if (selected) palette.accent else palette.textSecondary
            val icon = view.getChildAt(0) as TextView
            icon.setTextColor(tint)
            (view.getChildAt(1) as TextView).apply {
                setTextColor(tint)
                typeface = Typeface.create("sans-serif-medium", if (selected) Typeface.BOLD else Typeface.NORMAL)
            }
            if (selected && shown != null) neu.bounce(icon)
        }
        showTab(direction)
    }

    private fun buildScroll(): ScrollView {
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            clipToPadding = false
            overScrollMode = View.OVER_SCROLL_IF_CONTENT_SCROLLS
            setPadding(neu.dp(16), neu.dp(8), neu.dp(16), neu.dp(16))
        }
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        when (currentTab) {
            Tab.PROTECT -> buildProtect(column)
            Tab.HISTORY -> buildHistory(column)
            Tab.WHITELIST -> buildWhitelist(column)
            Tab.SETTINGS -> buildSettings(column)
        }
        scroll.addView(column, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        return scroll
    }

    /** Switches to [currentTab]: the old page slides out the way you're leaving while the new
     *  page's cards slide in from the other side, one after another. */
    private fun showTab(direction: Int) {
        if (!::content.isInitialized) return
        while (content.childCount > 1) content.removeViewAt(0)      // a previous transition still running
        val old = content.getChildAt(0)
        val scroll = buildScroll()
        content.addView(scroll)
        val shift = neu.dp(36).toFloat() * direction
        old?.animate()?.alpha(0f)?.translationX(-shift)?.setDuration(170)?.setInterpolator(Neu.EMPHASIZED)
            ?.withEndAction { content.removeView(old) }?.start()
        val column = scroll.getChildAt(0) as LinearLayout
        val base = if (old != null) 70L else 0L
        for (i in 0 until column.childCount) {
            val child = column.getChildAt(i)
            child.alpha = 0f
            child.translationX = shift
            child.translationY = neu.dp(if (direction == 0) 18 else 8).toFloat()
            child.animate().alpha(1f).translationX(0f).translationY(0f)
                .setStartDelay(base + i * 45L).setDuration(340)
                .setInterpolator(Neu.EMPHASIZED).start()
        }
    }

    /** Refreshes the open tab in place after a change: same scroll position, no re-entrance,
     *  just a soft fade so the update is noticeable without jumping. */
    private fun rebuild() {
        if (!::content.isInitialized) return
        val old = content.getChildAt(content.childCount - 1) as? ScrollView
        val y = old?.scrollY ?: 0
        val scroll = buildScroll()
        content.removeAllViews()
        content.addView(scroll)
        if (y > 0) scroll.doOnPreDraw { scroll.scrollTo(0, y) }
        scroll.alpha = 0.55f
        scroll.animate().alpha(1f).setDuration(200).setInterpolator(Neu.EMPHASIZED).start()
    }

    private fun openContactPicker() {
        val intent = Intent(Intent.ACTION_PICK, ContactsContract.CommonDataKinds.Phone.CONTENT_URI)
        try {
            pickContact.launch(intent)
        } catch (e: android.content.ActivityNotFoundException) {
            try {
                pickContact.launch(Intent(Intent.ACTION_PICK).apply { type = ContactsContract.CommonDataKinds.Phone.CONTENT_TYPE })
            } catch (e2: android.content.ActivityNotFoundException) {
                toast("No contacts app found on this phone.")
            }
        }
    }

    private fun readPickedContact(uri: Uri): Pair<String, String>? {
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
        )
        return contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (!cursor.moveToFirst()) return null
            val number = cursor.getString(0)?.trim().orEmpty()
            val name = cursor.getString(1)?.trim().orEmpty()
            if (number.isEmpty()) null else name to number
        }
    }

    private fun card(column: LinearLayout, highlight: Boolean = false, build: LinearLayout.() -> Unit): LinearLayout {
        val c = neu.card().apply(build)
        if (highlight) {
            c.background = neu.glass(24, 10,
                androidx.core.graphics.ColorUtils.setAlphaComponent(palette.accent, if (palette.dark) 0x2E else 0x24),
                androidx.core.graphics.ColorUtils.setAlphaComponent(palette.accent, 0x10),
                androidx.core.graphics.ColorUtils.setAlphaComponent(palette.accent, 0x66))
        }
        column.addView(c, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            .apply { topMargin = neu.dp(12) })
        return c
    }

    /** Slides a card off to the side, closes the gap it leaves, then runs [then]. */
    private fun dismissCard(card: View, then: () -> Unit) {
        card.isEnabled = false
        card.animate().alpha(0f).translationX(card.width * 0.35f).setDuration(200)
            .setInterpolator(Neu.EMPHASIZED).withEndAction {
                val start = card.height
                ValueAnimator.ofInt(start, 0).apply {
                    duration = 200
                    interpolator = Neu.EMPHASIZED
                    addUpdateListener {
                        card.layoutParams = card.layoutParams.apply { height = it.animatedValue as Int }
                    }
                    doOnEnd { then() }
                    start()
                }
            }.start()
    }

    private fun LinearLayout.gap(h: Int) = addView(neu.spacer(h))
    private fun LinearLayout.wrap(v: View, top: Int = 0) =
        addView(v, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = neu.dp(top) })

    // ------------------------------------------------------------------------------------------
    // Tab: Protect
    // ------------------------------------------------------------------------------------------

    private fun buildProtect(column: LinearLayout) {
        val armed = coordinator.isArmed
        card(column) {
            wrap(neu.row().apply {
                addView(TextView(context).apply { text = if (armed) "🛡" else "💤"; textSize = 26f })
                addView(LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(neu.title(if (armed) "Protected" else "Protection off", 18f,
                        if (armed) palette.success else palette.textPrimary))
                    addView(neu.body(coordinator.currentStatus()))
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = neu.dp(12) })
            })
            gap(14)
            wrap(neu.button(if (armed) "Disarm" else "Arm protection", filled = !armed,
                color = if (armed) palette.danger else palette.accent) {
                if (armed) coordinator.disarm() else coordinator.arm()
                rebuild()
            })
            wrap(neu.body(
                "VishyWashy stays idle until a call connects, then shows a safety badge over the " +
                    "dialer and analyses the call in real time.",
            ), 10)
            gap(14)
            wrap(toggleRow("Hold call on fraud",
                "At 70% risk, put the call on hold and ask: continue or hang up?",
                settings.holdOnFraud) { settings.holdOnFraud = it })
        }

        val steps = coordinator.setupSteps()
        if (steps.any { !it.done }) {
            card(column, highlight = steps.any { it.required && !it.done }) {
                val left = steps.count { !it.done }
                wrap(neu.row().apply {
                    addView(neu.title("Finish setup", 15f), LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                    addView(neu.pill("${steps.size - left}/${steps.size} done", if (steps.any { it.required && !it.done }) palette.warning else palette.success))
                })
                wrap(neu.body("Everything VishyWashy needs to protect your calls. Steps marked required " +
                    "must be done before protection can switch on."), 4)
                steps.forEach { step ->
                    wrap(permissionRow(step.title + if (step.required) " · required" else "", step.done, step.detail,
                        fixLabel = step.fixLabel) { step.fix() }, 12)
                }
            }
        }

        CallGuardService.lastReport?.takeIf { it.valid }?.let { handle ->
            card(column) {
                wrap(neu.title("Flagged call ready to report", 15f, palette.danger))
                wrap(neu.body("${handle.peer} — send the transcript and recording to the cyber cell " +
                    "(available briefly, then deleted)."), 4)
                wrap(neu.button("Report to cyber cell", filled = true, color = palette.danger) {
                    ReportActivity.launch(this@HudControlActivity, handle.sessionId, handle.token, handle.peer)
                }, 12)
            }
        }

        card(column) {
            val saved = settings.voiceprint != null
            wrap(neu.title("Your voice", 15f))
            wrap(neu.body(if (saved) "Saved on this phone — the transcript labels your words as “You”."
                          else "Record yourself once so the transcript can tell your words from the caller's."), 4)
            wrap(neu.button(if (saved) "Re-record in Settings" else "Record my voice", color = palette.accent) {
                selectTab(Tab.SETTINGS)
            }, 12)
        }
    }

    // ------------------------------------------------------------------------------------------
    // Tab: History
    // ------------------------------------------------------------------------------------------

    private fun buildHistory(column: LinearLayout) {
        val records = CallHistory(this).records()
        if (records.isEmpty()) {
            card(column) {
                wrap(neu.title("No calls yet", 15f))
                wrap(neu.body("Calls VishyWashy watches appear here with the time, who called and the risk."), 4)
            }
            return
        }
        card(column) {
            wrap(neu.row().apply {
                addView(neu.title("Watched calls", 15f), LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                addView(neu.body("${records.count { it.flagged }} flagged"))
            })
        }
        val report = CallGuardService.lastReport
        records.forEach { r ->
            card(column) {
                wrap(neu.row().apply {
                    addView(LinearLayout(context).apply {
                        orientation = LinearLayout.VERTICAL
                        addView(neu.title(displayPeer(r.peer), 15f))
                        addView(neu.body(DateUtils.getRelativeTimeSpanString(r.startedAtMs).toString() +
                            " · " + formatDuration(r.durationS)))
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                    addView(outcomePill(r))
                })
                wrap(riskBar(r.peakRisk), 12)
                if (report != null && report.sessionId == r.id && report.valid &&
                    r.outcome != CallHistory.Outcome.REPORTED) {
                    wrap(neu.button("Report to cyber cell", filled = true, color = palette.danger) {
                        ReportActivity.launch(this@HudControlActivity, report.sessionId, report.token, report.peer)
                    }, 12)
                }
            }
        }
        card(column) {
            wrap(neu.button("Clear history", color = palette.textSecondary) {
                MaterialAlertDialogBuilder(this@HudControlActivity)
                    .setTitle("Clear call history?")
                    .setMessage("This removes all ${records.size} watched calls from this phone. It can't be undone.")
                    .setNegativeButton("Cancel", null)
                    .setPositiveButton("Clear") { _, _ -> CallHistory(this@HudControlActivity).clear(); rebuild() }
                    .show()
            })
        }
    }

    /** Older versions sometimes saved the app's own badge text as the caller's name. */
    private fun displayPeer(peer: String): String =
        if (peer.isBlank() || OWN_NAMES.any { peer.contains(it, ignoreCase = true) }) "Unknown caller" else peer

    /** Peak risk of a call as a thin bar that fills in when shown (green → amber → red). */
    private fun riskBar(risk: Double): View {
        val level = risk.coerceIn(0.0, 1.0).toFloat()
        val color = when {
            level >= 0.70f -> palette.danger
            level >= 0.35f -> palette.warning
            else -> palette.success
        }
        fun rounded(fill: Int) = GradientDrawable().apply { cornerRadius = neu.dp(3).toFloat(); setColor(fill) }
        val fill = View(this).apply {
            background = rounded(color)
            pivotX = 0f
            scaleX = 0f
        }
        return FrameLayout(this).apply {
            background = rounded(palette.insetFill)
            contentDescription = "Peak risk ${(level * 100).toInt()} percent"
            addView(fill, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, neu.dp(6)))
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, neu.dp(6))
            fill.animate().scaleX(level.coerceAtLeast(0.02f)).setStartDelay(250).setDuration(700)
                .setInterpolator(Neu.EMPHASIZED).start()
        }
    }

    private fun outcomePill(r: CallHistory.Record): View = when (r.outcome) {
        CallHistory.Outcome.REPORTED -> neu.pill("Reported", palette.accent)
        CallHistory.Outcome.HUNG_UP -> neu.pill("Hung up", palette.danger)
        CallHistory.Outcome.FLAGGED -> neu.pill("High risk ${(r.peakRisk * 100).toInt()}%", palette.danger)
        CallHistory.Outcome.MONITORED -> neu.pill("Safe", palette.success)
    }

    // ------------------------------------------------------------------------------------------
    // Tab: Whitelist / Trusted
    // ------------------------------------------------------------------------------------------

    private fun buildWhitelist(column: LinearLayout) {
        val whitelist = Whitelist(this)
        card(column) {
            wrap(neu.title("Trusted numbers", 15f))
            wrap(neu.body("Calls from these numbers (or saved names) are never recorded or analysed — " +
                "VishyWashy stays off for them."), 4)
            val input = inputField("Number or contact name")
            wrap(input, 12)
            wrap(neu.row().apply {
                addView(neu.button("Add typed", color = palette.accent, compact = true) {
                    val text = input.text.toString().trim()
                    if (text.isEmpty()) return@button
                    if (whitelist.add(text)) { input.setText(""); rebuild() }
                    else toast("Already trusted, or not a valid entry.")
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = neu.dp(8) })
                addView(neu.button("From contacts", color = palette.textPrimary, compact = true) {
                    openContactPicker()
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            }, 10)
        }
        val entries = whitelist.entries()
        if (entries.isEmpty()) {
            card(column) { wrap(neu.body("No trusted numbers yet.")) }
            return
        }
        entries.forEach { entry ->
            card(column) {
                val entryCard = this
                wrap(neu.row().apply {
                    addView(LinearLayout(context).apply {
                        orientation = LinearLayout.VERTICAL
                        addView(neu.title(entry.label, 15f))
                        addView(neu.body(if (entry.isNumber) "Number" else "Name match"))
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                    addView(neu.button("Remove", color = palette.danger) {
                        dismissCard(entryCard) {
                            whitelist.remove(entry)
                            rebuild()
                            Snackbar.make(content, "Removed ${entry.label}", Snackbar.LENGTH_LONG)
                                .setDuration(UNDO_MS)
                                .setAnchorView(tabBar)
                                .setAction("Undo") { neu.haptic(it); whitelist.restore(entry); rebuild() }
                                .setTextColor(palette.textPrimary)
                                .setActionTextColor(palette.accent)
                                .setBackgroundTint(neu.surfaceColor)
                                .show()
                        }
                    })
                })
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Tab: Settings
    // ------------------------------------------------------------------------------------------

    private fun buildSettings(column: LinearLayout) {
        buildServerCard(column)
        buildVoiceCard(column)
        buildCallControlsCard(column)
        buildAppearanceCard(column)
        buildReportCard(column)
        buildPermissionsCard(column)
    }

    private fun buildServerCard(column: LinearLayout) {
        card(column) {
            wrap(neu.title("Edge server", 15f))
            wrap(neu.body("Run python server.py and enter the address it prints. Phone and laptop must " +
                "be on the same Wi-Fi."), 4)
            val input = inputField("192.168.1.20:8000", InputType.TYPE_TEXT_VARIATION_URI)
            input.setText(EdgeEndpoint.load(this@HudControlActivity)?.display.orEmpty())
            wrap(input, 12)
            val result = neu.body("")
            wrap(neu.button("Test connection", color = palette.accent) {
                testConnection(input.text.toString(), result)
            }, 10)
            wrap(result, 8)
        }
    }

    private fun buildVoiceCard(column: LinearLayout) {
        card(column) {
            wrap(neu.title("Your voice", 15f))
            wrap(neu.body("Read this aloud once so the transcript can label your words as “You”:"), 4)
            wrap(neu.body("“${VoiceEnrollment.SCRIPT}”", color = palette.textPrimary).apply {
                setTypeface(typeface, Typeface.ITALIC)
            }, 10)
            val status = neu.body(
                if (settings.voiceprint != null) "Voiceprint saved on this phone." else "Not recorded yet.",
                color = if (settings.voiceprint != null) palette.success else palette.textSecondary,
            )
            wrap(status, 10)
            wrap(neu.row().apply {
                addView(neu.button(if (settings.voiceprint != null) "Re-record" else "Record (10 s)", color = palette.accent, compact = true) {
                    recordVoice(status)
                })
                if (settings.voiceprint != null) {
                    addView(neu.button("Delete", color = palette.danger, compact = true) {
                        settings.voiceprint = null; rebuild()
                    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                        .apply { marginStart = neu.dp(10) })
                }
            }, 12)
        }
    }

    private fun buildCallControlsCard(column: LinearLayout) {
        card(column) {
            wrap(neu.title("During a call", 15f))
            gap(10)
            wrap(toggleRow("Hold call on fraud",
                "When a scam is detected, VishyWashy puts the call on hold (or mutes you where there is " +
                    "no Hold button, e.g. WhatsApp) so the caller can't hear you, and asks whether to " +
                    "continue talking or hang up.",
                settings.holdOnFraud) { settings.holdOnFraud = it })
            gap(14)
            wrap(toggleRow("Hear both sides (auto loudspeaker)",
                "Switches the call to loudspeaker so the microphone hears the caller too. Without it, " +
                    "most phones only let VishyWashy hear your voice, and the scammer's words are missed.",
                settings.autoSpeaker) { settings.autoSpeaker = it })
        }
    }

    private fun buildAppearanceCard(column: LinearLayout) {
        card(column) {
            wrap(neu.title("Appearance", 15f))
            gap(10)
            val modes = listOf(
                ProtectionSettings.THEME_SYSTEM to "System",
                ProtectionSettings.THEME_LIGHT to "Light",
                ProtectionSettings.THEME_DARK to "Dark",
            )
            wrap(neu.row().apply {
                modes.forEach { (mode, label) ->
                    addView(neu.button(label, filled = settings.themeMode == mode, compact = true) {
                        if (settings.themeMode != mode) { settings.themeMode = mode; recreate() }
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                        .apply { marginEnd = if (mode != ProtectionSettings.THEME_DARK) neu.dp(8) else 0 })
                }
            })
        }
        buildSeniorCard(column)
    }

    private fun buildSeniorCard(column: LinearLayout) {
        val on = settings.seniorMode
        card(column, highlight = on) {
            wrap(neu.row().apply {
                addView(TextView(context).apply { text = "👵"; textSize = 22f })
                addView(LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(neu.title("Senior mode", 16f, if (on) palette.accent else palette.textPrimary))
                    addView(neu.body(if (on) "ON" else "OFF", color = if (on) palette.accent else palette.textSecondary))
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = neu.dp(10) })
                addView(seniorKnob(on))
            })
            wrap(neu.body("For elders: when a call's risk reaches HIGH (70%), VishyWashy shows a warning " +
                "and then automatically ends the call for you and opens the report to the cyber cell. " +
                "The alert badge also opens straight to a large one-tap “Hang up & report” button."), 10)
        }
    }

    private fun seniorKnob(value: Boolean): TextView {
        lateinit var knob: TextView
        var state = value
        fun render() {
            knob.text = if (state) "ON" else "OFF"
            knob.setTextColor(if (state) palette.onAccent else palette.textSecondary)
            knob.background = neu.knobBackground(state)
            val sp = neu.dp(7)
            knob.setPadding(sp + neu.dp(12), sp + neu.dp(5), sp + neu.dp(12), sp + neu.dp(5))
        }
        knob = TextView(this).apply {
            textSize = 12f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            gravity = Gravity.CENTER
            isClickable = true
            neu.pressable(this, 0.9f)
            setOnClickListener {
                state = !state
                settings.seniorMode = state
                neu.haptic(it, strong = true)
                render()
                neu.bounce(it)
                it.postDelayed({ rebuild() }, 260)        // let the switch finish its bounce first
            }
        }
        render()
        return knob
    }

    private fun buildReportCard(column: LinearLayout) {
        card(column) {
            wrap(neu.title("Report to cyber cell", 15f))
            wrap(neu.body("When you report a flagged call, VishyWashy opens the share sheet addressed to " +
                "this number. You choose the app and send it yourself. Enter your local cyber-cell " +
                "number, e.g. +91 1930."), 4)
            val input = inputField("+91 1930", InputType.TYPE_CLASS_PHONE)
            input.setText(settings.reportNumber)
            wrap(input, 12)
            val status = neu.body("")
            wrap(neu.button("Save number", color = palette.accent) {
                val value = input.text.toString().trim()
                val digits = value.filter { it.isDigit() || it == '+' }
                if (digits.count { it.isDigit() } < 3) {
                    status.text = "Enter a valid phone number."
                    status.setTextColor(palette.danger)
                } else {
                    settings.reportNumber = value
                    status.text = "Saved. Reports will be addressed to $value."
                    status.setTextColor(palette.success)
                    neu.haptic(input, strong = true)
                }
            }, 10)
            wrap(status, 8)
        }
    }

    private fun buildPermissionsCard(column: LinearLayout) {
        card(column) {
            wrap(neu.title("Permissions", 15f))
            gap(8)
            wrap(permissionRow("Accessibility service", CallGuardService.isEnabled(this@HudControlActivity),
                "Lets VishyWashy hear and watch during a call.") {
                startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            })
            wrap(permissionRow("Microphone",
                ContextCompat.checkSelfPermission(this@HudControlActivity, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED, "Needed to analyse call audio.") {
                startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
            }, 8)
            gap(14)
            wrap(neu.button("Withdraw consent", color = palette.danger) {
                coordinator.withdrawConsent(); rebuild()
            })
        }
    }

    // ------------------------------------------------------------------------------------------
    // Pieces
    // ------------------------------------------------------------------------------------------

    private fun toggleRow(title: String, body: String, value: Boolean, onChange: (Boolean) -> Unit): View {
        lateinit var knob: TextView
        var state = value
        fun render() {
            knob.text = if (state) "ON" else "OFF"
            knob.setTextColor(if (state) palette.onAccent else palette.textSecondary)
            knob.background = neu.knobBackground(state)
            val sp = neu.dp(7)
            knob.setPadding(sp + neu.dp(10), sp + neu.dp(5), sp + neu.dp(10), sp + neu.dp(5))
        }
        return neu.row().apply {
            addView(LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                addView(neu.title(title, 15f))
                addView(neu.body(body))
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = neu.dp(10) })
            knob = TextView(context).apply {
                textSize = 12f
                typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
                gravity = Gravity.CENTER
                isClickable = true
                neu.pressable(this, 0.9f)
                setOnClickListener {
                    state = !state
                    onChange(state)
                    neu.haptic(it, strong = true)
                    render()
                    neu.bounce(it)
                }
            }
            render()
            addView(knob)
        }
    }

    private fun permissionRow(title: String, ok: Boolean, body: String, fixLabel: String = "Fix", onFix: () -> Unit): View =
        neu.row().apply {
            addView(TextView(context).apply {
                text = if (ok) "✓" else "⚠"
                textSize = 18f
                setTextColor(if (ok) palette.success else palette.warning)
            }, LinearLayout.LayoutParams(neu.dp(26), ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                addView(neu.title(title, 14f))
                addView(neu.body(body))
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            if (!ok) addView(neu.button(fixLabel, color = palette.accent, compact = true) { onFix() })
        }

    private fun inputField(hint: String, type: Int = InputType.TYPE_CLASS_TEXT): EditText =
        EditText(this).apply {
            this.hint = hint
            inputType = type or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            isSingleLine = true
            imeOptions = EditorInfo.IME_ACTION_DONE
            setTextColor(palette.textPrimary)
            setHintTextColor(palette.textSecondary)
            background = neu.inputBackground()
            val sp = neu.dp(8)
            setPadding(sp + neu.dp(12), sp + neu.dp(10), sp + neu.dp(12), sp + neu.dp(10))
        }

    // ------------------------------------------------------------------------------------------
    // Actions
    // ------------------------------------------------------------------------------------------

    private fun testConnection(raw: String, result: TextView) {
        val endpoint = EdgeEndpoint.parse(raw)
        if (endpoint == null) {
            result.text = "Enter an address like 192.168.1.20:8000"
            result.setTextColor(palette.danger)
            return
        }
        EdgeEndpoint.save(this, endpoint)
        pendingTest?.cancel()
        result.text = "Contacting ${endpoint.display}…"
        result.setTextColor(palette.textSecondary)
        val started = SystemClock.elapsedRealtime()
        val call = httpClient.newCall(Request.Builder().url(endpoint.healthUrl).build())
        pendingTest = call
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (call.isCanceled()) return
                runOnUiThread {
                    result.text = "Can't reach ${endpoint.display}. Check Wi-Fi, that the server is " +
                        "running, and the laptop firewall allows port ${EdgeEndpoint.DEFAULT_PORT}."
                    result.setTextColor(palette.danger)
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val ms = SystemClock.elapsedRealtime() - started
                val ok = response.use { it.isSuccessful && runCatching { JSONObject(it.body?.string().orEmpty()) }.isSuccess }
                runOnUiThread {
                    result.text = if (ok) "Connected · $ms ms" else "Reached it, but it isn't a VishyWashy server."
                    result.setTextColor(if (ok) palette.success else palette.danger)
                }
            }
        })
    }

    private fun recordVoice(status: TextView) {
        if (enrollment.isRunning) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            micPermission.launch(Manifest.permission.RECORD_AUDIO)
            status.text = "Allow the microphone, then tap Record again."
            status.setTextColor(palette.warning)
            return
        }
        val endpoint = EdgeEndpoint.load(this) ?: run {
            status.text = "Set the server address first."
            status.setTextColor(palette.danger); return
        }
        status.text = "Recording… read the line aloud."
        status.setTextColor(palette.textSecondary)
        @Suppress("MissingPermission")
        enrollment.start(endpoint, settings, object : VoiceEnrollment.Listener {
            override fun onProgress(fraction: Float, uploading: Boolean) {
                status.text = if (uploading) "Creating your voiceprint…" else "Recording… ${(fraction * 100).toInt()}%"
            }
            override fun onSaved(speechSeconds: Double) {
                status.text = "Saved from ${"%.1f".format(speechSeconds)} s of speech."
                status.setTextColor(palette.success)
            }
            override fun onFailed(message: String) {
                status.text = message
                status.setTextColor(palette.danger)
            }
        })
    }

    private fun applyNightMode(mode: String) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return
        val uiMode = getSystemService(UiModeManager::class.java) ?: return
        uiMode.setApplicationNightMode(
            when (mode) {
                ProtectionSettings.THEME_LIGHT -> UiModeManager.MODE_NIGHT_NO
                ProtectionSettings.THEME_DARK -> UiModeManager.MODE_NIGHT_YES
                else -> UiModeManager.MODE_NIGHT_AUTO
            },
        )
    }

    private fun formatDuration(seconds: Int): String =
        if (seconds >= 60) "${seconds / 60}m ${seconds % 60}s" else "${seconds}s"

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
}
