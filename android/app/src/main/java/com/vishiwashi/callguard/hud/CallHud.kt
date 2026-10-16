package com.vishiwashi.callguard.hud

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.annotation.SuppressLint
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.os.Build
import android.text.TextUtils
import android.util.Log
import android.view.ContextThemeWrapper
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowManager
import android.view.animation.DecelerateInterpolator
import android.view.animation.PathInterpolator
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.graphics.ColorUtils
import androidx.core.graphics.PathParser
import androidx.core.view.doOnPreDraw
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.color.DynamicColors
import com.google.android.material.color.MaterialColors
import com.google.android.material.progressindicator.LinearProgressIndicator
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import com.google.android.material.R as MaterialR

/** What the HUD shows. */
internal sealed interface HudState {
    data class Unavailable(val message: String) : HudState

    /** [callerAudible]: the other party's voice is being captured (call audio or speakerphone). */
    data class Normal(val latencyMs: Int, val callerAudible: Boolean) : HudState
    data class Caution(val score: Double, val callerAudible: Boolean) : HudState
    data class Alert(
        val score: Double,
        val signals: List<String>,
        val guidance: String,
    ) : HudState

    /**
     * Fraud detected and the call paused for the user to decide. [paused] says how: the dialer's
     * Hold button, its Mute button (the caller can't hear you), or neither could be pressed.
     */
    data class OnHold(
        val score: Double,
        val signals: List<String>,
        val paused: Pause,
    ) : HudState {
        enum class Pause { HELD, MUTED, NOT_PAUSED }
    }
}

/**
 * Draggable Material 3 caller-security HUD.
 *
 *  - Normal  → compact pill: green shield + live end-to-end latency
 *  - Caution → compact pill: amber shield + risk percentage
 *  - Alert   → expands into a card: red gradient header, risk meter, signal chips,
 *              Hang up / Report 1930
 *
 * Surfaces and text follow the device's Material You palette; status colours are fixed so their
 * meaning never depends on the wallpaper. [windowType] is `TYPE_ACCESSIBILITY_OVERLAY` when hosted
 * by the accessibility service (no overlay permission needed, drawn above the dialer).
 */
internal class CallHud(
    host: Context,
    private val windowManager: WindowManager,
    windowType: Int,
    private val seniorMode: Boolean = false,
    private val actions: Actions,
) {
    interface Actions {
        fun onHangUp()
        fun onReport()

        /** Senior mode: end the call and start a report in one tap. */
        fun onHangUpAndReport()

        /** The user asked to capture the caller too (pill tapped while only their voice is heard). */
        fun onEnableSpeaker()

        /** The call was put on hold for a fraud alert and the user chose to keep talking. */
        fun onContinueCall()
    }

    private val ctx: Context = DynamicColors.wrapContextIfAvailable(
        ContextThemeWrapper(host, MaterialR.style.Theme_Material3_DynamicColors_DayNight),
    )
    private val density = host.resources.displayMetrics.density
    private val prefs = host.getSharedPreferences(HUD_PREFS, Context.MODE_PRIVATE)
    private val touchSlop = ViewConfiguration.get(ctx).scaledTouchSlop

    // -- Theme roles ---------------------------------------------------------------------------
    private val surfaceHigh = themeColor(MaterialR.attr.colorSurfaceContainerHigh, 0xFFECE6F0.toInt())
    private val surfaceLow = themeColor(MaterialR.attr.colorSurfaceContainerLow, 0xFFF7F2FA.toInt())
    private val onSurface = themeColor(MaterialR.attr.colorOnSurface, 0xFF1D1B20.toInt())
    private val onSurfaceVariant = themeColor(MaterialR.attr.colorOnSurfaceVariant, 0xFF49454F.toInt())
    private val outline = themeColor(MaterialR.attr.colorOutline, 0xFF79747E.toInt())
    private val outlineVariant = themeColor(MaterialR.attr.colorOutlineVariant, 0xFFCAC4D0.toInt())
    private val errorContainer = themeColor(MaterialR.attr.colorErrorContainer, 0xFFF9DEDC.toInt())
    private val onErrorContainer = themeColor(MaterialR.attr.colorOnErrorContainer, 0xFF410E0B.toInt())

    // -- Views ---------------------------------------------------------------------------------
    private val root = FrameLayout(ctx)

    private val pill = MaterialCardView(ctx)
    private val pillBadgeBg = GradientDrawable().apply { shape = GradientDrawable.OVAL }
    private val pillShield = ShieldDrawable()
    private val pillTitle = TextView(ctx)
    private val pillSubtitle = TextView(ctx)
    private val pillDot = View(ctx)
    private val pillDotBg = GradientDrawable().apply { shape = GradientDrawable.OVAL }

    private val card = MaterialCardView(ctx)
    private val cardTitle = TextView(ctx)
    private val cardSubtitle = TextView(ctx)
    private lateinit var cardHangUp: MaterialButton
    private lateinit var cardSecondary: MaterialButton
    private var cardOnHold = false
    private val cardPercent = TextView(ctx)
    private val cardMeter = LinearProgressIndicator(ctx)
    private val cardChips = ChipGroup(ctx)
    private val cardGuidance = TextView(ctx)

    private val params = WindowManager.LayoutParams(
        WindowManager.LayoutParams.WRAP_CONTENT,
        WindowManager.LayoutParams.WRAP_CONTENT,
        windowType,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
        PixelFormat.TRANSLUCENT,
    ).apply {
        gravity = Gravity.TOP or Gravity.START
        title = "VishyWashy HUD"
    }

    // -- State ---------------------------------------------------------------------------------
    private var anchoredRight = prefs.getBoolean(KEY_ANCHOR_RIGHT, true)
    private var alertMinimized = false
    private var cardShown = false
    private var lastState: HudState? = null
    private var lastChipLabels: List<String> = emptyList()
    private var pillAccent = COLOR_GREY
    private var pillGlyph = ShieldDrawable.Glyph.NONE

    private var snapAnimator: ValueAnimator? = null
    private var accentAnimator: ValueAnimator? = null
    private var pulseAnimator: ObjectAnimator? = null

    init {
        buildPill()
        buildCard()
        // Room for the elevation shadow, which a WRAP_CONTENT window would otherwise clip.
        root.setPadding(dp(14), dp(10), dp(14), dp(18))
        root.clipToPadding = false
        root.clipChildren = false
        root.addView(pill, FrameLayout.LayoutParams(WRAP, WRAP))
        root.addView(card, FrameLayout.LayoutParams(cardWidth(), WRAP))
        card.visibility = View.GONE
        applyAccent(COLOR_GREY)

        root.setOnTouchListener(DragHandler())
        root.setOnClickListener { onHudTapped() }
        root.addOnLayoutChangeListener { _, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
            val resized = right - left != oldRight - oldLeft || bottom - top != oldBottom - oldTop
            if (resized) keepAnchored()
        }

        val (screenW, _) = usableScreenSize()
        params.x = if (anchoredRight) screenW else 0 // corrected after the first layout pass
        params.y = prefs.getInt(KEY_Y, dp(140))
    }

    val isAttached: Boolean get() = root.isAttachedToWindow

    fun attach(): Boolean = try {
        windowManager.addView(root, params)
        root.alpha = 0f
        root.animate().alpha(1f).setDuration(220).setInterpolator(EMPHASIZED_DECELERATE).start()
        true
    } catch (e: RuntimeException) {
        Log.e(TAG, "Unable to add overlay", e)
        false
    }

    fun detach() {
        snapAnimator?.cancel()
        accentAnimator?.cancel()
        pulseAnimator?.cancel()
        card.animate().cancel()
        pill.animate().cancel()
        root.animate().cancel()
        if (root.isAttachedToWindow) {
            try {
                windowManager.removeView(root)
            } catch (e: IllegalArgumentException) {
                Log.w(TAG, "Overlay already removed", e)
            }
        }
    }

    fun render(state: HudState) {
        val previous = lastState
        lastState = state
        when (state) {
            is HudState.Unavailable -> showPill(
                accent = COLOR_GREY,
                glyph = ShieldDrawable.Glyph.NONE,
                title = "VishyWashy",
                subtitle = state.message,
                live = false,
                description = "VishyWashy: ${state.message}",
            )

            is HudState.Normal -> if (state.callerAudible) {
                showPill(
                    accent = COLOR_GREEN,
                    glyph = ShieldDrawable.Glyph.CHECK,
                    title = "Call protected",
                    subtitle = "${state.latencyMs} ms · both voices",
                    live = true,
                    description = "VishyWashy: no risk detected. Latency ${state.latencyMs} milliseconds.",
                )
            } else {
                // Without the loudspeaker the microphone can't hear the caller on a SIM call.
                showPill(
                    accent = COLOR_AMBER,
                    glyph = ShieldDrawable.Glyph.EXCLAIM,
                    title = "Caller not heard",
                    subtitle = "Turn on Speaker · tap here",
                    live = true,
                    description = "VishyWashy can only hear your voice. Turn on the loudspeaker so it can " +
                        "hear the caller. Tap to switch it on.",
                )
            }

            is HudState.Caution -> if (seniorMode) {
                showCard(HudState.Alert(state.score, emptyList(),
                    "This caller is not verified. Do not share any OTP, PIN or password."))
            } else {
                showPill(
                    accent = COLOR_AMBER,
                    glyph = ShieldDrawable.Glyph.EXCLAIM,
                    title = "Unverified caller",
                    subtitle = "Risk ${percent(state.score)} · don't share codes",
                    live = true,
                    description = "VishyWashy caution: caller intent unverified. Risk ${percent(state.score)}.",
                )
            }

            is HudState.OnHold -> {
                if (previous !is HudState.OnHold) {
                    alertMinimized = false
                    root.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                    root.announceForAccessibility("$ALERT_TITLE. The call is on hold. Continue or hang up?")
                }
                if (alertMinimized) {
                    showPill(
                        accent = COLOR_RED,
                        glyph = ShieldDrawable.Glyph.EXCLAIM,
                        title = "Call on hold",
                        subtitle = "Possible scam · tap to decide",
                        live = true,
                        description = "$ALERT_TITLE. The call is on hold. Tap to continue or hang up.",
                    )
                } else {
                    showCard(
                        HudState.Alert(state.score, state.signals, holdMessage(state.paused)),
                        onHold = true,
                    )
                }
            }

            is HudState.Alert -> {
                if (previous is HudState.OnHold) {
                    alertMinimized = true          // the user chose to continue; keep the warning small
                } else if (previous !is HudState.Alert) {
                    alertMinimized = false
                    root.performHapticFeedback(
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                            HapticFeedbackConstants.REJECT
                        } else {
                            HapticFeedbackConstants.LONG_PRESS
                        },
                    )
                    root.announceForAccessibility(ALERT_TITLE)
                }
                if (alertMinimized) {
                    showPill(
                        accent = COLOR_RED,
                        glyph = ShieldDrawable.Glyph.EXCLAIM,
                        title = "High-risk call",
                        subtitle = "${percent(state.score)} · tap to open",
                        live = true,
                        description = "$ALERT_TITLE. Tap to open actions.",
                    )
                } else {
                    showCard(state)
                }
            }
        }
        if (state is HudState.Caution && previous !is HudState.Caution && previous !is HudState.Alert) {
            root.announceForAccessibility("VishyWashy caution: caller intent unverified.")
        }
    }

    private fun onHudTapped() {
        when (val state = lastState) {
            is HudState.Alert, is HudState.OnHold -> if (alertMinimized) {
                alertMinimized = false
                render(state)
            }
            is HudState.Normal -> if (!state.callerAudible) actions.onEnableSpeaker()
            is HudState.Caution -> if (!state.callerAudible) actions.onEnableSpeaker()
            else -> Unit
        }
    }

    // -- Pill ----------------------------------------------------------------------------------

    private fun showPill(
        accent: Int,
        glyph: ShieldDrawable.Glyph,
        title: String,
        subtitle: String,
        live: Boolean,
        description: String,
    ) {
        if (cardShown) {
            cardShown = false
            card.animate().cancel()
            card.animate()
                .alpha(0f).scaleX(0.94f).scaleY(0.94f)
                .setDuration(150)
                .setInterpolator(EMPHASIZED_ACCELERATE)
                .withEndAction {
                    if (!cardShown) {
                        card.visibility = View.GONE
                        revealPill()
                    }
                }
                .start()
        } else if (pill.visibility != View.VISIBLE) {
            revealPill()
        }

        pillGlyph = glyph
        animateAccent(accent)
        pillTitle.text = title
        pillSubtitle.text = subtitle
        pill.contentDescription = description
        setPulsing(live)
    }

    private fun revealPill() {
        pill.visibility = View.VISIBLE
        pill.alpha = 0f
        pill.scaleX = 0.96f
        pill.scaleY = 0.96f
        pill.animate().alpha(1f).scaleX(1f).scaleY(1f)
            .setDuration(200)
            .setInterpolator(EMPHASIZED_DECELERATE)
            .start()
    }

    private fun animateAccent(target: Int) {
        if (target == pillAccent && accentAnimator?.isRunning != true) {
            applyAccent(target)
            return
        }
        accentAnimator?.cancel()
        accentAnimator = ValueAnimator.ofArgb(pillAccent, target).apply {
            duration = 260
            addUpdateListener { applyAccent(it.animatedValue as Int) }
            start()
        }
    }

    private fun applyAccent(color: Int) {
        pillAccent = color
        pillBadgeBg.setColor(ColorUtils.setAlphaComponent(color, 0x3D))
        pillShield.update(color, pillGlyph, surfaceHigh)
        pillDotBg.setColor(color)
    }

    private fun setPulsing(live: Boolean) {
        if (live) {
            if (pulseAnimator?.isRunning == true) return
            pulseAnimator = ObjectAnimator.ofFloat(pillDot, View.ALPHA, 1f, 0.25f).apply {
                duration = 900
                repeatMode = ValueAnimator.REVERSE
                repeatCount = ValueAnimator.INFINITE
                start()
            }
        } else {
            pulseAnimator?.cancel()
            pulseAnimator = null
            pillDot.alpha = 0.6f
        }
    }

    private fun buildPill() {
        pill.apply {
            radius = dp(28).toFloat()
            cardElevation = dp(10).toFloat()
            setCardBackgroundColor(surfaceHigh)
            strokeWidth = dp(1)
            strokeColor = ColorUtils.setAlphaComponent(outlineVariant, 0x80)
            minimumHeight = dp(52)
            isClickable = false
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
        }

        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(6), dp(6), dp(16), dp(6))
        }

        val badge = FrameLayout(ctx).apply { background = pillBadgeBg }
        val shield = ImageView(ctx).apply {
            setImageDrawable(pillShield)
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        badge.addView(shield, FrameLayout.LayoutParams(dp(22), dp(22), Gravity.CENTER))
        row.addView(badge, LinearLayout.LayoutParams(dp(40), dp(40)).apply { marginEnd = dp(10) })

        val texts = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL }
        pillTitle.apply {
            textSize = 14f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(onSurface)
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        pillSubtitle.apply {
            textSize = 12f
            setTextColor(onSurfaceVariant)
            fontFeatureSettings = "tnum"
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        texts.addView(pillTitle, LinearLayout.LayoutParams(WRAP, WRAP))
        texts.addView(pillSubtitle, LinearLayout.LayoutParams(WRAP, WRAP))
        row.addView(texts, LinearLayout.LayoutParams(WRAP, WRAP))

        pillDot.background = pillDotBg
        pillDot.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        row.addView(pillDot, LinearLayout.LayoutParams(dp(8), dp(8)).apply { marginStart = dp(12) })

        pill.addView(row, FrameLayout.LayoutParams(WRAP, WRAP))
    }

    // -- Alert card ----------------------------------------------------------------------------

    private fun holdMessage(paused: HudState.OnHold.Pause): String = when (paused) {
        HudState.OnHold.Pause.HELD -> "VishyWashy put this call on hold: the caller can't hear you. " +
            "Never share an OTP, PIN or password. Continue only if you trust this caller."
        HudState.OnHold.Pause.MUTED -> "VishyWashy muted your microphone: the caller can't hear you. " +
            "Never share an OTP, PIN or password. Continue only if you trust this caller."
        HudState.OnHold.Pause.NOT_PAUSED -> "VishyWashy couldn't put this call on hold. Don't share any " +
            "OTP, PIN or password. Hang up unless you trust this caller."
    }

    private fun applyCardMode(onHold: Boolean) {
        if (onHold == cardOnHold) return
        cardOnHold = onHold
        cardTitle.text = if (onHold) "Call on hold" else "High-risk call"
        cardSubtitle.text = if (onHold) "Possible scam — you decide" else "Possible scam call"
        cardHangUp.text = if (onHold || !seniorMode) "Hang up" else "Hang up & report"
        cardSecondary.text = if (onHold) "Continue call" else "Report $CYBER_HELPLINE"
        cardSecondary.icon = if (onHold) PathIconDrawable(ICON_PHONE, onSurface) else PathIconDrawable(ICON_PHONE, COLOR_RED)
        cardSecondary.contentDescription = if (onHold) "Take the call off hold and keep talking" else
            "Call $CYBER_HELPLINE, the National Cyber Crime Helpline"
    }

    private fun showCard(state: HudState.Alert, onHold: Boolean = false) {
        applyCardMode(onHold)
        val pct = (state.score * 100).roundToInt().coerceIn(0, 100)
        cardPercent.text = "$pct%"
        cardMeter.setProgressCompat(pct, cardShown)
        cardTitle.contentDescription = "$ALERT_TITLE. Risk $pct percent."
        bindChips(state)
        cardGuidance.text = state.guidance.ifBlank {
            "Do not share OTPs, PINs, card details or remote-access codes. " +
                "Banks and police never ask for them on a call."
        }

        if (cardShown) return
        cardShown = true
        setPulsing(false)
        pill.animate().cancel()
        pill.visibility = View.GONE

        card.layoutParams = (card.layoutParams as FrameLayout.LayoutParams).apply { width = cardWidth() }
        card.animate().cancel()
        card.visibility = View.VISIBLE
        card.alpha = 0f
        card.scaleX = 0.88f
        card.scaleY = 0.88f
        card.doOnPreDraw { view ->
            view.pivotX = if (anchoredRight) view.width.toFloat() else 0f
            view.pivotY = 0f
            view.animate()
                .alpha(1f).scaleX(1f).scaleY(1f)
                .setDuration(340)
                .setInterpolator(EMPHASIZED_DECELERATE)
                .start()
        }
    }

    private fun bindChips(state: HudState.Alert) {
        val labels = buildList {
            state.signals.mapTo(this) { signalLabel(it) }
            if (isEmpty()) add("Unusual call pattern")
        }.distinct().take(MAX_CHIPS)

        if (labels == lastChipLabels) return
        lastChipLabels = labels
        cardChips.removeAllViews()
        labels.forEach { cardChips.addView(signalChip(it)) }
    }

    private fun signalChip(label: String): Chip = Chip(ctx).apply {
        text = label
        isClickable = false
        isCheckable = false
        isFocusable = false
        setEnsureMinTouchTargetSize(false)
        chipMinHeight = dp(30).toFloat()
        chipStrokeWidth = 0f
        chipBackgroundColor = ColorStateList.valueOf(errorContainer)
        setTextColor(onErrorContainer)
        textSize = 12.5f
        chipStartPadding = dp(4).toFloat()
        chipEndPadding = dp(4).toFloat()
    }

    private fun buildCard() {
        card.apply {
            radius = dp(28).toFloat()
            cardElevation = dp(18).toFloat()
            setCardBackgroundColor(surfaceLow)
            strokeWidth = 0
            clipToOutline = true
            isClickable = false
        }
        val column = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL }

        // ---- Gradient header ----------------------------------------------------------------
        val header = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(18), dp(12), dp(18))
            background = GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                intArrayOf(COLOR_RED_BRIGHT, COLOR_RED, COLOR_RED_DEEP),
            ).apply {
                val r = dp(28).toFloat()
                cornerRadii = floatArrayOf(r, r, r, r, 0f, 0f, 0f, 0f)
            }
        }

        val titleRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val badge = FrameLayout(ctx).apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.WHITE)
            }
            elevation = dp(2).toFloat()
        }
        badge.addView(
            ImageView(ctx).apply {
                setImageDrawable(ShieldDrawable().apply { update(COLOR_RED, ShieldDrawable.Glyph.EXCLAIM, Color.WHITE) })
                importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
            },
            FrameLayout.LayoutParams(dp(26), dp(26), Gravity.CENTER),
        )
        titleRow.addView(badge, LinearLayout.LayoutParams(dp(44), dp(44)).apply { marginEnd = dp(14) })

        val titles = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL }
        cardTitle.apply {
            text = "High-risk call"
            textSize = 19f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            setTextColor(Color.WHITE)
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_ASSERTIVE
        }
        cardSubtitle.apply {
            text = "Possible scam call"
            textSize = 13f
            setTextColor(0xE6FFFFFF.toInt())
        }
        titles.addView(cardTitle)
        titles.addView(cardSubtitle)
        titleRow.addView(titles, LinearLayout.LayoutParams(0, WRAP, 1f))

        val minimize = ImageButton(ctx).apply {
            setImageDrawable(PathIconDrawable(ICON_MINIMIZE, Color.WHITE))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dp(12), dp(12), dp(12), dp(12))
            background = RippleDrawable(ColorStateList.valueOf(0x40FFFFFF), null, null)
            contentDescription = "Minimise warning"
            setOnClickListener {
                alertMinimized = true
                lastState?.let { render(it) }
            }
        }
        titleRow.addView(minimize, LinearLayout.LayoutParams(dp(48), dp(48)))
        header.addView(titleRow)

        val meterRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        cardMeter.apply {
            max = 100
            isIndeterminate = false
            trackThickness = dp(8)
            trackCornerRadius = dp(4)
            setIndicatorColor(Color.WHITE)
            trackColor = 0x47FFFFFF
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        meterRow.addView(cardMeter, LinearLayout.LayoutParams(0, WRAP, 1f))
        cardPercent.apply {
            textSize = 22f
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
            fontFeatureSettings = "tnum"
            setTextColor(Color.WHITE)
            gravity = Gravity.END
            minWidth = dp(64)
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        meterRow.addView(cardPercent, LinearLayout.LayoutParams(WRAP, WRAP).apply { marginStart = dp(12) })
        header.addView(
            meterRow,
            LinearLayout.LayoutParams(MATCH, WRAP).apply {
                topMargin = dp(14)
                marginEnd = dp(8)
            },
        )
        column.addView(header, LinearLayout.LayoutParams(MATCH, WRAP))

        // ---- Body ---------------------------------------------------------------------------
        val body = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(18), dp(20), dp(16))
        }
        body.addView(
            TextView(ctx).apply {
                text = "WHY THIS WAS FLAGGED"
                textSize = 11f
                letterSpacing = 0.08f
                typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                setTextColor(onSurfaceVariant)
            },
        )

        cardChips.apply {
            chipSpacingHorizontal = dp(8)
            chipSpacingVertical = dp(6)
            isSingleLine = false
        }
        body.addView(cardChips, marginTop(dp(8)))

        cardGuidance.apply {
            textSize = 14f
            setTextColor(onSurface)
            setLineSpacing(dp(2).toFloat(), 1f)
        }
        body.addView(cardGuidance, marginTop(dp(14)))

        val buttons = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }
        cardHangUp = MaterialButton(ctx).apply {
            text = if (seniorMode) "Hang up & report" else "Hang up"
            isAllCaps = false
            textSize = if (seniorMode) 16f else 15f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            maxLines = 1
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(COLOR_RED)
            rippleColor = ColorStateList.valueOf(0x40FFFFFF)
            icon = PathIconDrawable(ICON_CALL_END, Color.WHITE)
            iconTint = ColorStateList.valueOf(Color.WHITE)
            iconSize = dp(22)
            iconPadding = dp(8)
            iconGravity = MaterialButton.ICON_GRAVITY_TEXT_START
            cornerRadius = dp(28)
            insetTop = 0
            insetBottom = 0
            minHeight = dp(56)
            minimumHeight = dp(56)
            elevation = dp(2).toFloat()
            contentDescription = "Hang up this call"
            setOnClickListener {
                it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                if (seniorMode && !cardOnHold) actions.onHangUpAndReport() else actions.onHangUp()
            }
        }
        cardSecondary = MaterialButton(ctx, null, MaterialR.attr.materialButtonOutlinedStyle).apply {
            text = "Report $CYBER_HELPLINE"
            isAllCaps = false
            textSize = 15f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            maxLines = 1
            setTextColor(onSurface)
            backgroundTintList = ColorStateList.valueOf(ColorUtils.setAlphaComponent(COLOR_RED, 0x1A))
            strokeColor = ColorStateList.valueOf(outline)
            strokeWidth = dp(1)
            rippleColor = ColorStateList.valueOf(ColorUtils.setAlphaComponent(COLOR_RED, 0x33))
            icon = PathIconDrawable(ICON_PHONE, COLOR_RED)
            iconTint = ColorStateList.valueOf(COLOR_RED)
            iconSize = dp(20)
            iconPadding = dp(8)
            iconGravity = MaterialButton.ICON_GRAVITY_TEXT_START
            cornerRadius = dp(28)
            insetTop = 0
            insetBottom = 0
            minHeight = dp(56)
            minimumHeight = dp(56)
            contentDescription = "Call $CYBER_HELPLINE, the National Cyber Crime Helpline"
            setOnClickListener {
                it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                if (cardOnHold) actions.onContinueCall() else actions.onReport()
            }
        }
        buttons.addView(cardHangUp, LinearLayout.LayoutParams(0, dp(56), 1f).apply { marginEnd = dp(12) })
        buttons.addView(cardSecondary, LinearLayout.LayoutParams(0, dp(56), 1f))
        body.addView(buttons, marginTop(dp(18)))

        body.addView(
            TextView(ctx).apply {
                text = "$CYBER_HELPLINE · National Cyber Crime Helpline · cybercrime.gov.in"
                textSize = 11f
                setTextColor(onSurfaceVariant)
                gravity = Gravity.CENTER
            },
            marginTop(dp(12)),
        )

        column.addView(body, LinearLayout.LayoutParams(MATCH, WRAP))
        card.addView(column, FrameLayout.LayoutParams(MATCH, WRAP))
    }

    // -- Dragging & edge snapping ----------------------------------------------------------------

    @SuppressLint("ClickableViewAccessibility") // Taps are forwarded via performClick().
    private inner class DragHandler : View.OnTouchListener {
        private var downRawX = 0f
        private var downRawY = 0f
        private var startX = 0
        private var startY = 0
        private var dragging = false

        override fun onTouch(view: View, event: MotionEvent): Boolean {
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    snapAnimator?.cancel()
                    downRawX = event.rawX
                    downRawY = event.rawY
                    startX = params.x
                    startY = params.y
                    dragging = false
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downRawX
                    val dy = event.rawY - downRawY
                    if (!dragging && hypot(dx, dy) > touchSlop) dragging = true
                    if (dragging) moveTo(startX + dx.roundToInt(), startY + dy.roundToInt())
                }
                MotionEvent.ACTION_UP -> if (dragging) snapToNearestEdge() else view.performClick()
                MotionEvent.ACTION_CANCEL -> if (dragging) snapToNearestEdge()
            }
            return true
        }
    }

    private fun moveTo(x: Int, y: Int) {
        val (screenW, screenH) = usableScreenSize()
        params.x = x.coerceIn(0, max(0, screenW - root.width))
        params.y = y.coerceIn(0, max(0, screenH - root.height))
        if (root.isAttachedToWindow) windowManager.updateViewLayout(root, params)
    }

    private fun snapToNearestEdge() {
        val (screenW, _) = usableScreenSize()
        anchoredRight = params.x + root.width / 2 > screenW / 2
        val target = if (anchoredRight) max(0, screenW - root.width) else 0
        snapAnimator?.cancel()
        snapAnimator = ValueAnimator.ofInt(params.x, target).apply {
            duration = 220
            interpolator = DecelerateInterpolator(1.6f)
            addUpdateListener { moveTo(it.animatedValue as Int, params.y) }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) = savePosition()
            })
            start()
        }
    }

    /** Keeps the HUD on its edge and on-screen when it grows (pill → card) or shrinks. */
    private fun keepAnchored() {
        val (screenW, _) = usableScreenSize()
        val x = if (anchoredRight) screenW - root.width else 0
        moveTo(x, params.y)
    }

    private fun savePosition() {
        prefs.edit()
            .putInt(KEY_Y, params.y)
            .putBoolean(KEY_ANCHOR_RIGHT, anchoredRight)
            .apply()
    }

    private fun usableScreenSize(): Pair<Int, Int> =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val metrics = windowManager.currentWindowMetrics
            val insets = metrics.windowInsets.getInsetsIgnoringVisibility(
                WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout(),
            )
            Pair(
                metrics.bounds.width() - insets.left - insets.right,
                metrics.bounds.height() - insets.top - insets.bottom,
            )
        } else {
            val dm = ctx.resources.displayMetrics
            Pair(dm.widthPixels, dm.heightPixels)
        }

    /** Alert card width: 380 dp on large screens, full width minus gutters on phones. */
    private fun cardWidth(): Int {
        val (screenW, _) = usableScreenSize()
        val horizontalPadding = root.paddingLeft + root.paddingRight
        return min(dp(380), screenW - horizontalPadding - dp(8))
    }

    private fun themeColor(attr: Int, fallback: Int): Int = MaterialColors.getColor(ctx, attr, fallback)

    private fun marginTop(px: Int) = LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = px }

    private fun percent(score: Double) = "${(score * 100).roundToInt()}%"

    private fun dp(value: Int): Int = (value * density).roundToInt()

    /** `"financial_credential_request:otp"` → `"OTP"`, `"urgency:do not disconnect"` → `"Do not disconnect"`. */
    private fun signalLabel(tag: String): String {
        val phrase = tag.substringAfter(':', tag).replace('_', ' ').trim()
        return when {
            phrase.isEmpty() -> "Suspicious phrase"
            phrase.length <= 4 && ' ' !in phrase && phrase.all { it.code < 0x80 } -> phrase.uppercase()
            else -> phrase.replaceFirstChar { it.uppercaseChar() }
        }
    }

    companion object {
        private const val TAG = "CallHud"

        const val CYBER_HELPLINE = "1930"
        const val ALERT_TITLE = "Warning: High-Risk Interaction Detected"
        const val CAUTION_THRESHOLD = RiskLevel.THRESHOLD_CAUTION
        const val ALERT_THRESHOLD = RiskLevel.THRESHOLD_ALERT

        private const val MAX_CHIPS = 5

        private const val HUD_PREFS = "callguard_hud"
        private const val KEY_Y = "y"
        private const val KEY_ANCHOR_RIGHT = "anchor_right"

        private const val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT
        private const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT

        private const val COLOR_GREEN = 0xFF1E8E3E.toInt()
        private const val COLOR_AMBER = 0xFFE8A200.toInt()
        private const val COLOR_RED = 0xFFB3261E.toInt()
        private const val COLOR_RED_BRIGHT = 0xFFDC362E.toInt()
        private const val COLOR_RED_DEEP = 0xFF7A1A14.toInt()
        private const val COLOR_GREY = 0xFF79747E.toInt()

        private val EMPHASIZED_DECELERATE = PathInterpolator(0.05f, 0.7f, 0.1f, 1f)
        private val EMPHASIZED_ACCELERATE = PathInterpolator(0.3f, 0f, 0.8f, 0.15f)

        // Material Symbols path data (24 × 24 viewport).
        private const val ICON_CALL_END =
            "M12,9c-1.6,0 -3.15,0.25 -4.6,0.72v3.1c0,0.39 -0.23,0.74 -0.56,0.9 -0.98,0.49 -1.87,1.12 " +
                "-2.66,1.85 -0.18,0.18 -0.43,0.28 -0.7,0.28 -0.28,0 -0.53,-0.11 -0.71,-0.29L0.29,13.08" +
                "c-0.18,-0.17 -0.29,-0.42 -0.29,-0.7 0,-0.28 0.11,-0.53 0.29,-0.71C3.34,8.78 7.46,7 12,7" +
                "s8.66,1.78 11.71,4.67c0.18,0.18 0.29,0.43 0.29,0.71 0,0.28 -0.11,0.53 -0.29,0.71l-2.48,2.48" +
                "c-0.18,0.18 -0.43,0.29 -0.71,0.29 -0.27,0 -0.52,-0.11 -0.7,-0.28 -0.79,-0.74 -1.69,-1.36 " +
                "-2.67,-1.85 -0.33,-0.16 -0.56,-0.5 -0.56,-0.9v-3.1C15.15,9.25 13.6,9 12,9z"
        private const val ICON_PHONE =
            "M20.01,15.38c-1.23,0 -2.42,-0.2 -3.53,-0.56 -0.35,-0.12 -0.74,-0.03 -1.01,0.24l-1.57,1.97" +
                "c-2.83,-1.35 -5.48,-3.9 -6.89,-6.83l1.95,-1.66c0.27,-0.28 0.35,-0.67 0.24,-1.02 " +
                "-0.37,-1.11 -0.56,-2.3 -0.56,-3.53 0,-0.54 -0.45,-0.99 -0.99,-0.99H4.19C3.65,3 3,3.24 3,3.99 " +
                "3,13.28 10.73,21 20.01,21c0.71,0 0.99,-0.63 0.99,-1.18v-3.45c0,-0.54 -0.45,-0.99 -0.99,-0.99z"
        private const val ICON_MINIMIZE = "M6,11h12v2H6z"
    }
}

// ----------------------------------------------------------------------------------------------
// Risk level with hysteresis
// ----------------------------------------------------------------------------------------------

internal enum class RiskLevel {
    NORMAL, CAUTION, ALERT;

    companion object {
        const val THRESHOLD_CAUTION = 0.40
        const val THRESHOLD_ALERT = 0.70

        fun fromScore(score: Double): RiskLevel = when {
            score >= THRESHOLD_ALERT -> ALERT
            score >= THRESHOLD_CAUTION -> CAUTION
            else -> NORMAL
        }
    }
}

/**
 * Escalates immediately; de-escalates only after [framesToDowngrade] consecutive lower readings
 * (~1.5 s at the server's 500 ms cadence) so a warning doesn't flicker away mid-sentence.
 */
internal class RiskLevelTracker(private val framesToDowngrade: Int = 3) {
    var level: RiskLevel = RiskLevel.NORMAL
        private set
    private var lowerStreak = 0

    fun update(score: Double): RiskLevel {
        val target = RiskLevel.fromScore(score)
        if (target >= level) {
            level = target
            lowerStreak = 0
        } else if (++lowerStreak >= framesToDowngrade) {
            level = target
            lowerStreak = 0
        }
        return level
    }
}

// ----------------------------------------------------------------------------------------------
// Vector drawables (no resource files needed)
// ----------------------------------------------------------------------------------------------

/** Shield badge with an optional check / exclamation glyph. */
private class ShieldDrawable : Drawable() {
    enum class Glyph { NONE, CHECK, EXCLAIM }

    private var fillColor = Color.WHITE
    private var glyphColor = Color.BLACK
    private var glyph = Glyph.NONE
    private var drawAlpha = 255

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val glyphPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val shieldPath = Path()
    private val checkPath = Path()

    fun update(fill: Int, glyph: Glyph, glyphColor: Int) {
        if (fill == fillColor && glyph == this.glyph && glyphColor == this.glyphColor) return
        fillColor = fill
        this.glyph = glyph
        this.glyphColor = glyphColor
        invalidateSelf()
    }

    override fun onBoundsChange(bounds: Rect) {
        val w = bounds.width().toFloat()
        val h = bounds.height().toFloat()
        val l = bounds.left.toFloat()
        val t = bounds.top.toFloat()
        shieldPath.reset()
        shieldPath.moveTo(l + 0.50f * w, t + 0.04f * h)
        shieldPath.lineTo(l + 0.90f * w, t + 0.19f * h)
        shieldPath.lineTo(l + 0.90f * w, t + 0.47f * h)
        shieldPath.cubicTo(l + 0.90f * w, t + 0.73f * h, l + 0.73f * w, t + 0.89f * h, l + 0.50f * w, t + 0.97f * h)
        shieldPath.cubicTo(l + 0.27f * w, t + 0.89f * h, l + 0.10f * w, t + 0.73f * h, l + 0.10f * w, t + 0.47f * h)
        shieldPath.lineTo(l + 0.10f * w, t + 0.19f * h)
        shieldPath.close()

        checkPath.reset()
        checkPath.moveTo(l + 0.31f * w, t + 0.51f * h)
        checkPath.lineTo(l + 0.45f * w, t + 0.65f * h)
        checkPath.lineTo(l + 0.70f * w, t + 0.38f * h)

        glyphPaint.strokeWidth = 0.11f * w
    }

    override fun draw(canvas: Canvas) {
        fillPaint.color = fillColor
        fillPaint.alpha = Color.alpha(fillColor) * drawAlpha / 255
        canvas.drawPath(shieldPath, fillPaint)

        glyphPaint.color = glyphColor
        glyphPaint.alpha = drawAlpha
        dotPaint.color = glyphColor
        dotPaint.alpha = drawAlpha

        val b = bounds
        val w = b.width().toFloat()
        val h = b.height().toFloat()
        when (glyph) {
            Glyph.NONE -> Unit
            Glyph.CHECK -> canvas.drawPath(checkPath, glyphPaint)
            Glyph.EXCLAIM -> {
                val cx = b.left + 0.5f * w
                canvas.drawLine(cx, b.top + 0.27f * h, cx, b.top + 0.56f * h, glyphPaint)
                canvas.drawCircle(cx, b.top + 0.73f * h, 0.065f * w, dotPaint)
            }
        }
    }

    override fun setAlpha(alpha: Int) {
        drawAlpha = alpha
        invalidateSelf()
    }

    override fun setColorFilter(colorFilter: ColorFilter?) {
        fillPaint.colorFilter = colorFilter
        glyphPaint.colorFilter = colorFilter
        dotPaint.colorFilter = colorFilter
        invalidateSelf()
    }

    @Deprecated("Deprecated in Java")
    override fun getOpacity(): Int = PixelFormat.TRANSLUCENT
}

/**
 * Renders SVG path data (Material Symbols, 24 × 24 viewport) as a tintable drawable, so the HUD
 * can use real Material icons without shipping vector resources. Honours [setTintList], which
 * [MaterialButton] uses for `iconTint`.
 */
private class PathIconDrawable(pathData: String, color: Int, private val viewport: Float = 24f) : Drawable() {
    private val source: Path = PathParser.createPathFromPathData(pathData)
    private val scaled = Path()
    private val matrix = Matrix()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        this.color = color
    }
    private var tint: ColorStateList? = null
    private var drawAlpha = 255

    override fun onBoundsChange(bounds: Rect) {
        matrix.setScale(bounds.width() / viewport, bounds.height() / viewport)
        matrix.postTranslate(bounds.left.toFloat(), bounds.top.toFloat())
        source.transform(matrix, scaled)
    }

    override fun draw(canvas: Canvas) {
        canvas.drawPath(scaled, paint)
    }

    override fun setTintList(tint: ColorStateList?) {
        this.tint = tint
        applyTint(state)
    }

    override fun isStateful(): Boolean = tint?.isStateful == true

    override fun onStateChange(state: IntArray): Boolean = applyTint(state)

    private fun applyTint(state: IntArray): Boolean {
        val list = tint ?: return false
        val color = list.getColorForState(state, list.defaultColor)
        val withAlpha = ColorUtils.setAlphaComponent(color, Color.alpha(color) * drawAlpha / 255)
        if (withAlpha == paint.color) return false
        paint.color = withAlpha
        invalidateSelf()
        return true
    }

    override fun setAlpha(alpha: Int) {
        drawAlpha = alpha
        paint.alpha = alpha
        invalidateSelf()
    }

    override fun setColorFilter(colorFilter: ColorFilter?) {
        paint.colorFilter = colorFilter
        invalidateSelf()
    }

    @Deprecated("Deprecated in Java")
    override fun getOpacity(): Int = PixelFormat.TRANSLUCENT
}
