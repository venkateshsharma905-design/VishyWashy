package com.vishiwashi.callguard.ui

import android.content.Context
import android.content.res.Configuration
import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.os.Build
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.animation.OvershootInterpolator
import android.view.animation.PathInterpolator
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.graphics.ColorUtils
import com.vishiwashi.callguard.settings.ProtectionSettings
import kotlin.math.roundToInt

/**
 * A soft glassmorphism toolkit: frosted translucent cards over a gradient background, with a thin
 * light border and a diffuse shadow, plus pressable buttons, pills, inputs and tab chips. One
 * palette drives light and dark. [GlassDrawable] does the drawing; [Neu] builds the views.
 */
data class NeuPalette(
    val bgTop: Int,
    val bgBottom: Int,
    val glassTop: Int,
    val glassBottom: Int,
    val glassBorder: Int,
    val insetFill: Int,
    val shadow: Int,
    val textPrimary: Int,
    val textSecondary: Int,
    val accent: Int,
    val onAccent: Int,
    val danger: Int,
    val success: Int,
    val warning: Int,
    val dark: Boolean,
) {
    val background: Int get() = bgBottom

    companion object {
        private val DARK = NeuPalette(
            bgTop = 0xFF17202B.toInt(),
            bgBottom = 0xFF0C1218.toInt(),
            glassTop = 0x1FFFFFFF,
            glassBottom = 0x0AFFFFFF,
            glassBorder = 0x24FFFFFF,
            insetFill = 0x33000000,
            shadow = 0x6E000000,
            textPrimary = 0xFFF1F5F9.toInt(),
            textSecondary = 0xFF9FB1C2.toInt(),
            accent = 0xFF5B9DFF.toInt(),
            onAccent = 0xFFFFFFFF.toInt(),
            danger = 0xFFFF6B6B.toInt(),
            success = 0xFF45D18A.toInt(),
            warning = 0xFFF7C255.toInt(),
            dark = true,
        )
        private val LIGHT = NeuPalette(
            bgTop = 0xFFF2F6FC.toInt(),
            bgBottom = 0xFFDEE7F2.toInt(),
            glassTop = 0xCCFFFFFF.toInt(),
            glassBottom = 0x80FFFFFF.toInt(),
            glassBorder = 0xE6FFFFFF.toInt(),
            insetFill = 0x14000000,
            shadow = 0x30536175,
            textPrimary = 0xFF1F2E3D.toInt(),
            textSecondary = 0xFF5B6B7C.toInt(),
            accent = 0xFF2F6BFF.toInt(),
            onAccent = 0xFFFFFFFF.toInt(),
            danger = 0xFFE23B3B.toInt(),
            success = 0xFF1E9E63.toInt(),
            warning = 0xFFC8860B.toInt(),
            dark = false,
        )

        fun resolve(context: Context, mode: String): NeuPalette = when (mode) {
            ProtectionSettings.THEME_LIGHT -> LIGHT
            ProtectionSettings.THEME_DARK -> DARK
            else -> {
                val night = context.resources.configuration.uiMode and
                    Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
                if (night) DARK else LIGHT
            }
        }
    }
}

/**
 * A frosted-glass rounded rectangle: a translucent top-to-bottom gradient fill, a hairline light
 * border, a bright top sheen, and a soft drop shadow. [inset] draws it recessed (for inputs).
 */
class GlassDrawable(
    private val radius: Float,
    private val topFill: Int,
    private val bottomFill: Int,
    private val border: Int,
    private val shadow: Int,
    private val blur: Float,
    private val offset: Float,
    private val inset: Boolean = false,
) : Drawable() {

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.2f
        color = border
    }
    private val sheenPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 1.1f }
    private val pad = (blur + kotlin.math.abs(offset) + 1f).roundToInt()

    override fun onBoundsChange(bounds: Rect) {
        val r = RectF(bounds).apply { inset(pad.toFloat(), pad.toFloat()) }
        fillPaint.shader = LinearGradient(0f, r.top, 0f, r.bottom, topFill, bottomFill, Shader.TileMode.CLAMP)
        sheenPaint.shader = LinearGradient(
            0f, r.top, 0f, r.top + r.height() * 0.5f,
            ColorUtils.setAlphaComponent(0xFFFFFF, 0x55), 0x00FFFFFF, Shader.TileMode.CLAMP,
        )
    }

    override fun draw(canvas: Canvas) {
        val r = RectF(bounds).apply { inset(pad.toFloat(), pad.toFloat()) }
        if (!inset) {
            shadowPaint.color = shadow
            shadowPaint.setShadowLayer(blur, 0f, offset, shadow)
            canvas.drawRoundRect(r, radius, radius, shadowPaint)
            shadowPaint.clearShadowLayer()
        }
        canvas.drawRoundRect(r, radius, radius, fillPaint)
        if (!inset) canvas.drawRoundRect(RectF(r).apply { inset(0.6f, 0.6f) }, radius, radius, sheenPaint)
        canvas.drawRoundRect(RectF(r).apply { inset(0.6f, 0.6f) }, radius, radius, borderPaint)
    }

    // No intrinsic padding: every view sets its own (shadow room included). Reporting padding here
    // made Android replace a view's padding whenever its background changed — e.g. a button
    // swapping to its pressed state shrank to a thin bar.
    override fun getPadding(padding: Rect): Boolean = false

    override fun setAlpha(alpha: Int) {}
    override fun setColorFilter(colorFilter: ColorFilter?) {}

    @Deprecated("Deprecated in Java")
    override fun getOpacity(): Int = PixelFormat.TRANSLUCENT
}

/** View builders bound to one palette. Glass shadows draw on the GPU (shadow layers on any shape
 *  are hardware-accelerated from API 28, the app's minimum), so scrolling and animation stay smooth. */
class Neu(val ctx: Context, val p: NeuPalette) {

    private val density = ctx.resources.displayMetrics.density
    fun dp(v: Int): Int = (v * density).roundToInt()

    /** A full-screen vertical gradient for the activity background. */
    fun backgroundDrawable(): Drawable =
        GradientDrawable(GradientDrawable.Orientation.TOP_BOTTOM, intArrayOf(p.bgTop, p.bgBottom))

    fun glass(radius: Int, elevation: Int, top: Int = p.glassTop, bottom: Int = p.glassBottom,
              border: Int = p.glassBorder, inset: Boolean = false): GlassDrawable =
        GlassDrawable(dp(radius).toFloat(), top, bottom, border, p.shadow,
            dp(elevation).toFloat(), dp(elevation).toFloat() * 0.55f, inset)

    fun card(radius: Int = 24, elevation: Int = 10, padding: Int = 18): LinearLayout =
        LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            val bg = glass(radius, elevation)
            background = bg
            val shadowPad = dp(elevation) + dp(elevation)
            setPadding(shadowPad + dp(padding), shadowPad + dp(padding), shadowPad + dp(padding), shadowPad + dp(padding))
        }

    fun title(text: String, size: Float = 16f, color: Int = p.textPrimary, bold: Boolean = true): TextView =
        TextView(ctx).apply {
            this.text = text
            textSize = size
            setTextColor(color)
            if (bold) typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
        }

    fun body(text: String, size: Float = 13.5f, color: Int = p.textSecondary): TextView =
        TextView(ctx).apply {
            this.text = text
            textSize = size
            setTextColor(color)
            setLineSpacing(dp(2).toFloat(), 1f)
        }

    fun button(text: String, filled: Boolean = false, color: Int = p.accent, compact: Boolean = false, onClick: () -> Unit): TextView {
        val radius = dp(18)
        val elev = 6
        val raised = if (filled) {
            glass(radius, elev, lighten(color, 0.10f), darken(color, 0.16f), ColorUtils.setAlphaComponent(0xFFFFFF, 0x40))
        } else {
            glass(radius, elev)
        }
        val pressed = glass(radius, 3, if (filled) darken(color, 0.20f) else p.insetFill, if (filled) darken(color, 0.30f) else p.insetFill,
            p.glassBorder, inset = true)
        return TextView(ctx).apply {
            this.text = text
            gravity = Gravity.CENTER
            textSize = if (compact) 13f else 15f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            setTextColor(if (filled) p.onAccent else color)
            isClickable = true
            isFocusable = true
            maxLines = 1
            isSingleLine = true
            ellipsize = TextUtils.TruncateAt.END
            background = raised
            val shadowPad = dp(elev) + dp(elev)
            val hpad = if (compact) dp(3) else dp(14)
            setPadding(shadowPad + hpad, shadowPad + dp(10), shadowPad + hpad, shadowPad + dp(10))
            setOnTouchListener { v, e ->
                when (e.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        v.background = pressed
                        v.animate().scaleX(0.96f).scaleY(0.96f).setDuration(90).setInterpolator(EMPHASIZED).start()
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        v.background = raised
                        v.animate().scaleX(1f).scaleY(1f).setDuration(220).setInterpolator(OVERSHOOT).start()
                    }
                }
                false
            }
            setOnClickListener { haptic(it); onClick() }
        }
    }

    fun pill(text: String, color: Int): TextView = TextView(ctx).apply {
        this.text = text
        textSize = 11.5f
        typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
        setTextColor(color)
        background = glass(12, 3, ColorUtils.setAlphaComponent(color, 0x2E), ColorUtils.setAlphaComponent(color, 0x1A),
            ColorUtils.setAlphaComponent(color, 0x55))
        val sp = dp(5)
        setPadding(sp + dp(10), sp + dp(4), sp + dp(10), sp + dp(4))
    }

    /** Recessed glass for text inputs. */
    fun inputBackground(): GlassDrawable = glass(14, 5, p.insetFill, p.insetFill, p.glassBorder, inset = true)

    /** Highlighted glass chip for a selected tab. */
    fun tabChip(): GlassDrawable =
        glass(16, 5, ColorUtils.setAlphaComponent(p.accent, 0x30), ColorUtils.setAlphaComponent(p.accent, 0x14),
            ColorUtils.setAlphaComponent(p.accent, 0x55))

    fun knobBackground(on: Boolean): GlassDrawable = if (on) {
        glass(14, 4, lighten(p.accent, 0.08f), darken(p.accent, 0.14f), ColorUtils.setAlphaComponent(0xFFFFFF, 0x40))
    } else {
        glass(14, 4, p.insetFill, p.insetFill, p.glassBorder, inset = true)
    }

    fun row(gravity: Int = Gravity.CENTER_VERTICAL): LinearLayout = LinearLayout(ctx).apply {
        orientation = LinearLayout.HORIZONTAL
        this.gravity = gravity
    }

    /** Colour of floating surfaces (snackbars) in this palette. */
    val surfaceColor: Int get() = if (p.dark) 0xFF222C38.toInt() else 0xFFFFFFFF.toInt()

    /** Shrinks [view] slightly while pressed and springs it back on release. */
    fun pressable(view: View, scale: Float = 0.94f): View = view.apply {
        setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN ->
                    v.animate().scaleX(scale).scaleY(scale).setDuration(90).setInterpolator(EMPHASIZED).start()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                    v.animate().scaleX(1f).scaleY(1f).setDuration(240).setInterpolator(OVERSHOOT).start()
            }
            false
        }
    }

    /** A short tap of the vibration motor (respects the system's touch-feedback setting). */
    fun haptic(view: View, strong: Boolean = false) {
        val kind = when {
            strong && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R -> HapticFeedbackConstants.CONFIRM
            strong -> HapticFeedbackConstants.VIRTUAL_KEY
            else -> HapticFeedbackConstants.CONTEXT_CLICK
        }
        view.performHapticFeedback(kind)
    }

    /** A quick squash-and-spring, for a switch that just flipped. */
    fun bounce(view: View) {
        view.animate().cancel()
        view.scaleX = 0.82f
        view.scaleY = 0.82f
        view.animate().scaleX(1f).scaleY(1f).setDuration(320).setInterpolator(OVERSHOOT).start()
    }

    fun spacer(height: Int): View = View(ctx).apply {
        layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(height))
    }

    companion object {
        val EMPHASIZED = PathInterpolator(0.2f, 0f, 0f, 1f)
        val OVERSHOOT = OvershootInterpolator(2.2f)

        fun lighten(color: Int, t: Float): Int = blend(color, 0xFFFFFFFF.toInt(), t)
        fun darken(color: Int, t: Float): Int = blend(color, 0xFF000000.toInt(), t)
        private fun blend(a: Int, b: Int, t: Float): Int {
            fun ch(shift: Int) = (((a shr shift and 0xFF) * (1 - t)) + ((b shr shift and 0xFF) * t)).roundToInt()
            return (0xFF shl 24) or (ch(16) shl 16) or (ch(8) shl 8) or ch(0)
        }
    }
}
