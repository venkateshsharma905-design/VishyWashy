package com.vishiwashi.callguard.consent

import android.Manifest
import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.vishiwashi.callguard.call.CallGuardService
import com.vishiwashi.callguard.settings.ProtectionSettings

/**
 * Runs the arming gate in a fixed order, so nothing is captured before each step is satisfied:
 *
 *   1. DPDP consent dialog          (required — nothing else is requested before this)
 *   2. Runtime permissions           (RECORD_AUDIO required; READ_PHONE_STATE, ANSWER_PHONE_CALLS optional)
 *   3. Server address                (required)
 *   4. Accessibility service on      (required — the only way to hear audio during a call)
 *   5. Armed                         ([CallGuardService] then starts by itself on every call)
 *
 * Must be constructed before the activity reaches STARTED (i.e. in `onCreate`), because it
 * registers Activity Result launchers. Call [onResume] from the activity's `onResume`.
 */
class HudLaunchCoordinator(
    private val activity: ComponentActivity,
    /** Returns the telemetry WebSocket base URL, or null when no valid address has been entered. */
    private val endpointProvider: () -> String?,
    private val onStatus: (String) -> Unit,
) {
    private val consentManager = DpdpConsentManager(activity)
    private val settings = ProtectionSettings(activity)
    private var optionalPermissionsRequested = false
    private var awaitingAccessibility = false
    private val askedPermissions = mutableSetOf<String>()

    /** One line of the setup checklist shown until everything is in place. */
    data class SetupStep(
        val title: String,
        val detail: String,
        val done: Boolean,
        val required: Boolean,
        val fixLabel: String,
        val fix: () -> Unit,
    )

    private val stepPermissionLauncher = activity.registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { onStatus(currentStatus()) }

    private val permissionLauncher = activity.registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) {
        optionalPermissionsRequested = true
        when {
            isGranted(Manifest.permission.RECORD_AUDIO) -> arm()
            activity.shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) ->
                onStatus("Microphone access is needed for live scam detection.")
            else ->
                onStatus("Microphone access is blocked. Enable it in Settings › Apps › VishyWashy › Permissions.")
        }
    }

    fun arm() {
        when {
            !consentManager.hasValidConsent() ->
                consentManager.showConsentDialog(activity) { granted ->
                    if (granted) arm() else onStatus("Consent not given. No audio has been accessed.")
                }

            needsPermissionRequest() ->
                permissionLauncher.launch(missingPermissions().toTypedArray())

            endpointProvider() == null ->
                onStatus("Enter the server address shown in the server console first.")

            !CallGuardService.isEnabled(activity) -> promptAccessibility()

            else -> {
                settings.armed = true
                onStatus(currentStatus())
            }
        }
    }

    fun disarm() {
        settings.armed = false
        onStatus(currentStatus())
    }

    fun withdrawConsent() {
        settings.armed = false
        consentManager.withdrawConsent()
        onStatus("Consent withdrawn. Audio analysis has stopped.")
    }

    /** Finishes arming after the user returns from Accessibility settings. */
    fun onResume() {
        if (awaitingAccessibility && CallGuardService.isEnabled(activity)) {
            awaitingAccessibility = false
            arm()
        } else {
            onStatus(currentStatus())
        }
    }

    fun currentStatus(): String = when {
        !consentManager.hasValidConsent() -> "Protection is off. Your consent is needed first."
        !CallGuardService.isEnabled(activity) ->
            if (settings.armed) "Accessibility service is off: VishyWashy can't hear calls." else "Protection is off."
        !settings.armed -> "Protection is off."
        else -> "Armed. VishyWashy appears automatically when a call connects."
    }

    val isArmed: Boolean
        get() = settings.armed && consentManager.hasValidConsent() && CallGuardService.isEnabled(activity)

    /**
     * Everything VishyWashy needs after a plain APK install (no USB tools): what's done, what's
     * missing, and how to fix each. Missing *required* steps keep protection from arming.
     */
    fun setupSteps(): List<SetupStep> {
        val sdk = Build.VERSION.SDK_INT
        return buildList {
            add(SetupStep("Consent", "Agree to how call audio is analysed (DPDP Act).",
                consentManager.hasValidConsent(), required = true, fixLabel = "Review") {
                consentManager.showConsentDialog(activity) { onStatus(currentStatus()) }
            })
            add(permissionStep("Microphone", "Hears the call while protection is on.", required = true,
                Manifest.permission.RECORD_AUDIO))
            add(permissionStep("Phone & calls", "Notices calls and can hang up a scam call for you.",
                required = false, Manifest.permission.READ_PHONE_STATE, Manifest.permission.ANSWER_PHONE_CALLS))
            if (sdk >= Build.VERSION_CODES.TIRAMISU) {
                add(permissionStep("Notifications", "Lets you report a flagged call after it ends.",
                    required = false, Manifest.permission.POST_NOTIFICATIONS))
            }
            add(SetupStep("Accessibility service",
                "Settings › Accessibility › VishyWashy call protection › On." +
                    if (sdk >= Build.VERSION_CODES.TIRAMISU) " Greyed out? App info › ⋮ › Allow restricted settings first." else "",
                CallGuardService.isEnabled(activity), required = true, fixLabel = "Turn on") { promptAccessibility() })
            add(SetupStep("Battery", "Allow VishyWashy to run in the background so the phone doesn't stop it.",
                ignoresBatteryOptimizations(), required = false, fixLabel = "Allow") { requestBatteryExemption() })
        }
    }

    private fun permissionStep(title: String, detail: String, required: Boolean, vararg permissions: String): SetupStep {
        val missing = permissions.filterNot(::isGranted)
        return SetupStep(title, detail, missing.isEmpty(), required, fixLabel = "Allow") {
            // Asked before and refused for good: Android won't ask again, so open App info.
            val blocked = missing.any { it in askedPermissions && !activity.shouldShowRequestPermissionRationale(it) }
            if (blocked) {
                open(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${activity.packageName}")))
            } else {
                askedPermissions += missing
                stepPermissionLauncher.launch(missing.toTypedArray())
            }
        }
    }

    private fun ignoresBatteryOptimizations(): Boolean =
        activity.getSystemService(PowerManager::class.java)?.isIgnoringBatteryOptimizations(activity.packageName) == true

    @SuppressLint("BatteryLife") // Sideloaded safety app: the accessibility service must not be frozen.
    private fun requestBatteryExemption() {
        try {
            activity.startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                Uri.parse("package:${activity.packageName}")))
        } catch (e: ActivityNotFoundException) {
            open(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${activity.packageName}")))
        }
    }

    fun promptAccessibility() {
        val restricted = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
        val message = buildString {
            append(
                "Android only lets accessibility services hear audio during a call, so VishyWashy " +
                    "needs its accessibility service switched on.\n\n" +
                    "Settings › Accessibility › Installed apps › VishyWashy call protection › On.",
            )
            if (restricted) {
                append(
                    "\n\nIf the switch is greyed out (\"Restricted setting\"), open App info, tap ⋮ " +
                        "in the top corner, choose \"Allow restricted settings\", then try again.",
                )
            }
        }
        val builder = AlertDialog.Builder(activity)
            .setTitle("Turn on VishyWashy call protection")
            .setMessage(message)
            .setPositiveButton("Open Accessibility") { _, _ ->
                awaitingAccessibility = true
                open(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            }
            .setNegativeButton("Not now", null)
        if (restricted) {
            builder.setNeutralButton("App info") { _, _ ->
                awaitingAccessibility = true
                open(
                    Intent(
                        Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:${activity.packageName}"),
                    ),
                )
            }
        }
        builder.show()
        onStatus("Switch on \"VishyWashy call protection\" in Accessibility, then come back.")
    }

    private fun open(intent: Intent) {
        try {
            activity.startActivity(intent)
        } catch (e: ActivityNotFoundException) {
            onStatus("Open Settings › Accessibility and switch on VishyWashy call protection.")
        }
    }

    private fun missingPermissions(): List<String> = buildList {
        add(Manifest.permission.RECORD_AUDIO)
        add(Manifest.permission.READ_PHONE_STATE)
        add(Manifest.permission.ANSWER_PHONE_CALLS)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) add(Manifest.permission.POST_NOTIFICATIONS)
    }.filterNot(::isGranted)

    private fun needsPermissionRequest(): Boolean {
        val missing = missingPermissions()
        return Manifest.permission.RECORD_AUDIO in missing ||
            (!optionalPermissionsRequested && missing.isNotEmpty())
    }

    private fun isGranted(permission: String): Boolean =
        ContextCompat.checkSelfPermission(activity, permission) == PackageManager.PERMISSION_GRANTED
}
