plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.vishiwashi.callguard"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.vishiwashi.callguard"
        minSdk = 28 // TelecomManager.endCall() requires API 28
        targetSdk = 35
        versionCode = 8
        versionName = "1.6.0"

        // Base URL of the edge server's telemetry socket; the session id is appended.
        buildConfigField("String", "EDGE_WS_URL", "\"wss://edge.example.com/ws/telemetry\"")
        manifestPlaceholders["usesCleartextTraffic"] = "false"
    }

    // One signing key kept with the project, so every build — on any machine — can update an
    // installed copy through the normal APK installer (Android refuses updates signed differently).
    signingConfigs {
        create("vishywashy") {
            storeFile = rootProject.file("keystore/vishywashy.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("vishywashy")
            // LAN demo against `python server.py`. This is only the prefilled default: the address
            // can be changed in the app. Build with `-PedgeHost=192.168.1.20:8000` to bake in your
            // laptop's address (10.0.2.2 = the host machine as seen from the emulator).
            val edgeHost = (project.findProperty("edgeHost") as String?)?.trim().orEmpty().ifEmpty { "10.0.2.2:8000" }
            buildConfigField("String", "EDGE_WS_URL", "\"ws://$edgeHost/ws/telemetry\"")
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }
        release {
            signingConfig = signingConfigs.getByName("vishywashy")
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // Material 3 components + Material You dynamic colour for the overlay HUD.
    implementation("com.google.android.material:material:1.12.0")
}
