package kr.ac.kaist.nmsl.antivp.modules.mal_app_detection

import android.os.Bundle
import android.util.Log
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module

class MalAppDetectionModule : Module() {
    private val TAG = "MalAppDetection"

    override fun name(): String {
        return "mal_app_detection"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }
}