package kr.ac.kaist.nmsl.antivp.modules.mal_activity_detection

import android.os.Bundle
import android.util.Log
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module

class MalActivityDetectionModule: Module() {
    private val TAG = "MalActivityDetection"

    override fun name(): String {
        return "mal_activity_detection"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }

}