package kr.ac.kaist.nmsl.antivp.modules.net_based_detection

import android.os.Bundle
import android.util.Log
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module

class NetBasedDetectionModule : Module() {
    private val TAG = "NetBasedDetection"

    override fun name(): String {
        return "net_based_detection"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }
}