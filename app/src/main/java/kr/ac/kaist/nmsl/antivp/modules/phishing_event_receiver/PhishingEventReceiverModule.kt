package kr.ac.kaist.nmsl.antivp.modules.phishing_event_receiver

import android.os.Bundle
import android.util.Log
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module

class PhishingEventReceiverModule : Module() {
    private val TAG = "PhishingEventReceiver"

    init {
        subscribeEvent(EventType.PHISHING_CALL_DETECTED)
        subscribeEvent(EventType.PHISHING_APP_DETECTED)
    }

    override fun name(): String {
        return "phishing_event_receiver"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            EventType.PHISHING_CALL_DETECTED -> {
                Log.d(TAG, "Rcvd a suspicious phone call.")
            }
            EventType.PHISHING_APP_DETECTED -> {
                Log.d(TAG, "Detected a suspicious app.")
            }
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }
}