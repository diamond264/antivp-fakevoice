package kr.ac.kaist.nmsl.antivp.modules.text_based_detection

import android.os.Bundle
import android.util.Log
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module

class TextBasedDetectionModule : Module() {
    private val TAG = "TextBasedDetection"

    init {
        subscribeEvent(EventType.TEXT_TRANSCRIBED)
    }

    override fun name(): String {
        return "text_based_detection"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            EventType.TEXT_TRANSCRIBED -> {
                Log.d(TAG, "Rcvd a transcribed call dialogue.")

                val dialogue = bundle.getStringArray("dialogue")!!
                for (utter in dialogue)
                    Log.d(TAG, utter)

                val phishingType = detectPhishingType(dialogue)

                if (phishingType != null) {
                    bundle.putString("phishing_type", phishingType)
                    raiseEvent(EventType.PHISHING_CALL_DETECTED, bundle)
                }
            }
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }

    private fun detectPhishingType(dialogue: Array<String>): String? {
        return "insurance_scam"
    }
}