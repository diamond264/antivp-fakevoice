package kr.ac.kaist.nmsl.antivp.modules.speech_to_text

import android.os.Bundle
import android.util.Log
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module
import kotlinx.coroutines.*
import java.lang.Thread.sleep

class SpeechToTextModule() : Module() {
    private val TAG = "SpeechToText"
    val mOngoingRecordings = mutableSetOf<String>()

    init {
        subscribeEvent(EventType.CALL_OFFHOOK)
        subscribeEvent(EventType.CALL_IDLE)
    }

    override fun name(): String {
        return "speech_to_text"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            EventType.CALL_OFFHOOK -> {
                Log.d(TAG, "Rcvd a phone call.")

                val filename = bundle.getString("record_file")
                filename ?: return

                GlobalScope.launch {
                    synchronized(mOngoingRecordings) {
                        mOngoingRecordings.add(filename)
                    }
                    transcribeFile(filename)
                }
                Log.d(TAG, "Started transcribing $filename")
            }
            EventType.CALL_IDLE -> {
                Log.d(TAG, "A phone call hung up.")

                val filename = bundle.getString("record_file")
                filename ?: return

                synchronized(mOngoingRecordings) {
                    mOngoingRecordings.remove(filename)
                }
            }
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }

    fun transcribeFile(path: String) {
        var stillRecording = true

        while (true) {
            sleep(1000L)

            synchronized(mOngoingRecordings) {
                stillRecording = mOngoingRecordings.contains(path)
            }

            if (!stillRecording)
                break

            val bundle = Bundle()
            bundle.putInt("dialogue_id", 0)
            bundle.putStringArray("dialogue", arrayOf(
                "Hello",
                "Hi",
                "How are you?"
            ))

            raiseEvent(EventType.TEXT_TRANSCRIBED, bundle)
        }
    }
}