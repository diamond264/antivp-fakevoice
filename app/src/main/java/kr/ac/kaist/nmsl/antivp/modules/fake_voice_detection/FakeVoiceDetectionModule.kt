package kr.ac.kaist.nmsl.antivp.modules.fake_voice_detection

import android.os.Bundle
import android.util.Log
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module
import kr.ac.kaist.nmsl.antivp.core.util.FileManager
import org.pytorch.IValue
import org.pytorch.Tensor
import org.pytorch.Module as torchModule


class FakeVoiceDetectionModule : Module() {
    private val TAG = "FakeVoiceDetection"
    val mOngoingRecordings = mutableSetOf<String>()
    private val INPUT_SIZE: Long = 64600
    private val SAMPLE_RATE: Long = 16000
    private val CHUNK_SIZE: Int = 5

    init {
        subscribeEvent(EventType.CALL_OFFHOOK)
    }

    override fun name(): String {
        return "fake_voice_detection"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when(type) {
            EventType.CALL_OFFHOOK -> {
                Log.d(TAG, "Rcvd a phone call.")

                Log.d(TAG, "Check if the recording is exist")
                val filename = bundle.getString("record_file")
                filename ?: return

                GlobalScope.launch {
                    synchronized(mOngoingRecordings) {
                        mOngoingRecordings.add(filename)
                    }
                    detectFakeVoice(filename, bundle)
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

    private fun detectFakeVoice(path: String, bundle: Bundle) {
        var stillRecording = true

        while (true) {
            Thread.sleep(1000L)

            synchronized(mOngoingRecordings) {
                stillRecording = mOngoingRecordings.contains(path)
            }
            if (!stillRecording)
                break

            Log.d(TAG, "Load audio from file")
            val fileManager = FileManager.getInstance()
            val audioBuffer = fileManager.load(path)
            if (audioBuffer == null) {
                Log.e(TAG, "Failed to load audio from file")
                return
            }

            val floatInputBuffer = FloatArray((CHUNK_SIZE * SAMPLE_RATE).toInt())

            /* feed in float values between -1.0f and 1.0f by dividing the signed 16-bit inputs. */
            for (i in 0 until (CHUNK_SIZE * SAMPLE_RATE).toInt()) {
                floatInputBuffer[i] = audioBuffer[i] / Short.MAX_VALUE.toFloat()
            }
            val result = recognize(floatInputBuffer)
            if (result == "1") {
                Log.d(TAG, "Fake voice detected")
                bundle.putString("phishing_type", "fake_voice")
                raiseEvent(EventType.PHISHING_CALL_DETECTED, bundle)
                return
            }

        }
    }

    /**
     * Load module and recognize the input audio whether it is spoof or bonafide
     * @param floatInputBuffer input audio in float array
     * @return result in string format. 1 is spoof, 0 is bonafide
     */
    private fun recognize(floatInputBuffer: FloatArray): String {
        Log.d(TAG, "Start recognizing")
        /* load module */
        val fileManager = FileManager.getInstance()
        val moduleEncoder = torchModule.load(fileManager.assetFilePath("btsdetect.ptl"))

        /* convert float array to tensor */
        val inTensorBuffer = Tensor.allocateFloatBuffer(INPUT_SIZE.toInt())
        for (i in 0 until INPUT_SIZE) {
            inTensorBuffer.put(floatInputBuffer[i.toInt()])
        }

        val inTensor = Tensor.fromBlob(
            /* data = */
            inTensorBuffer,
            /* shape = */
            longArrayOf(1, INPUT_SIZE),
        )
        val result = moduleEncoder.forward(IValue.from(inTensor)).toLong()

        when (result) {
            1L -> {
                Log.d("Result", "spoof")
                Log.d("Result", "Result: 1")
            }
            0L -> {
                Log.d("Result", "bonafide")
                Log.d("Result", "Result: 0")
            }
            else -> {
                Log.d("Result", "Result: $result")
            }
        }

        return result.toString()
    }


}