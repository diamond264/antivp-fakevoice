package kr.ac.kaist.nmsl.antivp.modules.call_event_generation

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.Telephony
import android.telephony.TelephonyManager
import android.util.Log
import kr.ac.kaist.nmsl.antivp.AntiVPApplication
import kr.ac.kaist.nmsl.antivp.core.EventType
import java.io.IOException
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.*

class CallBroadcastReceiver : BroadcastReceiver() {
    val TAG = "CallTracker"

    var mCallRecorder: CallRecorder? = null

    companion object {
        const val ACTION_CALL_IN = "android.intent.action.PHONE_STATE"
        const val ACTION_CALL_OUT = "android.intent.action.NEW_OUTGOING_CALL"
        const val ACTION_SMS_IN = "android.provider.Telephony.SMS_RECEIVED"
    }

    fun registerSelf(context: Context) {
        val filter = IntentFilter()
        filter.addAction(ACTION_CALL_OUT)
        filter.addAction(ACTION_CALL_IN)
        filter.addAction(ACTION_SMS_IN)
        context.applicationContext.registerReceiver(this, filter)
    }

    override fun onReceive(context: Context?, intent: Intent) {
        Log.d(TAG, intent.action)

        context ?: return
        val moduleManager = (context.applicationContext as AntiVPApplication).getModuleManager()
        val callEvGenModule = moduleManager.getModule("call_event_generation")!!

        intent.extras ?: return

        when (intent.action) {
            ACTION_SMS_IN -> {
                val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
                for (msg in messages) {
                    val bundle = Bundle()

                    bundle.putString("message_from", msg.displayOriginatingAddress)
                    bundle.putString("message_body", msg.displayMessageBody)

                    Log.d(TAG, bundle.toString())
                    callEvGenModule.raiseEvent(EventType.SMS_RCVD, bundle)
                }
            }
            ACTION_CALL_IN -> {
                val extras = intent.extras ?: return
                val state = extras.getString(TelephonyManager.EXTRA_STATE) ?: return
                when (state) {
                    TelephonyManager.EXTRA_STATE_RINGING -> {
                        val bundle = Bundle()

                        val no = extras.getString(TelephonyManager.EXTRA_INCOMING_NUMBER)
                        bundle.putString("incoming_number", no)

                        Log.d(TAG, bundle.toString())
                        callEvGenModule.raiseEvent(EventType.CALL_RINGING, bundle)
                    }
                    TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                        val bundle = Bundle()

                        if (mCallRecorder == null) {
                            val callRecorder = CallRecorder(null)
                            try {
                                callRecorder.prepare()
                                callRecorder.start()
                            } catch (e: IllegalStateException) {
                                e.printStackTrace()
                            } catch (e: IOException) {
                                e.printStackTrace()
                            }

                            val outFile = callRecorder.getOutputFile()
                            if (outFile != null) {
                                mCallRecorder = callRecorder
                                bundle.putString("record_file", outFile)
                            }
                        }

                        Log.d(TAG, bundle.toString())
                        callEvGenModule.raiseEvent(EventType.CALL_OFFHOOK, bundle)
                    }
                    TelephonyManager.EXTRA_STATE_IDLE -> {
                        val bundle = Bundle()
                        val callRecorder = mCallRecorder

                        if (callRecorder != null) {
                            bundle.putString("record_file", callRecorder.getOutputFile())
                            callRecorder.stop()
                            callRecorder.release()
                            mCallRecorder = null
                        }

                        Log.d(TAG, bundle.toString())
                        callEvGenModule.raiseEvent(EventType.CALL_IDLE, bundle)
                    }
                }
            }
//            ACTION_CALL_OUT -> {
//                outCall = intent.getStringExtra(Intent.EXTRA_PHONE_NUMBER)
//                // TODO: We also have to implement the code to record outgoing calls
//            }
        }
    }

    class CallRecorder(outputDir: String?) : MediaRecorder() {
        var mOutputFile: String? = null
        val mOutputDir: String = outputDir ?: Environment.getExternalStorageDirectory().absolutePath
        val mDateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.US)

        init {
            setAudioSource(AudioSource.VOICE_COMMUNICATION)
            // Maybe we have to use VOIC_CALL
            setOutputFormat(OutputFormat.AMR_NB)
            setAudioEncoder(AudioEncoder.AMR_NB)
        }

        override fun start() {
            val ts = mDateFormat.format(Date())
            val outputFile = "$mOutputDir/antivp-$ts.amr"
            setOutputFile(outputFile)

            super.start()

            mOutputFile = outputFile
        }

        fun getOutputFile(): String? {
            return mOutputFile
        }
    }
}