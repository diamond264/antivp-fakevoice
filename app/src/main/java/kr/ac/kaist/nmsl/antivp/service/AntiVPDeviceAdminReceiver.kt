package kr.ac.kaist.nmsl.antivp.service

import android.app.admin.DeviceAdminReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kr.ac.kaist.nmsl.antivp.modules.call_event_generation.CallBroadcastReceiver

class AntiVPDeviceAdminReceiver : DeviceAdminReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        Log.d("ADMIN RECEIVER", "onReceive()")
        super.onReceive(context, intent)
    }
    override fun onEnabled(context: Context, intent: Intent) {
        Log.d("ADMIN RECEIVER", "onEnabled()")
        CallBroadcastReceiver().registerSelf(context.applicationContext)
    }
    override fun onDisabled(context: Context, intent: Intent) {
        Log.d("ADMIN RECEIVER", "onDisabled()")
    }
}