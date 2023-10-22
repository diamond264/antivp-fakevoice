package kr.ac.kaist.nmsl.antivp.service

import android.app.Activity
import android.app.admin.DevicePolicyManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.util.EventLog.Event
import kr.ac.kaist.nmsl.antivp.AntiVPApplication
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.ModuleManager
import kr.ac.kaist.nmsl.antivp.modules.call_event_generation.CallBroadcastReceiver

fun  beAdminAndStartCallTracker(activity: Activity) {
    try {
        val appctx = activity.applicationContext
        val mDPM = activity.getSystemService(Context.DEVICE_POLICY_SERVICE) as DevicePolicyManager
        val mAdminName = ComponentName(appctx, AntiVPDeviceAdminReceiver::class.java)
        if (!mDPM.isAdminActive(mAdminName)) {
            val intent = Intent(DevicePolicyManager.ACTION_ADD_DEVICE_ADMIN)
            intent.putExtra(DevicePolicyManager.EXTRA_DEVICE_ADMIN, mAdminName)
            intent.putExtra(
                DevicePolicyManager.EXTRA_ADD_EXPLANATION,
                "Click on Activate button to secure your application.")
            activity.startActivity(intent);
        } else {
            CallBroadcastReceiver().registerSelf(appctx)
            val moduleManager = (activity.application as AntiVPApplication).getModuleManager()
            moduleManager.setApplicationContext(appctx)
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
}
