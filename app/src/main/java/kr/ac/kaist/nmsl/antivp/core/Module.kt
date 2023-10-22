package kr.ac.kaist.nmsl.antivp.core

import android.content.Context
import android.os.Bundle
import android.os.Handler

abstract class Module() {
    private val TAG = "Module"

    private val mHandler: Handler = ModuleManager.ModuleHandler(this)
    private var mManager: ModuleManager? = null
    private var mPendingSubscriptions: ArrayList<Pair<String?, EventType>> = ArrayList()
    var applicationContext: Context? = null

    abstract fun name(): String
    abstract fun handleEvent(type: EventType, bundle: Bundle)

    fun handler(): Handler {
        return mHandler
    }

    fun getContext(): Context {
        return mManager!!.mContext
    }

    fun setManager(manager: ModuleManager) {
        mManager = manager
        if (!mPendingSubscriptions.isEmpty())
            doPendingSubscriptions()
    }

    private fun doPendingSubscriptions() {
        for (pair in mPendingSubscriptions) run {
            val moduleName = pair.first
            val event = pair.second

            subscribeEvent(moduleName, event)
        }
    }

    fun subscribeEvent(module: String?, type: EventType) {
        if (mManager == null) {
            mPendingSubscriptions.add(Pair(module, type))
            return
        }

        mManager!!.subscribeEvent(this, module, type)
    }

    fun subscribeEvent(type: EventType) {
        return subscribeEvent(null, type)
    }

//    fun unsubscribeEvent(module: String?, type: EventType) {
//
//    }
//
//    fun unsubscribeEvent(type: EventType) {
//        return unsubscribeEvent(null, type)
//    }

    fun raiseEvent(type: EventType, bundle: Bundle) {
        mManager?: return

        mManager!!.raiseEvent(this, type, bundle)
    }
}