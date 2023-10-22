package kr.ac.kaist.nmsl.antivp.core

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.util.Log
import kotlin.collections.ArrayList

class ModuleManager {

    var mModuleDict = mutableMapOf<String, Module>()
    var mEventSubscritpionMap = mutableMapOf<EventType, ArrayList<Pair<String?, Module>>>()
    var mContext: Context

    constructor(ctx: Context) {
        mContext = ctx
    }

     class ModuleHandler(module: Module) : Handler() {
        var mModule = module

         override fun handleMessage(msg: Message) {
            super.handleMessage(msg)
            val type = EventType.valueOf(msg.arg1) ?: return

            mModule.handleEvent(type, msg.data)
        }
    }

    private fun sendMessage(target: String, type: EventType, bundle: Bundle) {
        val targetModule = getModule(target)
        if (targetModule == null) {
            Log.e("sendMessage", "Cannot find $target. Available modules: " + moduleNames())
            return
        }

        sendMessage(targetModule, type, bundle)
    }

    private fun sendMessage(target: Module, type: EventType, bundle: Bundle) {
        val handler = target.handler()

        val msg = Message()
        msg.arg1 = type.value
        msg.data = bundle

        handler.sendMessage(msg)
    }

    fun raiseEvent(module: Module, type: EventType, bundle: Bundle) {
        val list = mEventSubscritpionMap[type] ?: return

        for (pair in list) {
            val target = pair.first
            val consumer = pair.second

            if (target != null && target != module.name())
                continue

            sendMessage(consumer, type, bundle)
        }
    }

    fun subscribeEvent(module: Module, target: String?, type: EventType) {
        if (!mEventSubscritpionMap.contains(type))
            mEventSubscritpionMap.put(type, ArrayList())

        val list = mEventSubscritpionMap.get(type)!!

        if (list.contains(Pair(target, module))) {
            Log.w(null, module.name() + " has already subscribed " + type.name + " from " + target)
            return
        }

        list.add(Pair(target, module))
    }

    fun register(module: Module) {
        module.setManager(this)
        mModuleDict.put(module.name(), module)
    }

    fun getModule(moduleName: String): Module? {
        return mModuleDict[moduleName]
    }

    fun moduleNames() : ArrayList<String> {
        val arr = ArrayList<String>()
        for (entry in mModuleDict.entries)
            arr.add(entry.key)
        return arr
    }

    fun setApplicationContext(applicationContext: Context) {
        for (entry in mModuleDict.entries)
            entry.value.applicationContext = applicationContext
    }
}