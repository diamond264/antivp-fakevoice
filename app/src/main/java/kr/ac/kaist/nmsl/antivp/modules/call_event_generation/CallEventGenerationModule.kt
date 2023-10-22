package kr.ac.kaist.nmsl.antivp.modules.call_event_generation

import android.os.Bundle
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module

class CallEventGenerationModule() : Module(){
    override fun name(): String {
        return "call_event_generation"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
    }
}