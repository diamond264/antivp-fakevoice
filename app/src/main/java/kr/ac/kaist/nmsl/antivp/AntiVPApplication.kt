package kr.ac.kaist.nmsl.antivp

import android.app.Application
import kr.ac.kaist.nmsl.antivp.core.ModuleManager
import kr.ac.kaist.nmsl.antivp.core.util.FileManager
import kr.ac.kaist.nmsl.antivp.modules.call_event_generation.CallEventGenerationModule
import kr.ac.kaist.nmsl.antivp.modules.fake_voice_detection.FakeVoiceDetectionModule
import kr.ac.kaist.nmsl.antivp.modules.mal_activity_detection.MalActivityDetectionModule
import kr.ac.kaist.nmsl.antivp.modules.mal_app_detection.MalAppDetectionModule
import kr.ac.kaist.nmsl.antivp.modules.model_optimization.ModelOptimizationModule
import kr.ac.kaist.nmsl.antivp.modules.net_based_detection.NetBasedDetectionModule
import kr.ac.kaist.nmsl.antivp.modules.phishing_event_receiver.PhishingEventReceiverModule
import kr.ac.kaist.nmsl.antivp.modules.speech_to_text.SpeechToTextModule
import kr.ac.kaist.nmsl.antivp.modules.text_based_detection.TextBasedDetectionModule

class AntiVPApplication: Application() {
    private val mModuleManager = ModuleManager(this)

    override fun onCreate() {
        super.onCreate()

        FileManager.initialize(this)

        mModuleManager.register(CallEventGenerationModule())
        mModuleManager.register(FakeVoiceDetectionModule())
        mModuleManager.register(MalActivityDetectionModule())
        mModuleManager.register(MalAppDetectionModule())
        mModuleManager.register(NetBasedDetectionModule())
        mModuleManager.register(PhishingEventReceiverModule())
        mModuleManager.register(SpeechToTextModule())
        mModuleManager.register(TextBasedDetectionModule())
        mModuleManager.register(ModelOptimizationModule())
    }

    fun getModuleManager(): ModuleManager {
        return mModuleManager
    }
}