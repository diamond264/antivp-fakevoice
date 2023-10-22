package kr.ac.kaist.nmsl.antivp.modules.text_based_detection

import android.os.Bundle
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.*
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module
import kr.ac.kaist.nmsl.antivp.core.ModuleManager
import org.junit.runner.RunWith
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
internal class TextBasedDetectionModuleTest {
    @Test
    fun testModule() {
        val latch = CountDownLatch(1)

        MainScope().launch {
            val mModuleManager = ModuleManager(ApplicationProvider.getApplicationContext())

            val textBasedDetectionModule = TextBasedDetectionModule()

            val dummySpeechToTextModule = object: Module() {
                override fun name(): String { return "speech_to_text" }
                override fun handleEvent(type: EventType, bundle: Bundle) {}
            }
            val dummyPhishingEventReceiverModule = object: Module() {
                init { subscribeEvent(EventType.PHISHING_CALL_DETECTED) }
                override fun name(): String { return "phishing_event_receiver" }
                override fun handleEvent(type: EventType, bundle: Bundle) {
                    when (type) {
                        EventType.PHISHING_CALL_DETECTED -> {
                            latch.countDown()
                        }
                        else -> {
                            throw Exception()
                        }
                    }
                }
            }

            mModuleManager.register(dummySpeechToTextModule)
            mModuleManager.register(textBasedDetectionModule)
            mModuleManager.register(dummyPhishingEventReceiverModule)

            val bundle = Bundle()
            bundle.putInt("dialogue_id", 0)
            bundle.putStringArray("dialogue", arrayOf(
                "Hello",
                "Hi",
                "How are you?"
            ))
            dummySpeechToTextModule.raiseEvent(EventType.TEXT_TRANSCRIBED, bundle)
        }
        latch.await(10000, TimeUnit.MILLISECONDS)
    }
}