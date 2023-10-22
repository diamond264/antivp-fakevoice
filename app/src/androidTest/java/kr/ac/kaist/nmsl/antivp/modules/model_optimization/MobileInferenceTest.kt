package kr.ac.kaist.nmsl.antivp.modules.model_optimization

import android.os.Bundle
import android.util.Log
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
internal class MobileInferenceTest {
    @Test
    fun testModule() {
        val latch = CountDownLatch(2)

        MainScope().launch {
            val mModuleManager = ModuleManager(ApplicationProvider.getApplicationContext())

            // dummy module to raise a fake event
            val dummyModule = object: Module() {
                override fun name(): String { return "dummy_module" }
                override fun handleEvent(type: EventType, bundle: Bundle) {}
            }

            // module for performing inference
            val sampleInferenceModule = object: Module() {
                init {
                    subscribeEvent(EventType.TEST_BERT_DATA_BROADCASTED)
                    subscribeEvent(EventType.TEST_S2T_DATA_BROADCASTED)
                }
                override fun name(): String { return "sample_inference_module" }
                override fun handleEvent(type: EventType, bundle: Bundle) {
                    Log.d(name(), "sampleInferenceModule got an event")
                    Log.d(name(), ""+type)
                    when (type) {
                        EventType.TEST_BERT_DATA_BROADCASTED -> {
                            var modelName: String? = null

                            when (ModelType.fromInt(bundle.getInt("model_type"))) {
                                ModelType.TEST_BERT_MODEL -> {
                                    Log.d(name(), "Rcvd a request to get a test model.")
                                    modelName = "test_bert_model"

                                    // Allocate input data
                                    val input_ids = IntArray(512) { 1 }
                                    val attention_mask = IntArray(512) { 1 }
                                    /* Inputs should be given in a flatten form.
                                    Tensor.fromBlob automatically unflattens the data with the given shape
                                    ex) 1 | FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*224*224);
                                    ex) 2 | Tensor tensor = Tensor.fromBlob(buffer, new long[]{1, 3, 224, 224});
                                     */

                                    // Compose all data into a Bundle
                                    val inputData = Bundle()
                                    inputData.putIntArray("input_ids", input_ids)
                                    inputData.putIntArray("attention_mask", attention_mask)

                                    // Perform inference
                                    val mim = MobileInferenceManager()
                                    val outBundle = mim.performPytorchInference(
                                        MobileInferenceManager.FrameworkType.Pytorch.value,
                                        ModelType.TEST_BERT_MODEL.value,
                                        "models/test_model.pt",
                                        inputData
                                    )
                                    val out = outBundle.getFloatArray("out")

                                    if (out != null) {
                                        Log.d(modelName, "[computed] out val: " + out.get(0))
                                        latch.countDown()
                                    }
                                }
                                else -> {
                                    Log.d(name(), "Unknown type of model")
                                }
                            }
                        }
                        EventType.TEST_S2T_DATA_BROADCASTED -> {
                            var modelName: String? = null

                            when (ModelType.fromInt(bundle.getInt("model_type"))) {
                                ModelType.TEST_S2T_MODEL -> {
                                    Log.d(name(), "Rcvd a request to get a test model.")
                                    modelName = "test_s2t_model"

                                    // Allocate input data
                                    val input_features = FloatArray(1*298*80) { 1.0f }
                                    val attention_mask = IntArray(1*298) { 1 }
                                    /* Inputs should be given in a flatten form.
                                    Tensor.fromBlob automatically unflattens the data with the given shape
                                    ex) 1 | FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*224*224);
                                    ex) 2 | Tensor tensor = Tensor.fromBlob(buffer, new long[]{1, 3, 224, 224});
                                     */

                                    // Compose all data into a Bundle
                                    val inputData = Bundle()
                                    inputData.putFloatArray("input_features", input_features)
                                    inputData.putIntArray("attention_mask", attention_mask)

                                    // Perform inference
                                    val mim = MobileInferenceManager()
                                    val outBundle = mim.performPytorchInference(
                                        MobileInferenceManager.FrameworkType.Pytorch.value,
                                        ModelType.TEST_S2T_MODEL.value,
                                        "models/test_model2.pt",
                                        inputData
                                    )
                                    val out = outBundle.getLongArray("out")

                                    if (out != null) {
                                        Log.d(modelName, "[computed] out val: " + out.get(0))
                                        latch.countDown()
                                    }
                                }
                                else -> {
                                    Log.d(name(), "Unknown type of model")
                                }
                            }
                        }
                        else -> {}
                    }
                }
            }

            mModuleManager.register(dummyModule)
            mModuleManager.register(sampleInferenceModule)

            val bundle_bert = Bundle()
            bundle_bert.putInt("model_type", ModelType.TEST_BERT_MODEL.value)
            dummyModule.raiseEvent(EventType.TEST_BERT_DATA_BROADCASTED, bundle_bert)

            val bundle_s2t = Bundle()
            bundle_s2t.putInt("model_type", ModelType.TEST_S2T_MODEL.value)
            dummyModule.raiseEvent(EventType.TEST_S2T_DATA_BROADCASTED, bundle_s2t)
            Log.d("INIT", "Raised an event")
        }
        latch.await(30000, TimeUnit.MILLISECONDS)
    }
}