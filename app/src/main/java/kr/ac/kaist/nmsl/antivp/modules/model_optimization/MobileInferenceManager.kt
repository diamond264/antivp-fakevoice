package kr.ac.kaist.nmsl.antivp.modules.model_optimization

import android.os.Bundle
import android.os.Environment
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

class MobileInferenceManager {
    private val TAG = "MobileInference"
    val storageDir: String = Environment.getExternalStorageDirectory().absolutePath

    enum class FrameworkType(val value: Int) {
        Pytorch(0),
        Tensorflow(1);

        companion object {
            fun fromInt(value: Int) = FrameworkType.values().first { it.value == value }
        }
    }

    fun performPytorchInference(frameworkType: Int, modelType: Int,
                                modelPath: String, inputData: Bundle): Bundle {
        val modelFile = "$storageDir/$modelPath"
        var bundle = Bundle()

        when (FrameworkType.fromInt(frameworkType)) {
            FrameworkType.Pytorch -> {
                Log.d(TAG, "Perform Pytorch Inference")
                when (ModelType.fromInt(modelType)) {
                    // Base implementation for mobile inference
                    ModelType.TEST_BERT_MODEL -> {
                        // 1. Loading PyTorch (torchscript) module file
                        Log.d(TAG, "module path: "+modelFile)
                        val module: Module = Module.load(modelFile)
                        Log.d(TAG, "module loaded: "+module)

                        // 2. Generating IValue(Tensor) objects from Bundle
                        val input_ids_array = inputData.getIntArray("input_ids")
                        val input_ids_shape = arrayOf<Long>(1, 512).toLongArray()
                        val input_ids_tensor = IValue.from(Tensor.fromBlob(input_ids_array, input_ids_shape))
                        val attention_mask_array = inputData.getIntArray("attention_mask")
                        val attention_mask_shape = arrayOf<Long>(1, 512).toLongArray()
                        val attention_mask_tensor = IValue.from(Tensor.fromBlob(attention_mask_array, attention_mask_shape))

                        // 3. Forwarding IValues to the loaded model
                        val outTensor = module.forward(input_ids_tensor, attention_mask_tensor).toTensor();
                        val out = outTensor.dataAsFloatArray

                        bundle.putFloatArray("out", out)
                        return bundle
                    }
                    ModelType.TEST_S2T_MODEL -> {
                        // 1. Loading PyTorch (torchscript) module file
                        Log.d(TAG, "module path: "+modelFile)
                        val module: Module = Module.load(modelFile)
                        Log.d(TAG, "module loaded: "+module)

                        // 2. Generating IValue(Tensor) objects from Bundle
                        val input_features_array = inputData.getFloatArray("input_features")
                        val input_features_shape = arrayOf<Long>(1, 298, 80).toLongArray()
                        val input_features_tensor = IValue.from(Tensor.fromBlob(input_features_array, input_features_shape))
                        val attention_mask_array = inputData.getIntArray("attention_mask")
                        val attention_mask_shape = arrayOf<Long>(1, 298).toLongArray()
                        val attention_mask_tensor = IValue.from(Tensor.fromBlob(attention_mask_array, attention_mask_shape))

                        // 3. Forwarding IValues to the loaded model
                        val outTensor = module.forward(input_features_tensor, attention_mask_tensor).toTensor();
                        val out = outTensor.dataAsLongArray

                        bundle.putLongArray("out", out)
                        return bundle
                    }
                    else -> {}
                }
            }
            FrameworkType.Tensorflow -> {
                Log.d(TAG, "Perform Tensorflow inference")
            }
            else -> {
                Log.d(TAG, "Unknown type of ML model")
            }
        }
        return bundle
    }
}