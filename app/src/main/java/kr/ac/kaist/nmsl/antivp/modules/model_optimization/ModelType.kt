package kr.ac.kaist.nmsl.antivp.modules.model_optimization

enum class ModelType(val value: Int) {
    TEST_BERT_MODEL(-1),
    TEST_S2T_MODEL(-2),
    SPEECH_TO_TEXT(0),
    PHISHING_DETECTION(1);

    companion object {
        fun fromInt(value: Int) = ModelType.values().first { it.value == value }
    }
}