package kr.ac.kaist.nmsl.antivp.ui

import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import kr.ac.kaist.nmsl.antivp.AntiVPApplication
import kr.ac.kaist.nmsl.antivp.R
import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module
import kr.ac.kaist.nmsl.antivp.modules.model_optimization.ModelType
import kr.ac.kaist.nmsl.antivp.service.beAdminAndStartCallTracker


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        beAdminAndStartCallTracker(this)

//        val moduleManager = (application as AntiVPApplication).getModuleManager()
//        val dummyModule = object: Module() {
//            override fun name(): String { return "dummy_module" }
//            override fun handleEvent(type: EventType, bundle: Bundle) {}
//        }
//        val btn = findViewById<Button>(R.id.get_sys_param);
//        btn.setOnClickListener {
//            val bundle = Bundle()
//            bundle.putInt("model_type", ModelType.TEST_MODEL.value)
//            moduleManager.raiseEvent(dummyModule, EventType.GET_OPTIMAL_MODEL, bundle)
//        }
    }
}