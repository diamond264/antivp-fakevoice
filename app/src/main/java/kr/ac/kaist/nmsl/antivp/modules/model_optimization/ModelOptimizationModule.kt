package kr.ac.kaist.nmsl.antivp.modules.model_optimization

import android.app.DownloadManager
import android.content.Context
import android.content.Context.HARDWARE_PROPERTIES_SERVICE
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.HardwarePropertiesManager
import android.util.Log
import androidx.annotation.RequiresApi

import kr.ac.kaist.nmsl.antivp.core.EventType
import kr.ac.kaist.nmsl.antivp.core.Module
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.io.RandomAccessFile

import com.android.volley.Request
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.Volley
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONObject

class ModelOptimizationModule : Module() {
    private val TAG = "ModelOptimization"
    private var memoryUsage = 0.0
    private var cpuUsage = 0.0
    private var temp = 0.0

    init {
        subscribeEvent(EventType.GET_OPTIMAL_MODEL)
    }

    override fun name(): String {
        return "model_optimization"
    }

    override fun handleEvent(type: EventType, bundle: Bundle) {
        when (type) {
            EventType.GET_OPTIMAL_MODEL -> {
                applicationContext ?: return
                var modelName: String? = null

                when (ModelType.fromInt(bundle.getInt("model_type"))) {
                    ModelType.SPEECH_TO_TEXT -> {
                        Log.d(TAG, "Rcvd a request to get optimal speech to text model.")
                        modelName = "stt_model"
                    }
                    ModelType.PHISHING_DETECTION -> {
                        Log.d(TAG, "Rcvd a request to get optimal phishing detection model.")
                        modelName = "vpdetect_model"
                    }
                    else -> {
                        Log.d(TAG, "Unknown type of model")
                    }
                }

                /* get system parameters */
                // memory usage
                memoryUsage = readMemInfo()
                Log.d(TAG, "memoryUsage: $memoryUsage")

                // read CPU usage (require root permission)
                cpuUsage = readStat()
                Log.d(TAG, "cpuUsage: $cpuUsage")

                // read temperature (require root permission)
                temp = readTemp()
                Log.d(TAG, "temp: $temp")

                val modelInfoUrl = "http://iu.kaist.ac.kr:5000/$modelName/info"
                val queue = Volley.newRequestQueue(applicationContext)

                val request = JsonObjectRequest(Request.Method.GET, modelInfoUrl, null,
                    { response ->
                        Log.d(TAG, response.toString())
                        modelName?.let { retrieveModel(response, it) }
                    },
                    { error -> Log.e(TAG, error.toString()) }
                )

                queue.add(request)
            }
            else -> {
                Log.e(TAG, "Unexpected event type: $type")
            }
        }
    }

    private fun readMemInfo(): Double {
        Log.d(TAG, "readMemInfo")
        lateinit var reader: RandomAccessFile
        var total = 0.0
        var available = 0.0
        var availableRatio = 0.0
        try {
            reader = RandomAccessFile("/proc/meminfo", "r")
        } catch (ex: IOException) {
            ex.printStackTrace()
        }
        var line = reader.readLine()
        while (line != null) {
//            Log.d(TAG, line)
            if (line.startsWith("MemTotal:")) {
                total = line.split("\\s+".toRegex())[1].toDouble()
            } else if (line.startsWith("MemAvailable:")) {
                available = line.split("\\s+".toRegex())[1].toDouble()
            }
            if (total != 0.0 && available != 0.0) {
                availableRatio = available / total
                break
            }
            line = reader.readLine()

//            Log.d(TAG, "Total: $total")
//            Log.d(TAG, "Available: $available")
        }
        reader.close()

        return 1.0 - availableRatio
    }

    // "/proc/stat" permission denied for third party apps
    private fun readStat(): Double {
        Log.d(TAG, "readStat")
        try {
            val cmdline = arrayOf("su", "sh", "-c", "cat /proc/stat")
            val readStat = Runtime.getRuntime().exec(cmdline)
            val reader = BufferedReader(InputStreamReader(readStat.inputStream))
            val buffer = CharArray(4096)
            val output = StringBuffer()
            readStat.waitFor()

            var read = reader.read(buffer)
            output.append(buffer, 0, read)

            val cpuStats = output.toString().split("\\s+".toRegex())
            var totalCpuUsage = 0.0
            for (i in 1..7) {
                totalCpuUsage += cpuStats[i].toDouble()
            }
            val idle = cpuStats[4].toDouble()

//            while (read > 0) {
//                output.append(buffer, 0, read)
//                Log.d(TAG, output.toString())
//                read = reader.read(buffer)
//            }
            reader.close()
            return 1.0 - (idle/totalCpuUsage)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return 0.0
    }

    private fun readTemp(): Double {
        Log.d(TAG, "readTemp")
        try {
            val cmdline = arrayOf("su", "sh", "-c", "cat /sys/class/thermal/tz-by-name/battery/temp")
            val readStat = Runtime.getRuntime().exec(cmdline)
            val reader = BufferedReader(InputStreamReader(readStat.inputStream))
            val buffer = CharArray(4096)
            val output = StringBuffer()
            readStat.waitFor()

            var read = reader.read(buffer)
            while (read > 0) {
                output.append(buffer, 0, read)
//                Log.d(TAG, output.toString())
                read = reader.read(buffer)
            }

            reader.close()

            return output.toString().toDouble() / 1000
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return 0.0
    }

    // requires this app to be a system app
    @RequiresApi(Build.VERSION_CODES.N)
    private fun getHardwareStats() {
        Log.d(TAG, "getHardwareStats")
        val hardwareManager = getContext().getSystemService(HARDWARE_PROPERTIES_SERVICE) as HardwarePropertiesManager
        val cpuUsageInfos = hardwareManager.cpuUsages
        Log.d(TAG, cpuUsageInfos.toString())
    }

    private fun retrieveModel(response: JSONObject, modelName: String) {
        val downloadManager = applicationContext!!.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

        val modelDownloadUrl = "http://iu.kaist.ac.kr:5000/$modelName/0"
        val downloadRequest = DownloadManager.Request(Uri.parse(modelDownloadUrl))
            .setAllowedNetworkTypes(DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE)
            .setTitle("AntiVP")
            .setDescription("Downloading Model: $modelName")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setAllowedOverMetered(true)
            .setDestinationInExternalFilesDir(applicationContext, Environment.DIRECTORY_DOWNLOADS, modelName)
        val downloadID = downloadManager.enqueue(downloadRequest)

        CoroutineScope(Dispatchers.IO).launch {
            while (getDownloadStatus(downloadManager, downloadID) != DownloadManager.STATUS_SUCCESSFUL) {
                delay(100)
            }

            Log.d(TAG, "Download complete: ${downloadManager.getUriForDownloadedFile(downloadID)}")
            // Log.d(TAG, "${applicationContext!!.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)?.listFiles()?.map { it.name }}")
        }
    }

    private fun getDownloadStatus(downloadManager: DownloadManager, id: Long): Int {
        val query = DownloadManager.Query()
        query.setFilterById(id)
        val cursor = downloadManager.query(query)
        if (cursor.moveToFirst()) {
            val status = cursor.getInt(cursor.getColumnIndex(DownloadManager.COLUMN_STATUS))
            cursor.close()
            return status
        }
        return -1
    }

}