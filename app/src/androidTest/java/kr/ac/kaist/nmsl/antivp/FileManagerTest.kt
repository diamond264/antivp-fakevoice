package kr.ac.kaist.nmsl.antivp;

import android.content.Context
import android.os.Bundle
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import junit.framework.TestCase.assertEquals

import org.junit.Test;
import org.junit.runner.RunWith;

import kr.ac.kaist.nmsl.antivp.core.util.FileManager;
import org.junit.Assert.assertArrayEquals

@RunWith(AndroidJUnit4::class)
class FileManagerTest {

    @Test
    fun testFileManager() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        /**
         * IMPORTANT: When you use FileManager, you don't have to initialize it
         * Initializing code is called in the main application code and the Singleton object initialized
         * Once it is initialized, you can just access it through FileManager.getInstance()
         */
        FileManager.initialize(context)

        /* From here you can replicate the code */
        val fileManager = FileManager.getInstance()

        // Save a string to a file
        val strFileName = "test_string.txt"
        val strData = "This is a test string."
        fileManager.save(strFileName, strData)
        assertEquals(true, fileManager.fileExists(strFileName))

        // Load the string from the file and check if it matches the original data
        val loadedStrData = fileManager.load(strFileName) as String?
        assertEquals(strData, loadedStrData)

        // Create a directory and check if it exists
        val directoryName = "test_directory"
        fileManager.createDirectory(directoryName)
        assertEquals(true, fileManager.directoryExists(directoryName))

        // Save a Bundle to a file
        val bundleFileName = "test_bundle.bin"
        val bundleData = Bundle().apply {
            putInt("count", 5)
            putString("message", "Hello, world!")
            putBoolean("flag", true)
        }
        fileManager.save("$directoryName/$bundleFileName", bundleData)
        assertEquals(true, fileManager.fileExists("$directoryName/$bundleFileName"))

        // Load the Bundle from the file and check if it matches the original data
        val loadedBundleData = fileManager.load("$directoryName/$bundleFileName") as Bundle?
        assertEquals(bundleData.getInt("count"), loadedBundleData?.getInt("count"))
        assertEquals(bundleData.getString("message"), loadedBundleData?.getString("message"))
        assertEquals(bundleData.getBoolean("flag"), loadedBundleData?.getBoolean("flag"))

        // Delete the Bundle and check whether it is deleted
        fileManager.deleteFile("$directoryName/$bundleFileName")
        assertEquals(false, fileManager.fileExists("$directoryName/$bundleFileName"))

        // Delete the directory and check whether it is deleted
        fileManager.removeDirectory(directoryName)
        assertEquals(false, fileManager.directoryExists(directoryName))
    }
}