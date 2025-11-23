package com.opticheck.service;


import ai.djl.translate.TranslateException;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.util.concurrent.atomic.AtomicBoolean;

public class WebcamFrameProcessor {

    static {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();
    }

    private final PatternClassifierService classifierService;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final int width = 64;
    private final int height = 64;

    public WebcamFrameProcessor(PatternClassifierService classifierService) {
        this.classifierService = classifierService;
    }

    public void start() {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println("❌ Cannot open camera!");
            return;
        }

        running.set(true);
        Mat frame = new Mat();

        System.out.println("🎥 Webcam stream started...");
        while (running.get()) {
            if (camera.read(frame)) {
                int prediction = processFrame(frame);
                String label = (prediction == 0) ? "ACCEPTED" : "DEFECT";
                System.out.println("Frame classified as: " + label);
            } else {
                System.err.println("⚠️ Frame not captured!");
            }
        }

        camera.release();
        System.out.println("🛑 Webcam stream stopped.");
    }

    public void stop() {
        running.set(false);
    }

    private int processFrame(Mat frame) {
        // Resize to match model input
        Mat resized = new Mat();
        Imgproc.resize(frame, resized, new Size(width, height));

        // Convert to RGB if needed
        Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);

        // Flatten to 1D float array
        int channels = 3;
        int totalSize = width * height * channels;
        float[] input = new float[totalSize];
        resized.get(0, 0, input);

        // Normalize pixel values (0–1)
        for (int i = 0; i < input.length; i++) {
            input[i] /= 255.0f;
        }

        try {
            return classifierService.predict(input);
        } catch (TranslateException e) {
            e.printStackTrace();
            return -1;
        }
    }
}
