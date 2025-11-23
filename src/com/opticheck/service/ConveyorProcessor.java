package com.opticheck.service;

import ai.djl.translate.TranslateException;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ConveyorProcessor {

    private final PatternClassifierService classifier;

    public ConveyorProcessor(PatternClassifierService classifier) {
        this.classifier = classifier;
    }

    public void startProcessing() throws IOException, TranslateException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        VideoCapture camera = new VideoCapture(0); // use camera index 0

        if (!camera.isOpened()) {
            throw new IOException("Camera not found!");
        }

        Mat frame = new Mat();
        while (camera.read(frame)) {
            processFrame(frame);
            if (frame.empty()) break;
        }

        camera.release();
    }

    private void processFrame(Mat frame) throws TranslateException {
        // Convert to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        // Threshold to isolate boards
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 100, 255, Imgproc.THRESH_BINARY_INV);

        // Find contours (boards)
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);

            // Skip tiny noise
            if (rect.width < 50 || rect.height < 50) continue;

            Mat board = new Mat(frame, rect);
            Imgproc.resize(board, board, new Size(64, 64));

            // Flatten to 1D float array
            float[] flattened = new float[(int) (board.total() * board.channels())];
            board.get(0, 0, flattened);

            // Predict with classifier
            int result = classifier.predict(flattened);
            String label = result == 0 ? "ACCEPTED" : "DEFECTIVE";

            // Draw bounding box and label
            Scalar color = result == 0 ? new Scalar(0, 255, 0) : new Scalar(0, 0, 255);
            Imgproc.rectangle(frame, rect, color, 2);
            Imgproc.putText(frame, label, new Point(rect.x, rect.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        // Optionally save annotated frame
        Imgcodecs.imwrite("output/frame_" + System.currentTimeMillis() + ".jpg", frame);
    }
}

