package com.opticheck.service;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ImagePreprocessor {

    public static BufferedImage resize(BufferedImage original, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    public static float[] toFloatArray(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        float[] data = new float[width * height * 3]; // RGB

        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                data[idx++] = ((rgb >> 16) & 0xFF) / 255f; // R
                data[idx++] = ((rgb >> 8) & 0xFF) / 255f;  // G
                data[idx++] = (rgb & 0xFF) / 255f;         // B
            }
        }
        return data;
    }
}
