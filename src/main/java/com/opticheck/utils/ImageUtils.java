package com.opticheck.utils;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Base64;
import javax.imageio.ImageIO;

public class ImageUtils {

    /**
     * Converts a BufferedImage to a Base64 string suitable for Vaadin Image component.
     *
     * @param img the BufferedImage to convert
     * @return a String like "data:image/png;base64,...."
     * @throws IOException if writing the image fails
     */
    public static String bufferedImageToVaadinSrc(BufferedImage img) throws IOException {
        // Create a byte array output stream
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            // Write the image as PNG to the output stream
            ImageIO.write(img, "png", baos);
            baos.flush();

            // Encode byte array to Base64
            String base64 = Base64.getEncoder().encodeToString(baos.toByteArray());

            // Return as a data URL
            return "data:image/png;base64," + base64;
        }
    }

    public static boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return file.isFile() &&
                !name.startsWith(".") &&
                (name.endsWith(".jpg") ||
                        name.endsWith(".jpeg") ||
                        name.endsWith(".png"));
    }
}

