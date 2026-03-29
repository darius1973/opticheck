package com.opticheck.utils;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class ImageDatasetLoader {

    private static final int WIDTH = 64;
    private static final int HEIGHT = 64;

    public static ArrayDataset loadDataset(String baseDir, NDManager manager) throws IOException {

        List<float[][][]> dataList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        File okDir = new File(baseDir, "ok");
        File nokDir = new File(baseDir, "notok");

        loadImagesFromDir(okDir, 0, dataList, labelList);
        loadImagesFromDir(nokDir, 1, dataList, labelList);

        // Convert list → 4D array
        float[][][][] data = dataList.toArray(new float[0][][][]);
        int[] labels = labelList.stream().mapToInt(i -> i).toArray();

        int N = dataList.size();

        float[] flat = new float[N * 3 * 64 * 64];

        int idx = 0;

        for (float[][][] img : dataList) {
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < 64; y++) {
                    for (int x = 0; x < 64; x++) {
                        flat[idx++] = img[c][y][x];
                    }
                }
            }
        }

        // Create NDArray and reshape to CNN format
        NDArray X = manager.create(flat)
                .reshape(N, 3, 64, 64);

        NDArray y = manager.create(labels); // shape (N)

        return new ArrayDataset.Builder()
                .setData(X)
                .optLabels(y)
                .setSampling(32,
true)
                .build();
    }

    private static void loadImagesFromDir(File dir,
                                          int label,
                                          List<float[][][]> dataList,
                                          List<Integer> labelList) throws IOException {

        for (File file : dir.listFiles()) {

            if (!file.getName().endsWith(".jpg") && !file.getName().endsWith(".png"))
                continue;

            float[][][] image = loadImageAsTensor(file); // 👈 IMPORTANT

            dataList.add(image);
            labelList.add(label);
        }
    }

    public static float[][][] loadImageAsTensor(File file) throws IOException {

        BufferedImage img = ImageIO.read(file);

        // Resize to 64x64
        BufferedImage resized = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, 64, 64, null);
        g.dispose();

        // [channels][height][width]
        float[][][] data = new float[3][64][64];

        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {

                int rgb = resized.getRGB(x, y);

                float r = (((rgb >> 16) & 0xFF) / 255f - 0.5f) / 0.5f;
                float gC = (((rgb >> 8) & 0xFF) / 255f - 0.5f) / 0.5f;
                float b = ((rgb & 0xFF) / 255f - 0.5f) / 0.5f;
                /*
                // Extract RGB channels and normalize to [0,1]
                float r = ((rgb >> 16) & 0xFF) / 255f;
                float gC = ((rgb >> 8) & 0xFF) / 255f;
                float b = (rgb & 0xFF) / 255f;*/

                data[0][y][x] = r;   // Red channel
                data[1][y][x] = gC;  // Green channel
                data[2][y][x] = b;   // Blue channel
            }
        }

        return data;
    }
}
