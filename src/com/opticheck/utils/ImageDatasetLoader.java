package com.opticheck.utils;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class ImageDatasetLoader {

    private static final int WIDTH = 64;
    private static final int HEIGHT = 64;

    public static ArrayDataset loadDataset(String baseDir, NDManager manager) throws IOException {
        List<float[]> dataList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        File acceptedDir = new File(baseDir, "ok");
        File defectDir = new File(baseDir, "notok");

        loadImagesFromDir(acceptedDir, 0, dataList, labelList);
        loadImagesFromDir(defectDir, 1, dataList, labelList);

        float[][] data = dataList.toArray(new float[0][]);
        int[] labels = labelList.stream().mapToInt(i -> i).toArray();

        NDArray X = manager.create(data);
        NDArray y = manager.create(labels);

        return new ArrayDataset.Builder()
                .setData(X)
                .optLabels(y)
                .setSampling(16, true)
                .build();
    }

    private static void loadImagesFromDir(File dir, int label,
                                          List<float[]> dataList, List<Integer> labelList) throws IOException {
        if (!dir.exists()) return;
        for (File file : Objects.requireNonNull(dir.listFiles((d, n) -> n.endsWith(".jpg") || n.endsWith(".png")))) {
            BufferedImage img = ImageIO.read(file);
            BufferedImage resized = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
            resized.getGraphics().drawImage(img, 0, 0, WIDTH, HEIGHT, null);

            float[] pixels = new float[WIDTH * HEIGHT * 3];
            int idx = 0;
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int rgb = resized.getRGB(x, y);
                    pixels[idx++] = ((rgb >> 16) & 0xFF) / 255f;
                    pixels[idx++] = ((rgb >> 8) & 0xFF) / 255f;
                    pixels[idx++] = (rgb & 0xFF) / 255f;
                }
            }
            dataList.add(pixels);
            labelList.add(label);
        }
    }
}
