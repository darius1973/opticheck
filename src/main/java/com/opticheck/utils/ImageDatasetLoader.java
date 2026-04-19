package com.opticheck.utils;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
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

    private static final int WIDTH = 224;
    private static final int HEIGHT = 224;

    // =========================
    // MAIN LOADER
    // =========================
    public static ArrayDataset loadDataset(String baseDir, NDManager manager) throws IOException {

        List<NDArray> dataList = new ArrayList<>();
        List<NDArray> labelList = new ArrayList<>();

        File okDir = new File(baseDir, "ok");
        File nokDir = new File(baseDir, "notok");

        loadImagesFromDir(okDir, 0f, manager, dataList, labelList);
        loadImagesFromDir(nokDir, 1f, manager, dataList, labelList);

        NDArray X = NDArrays.stack(new NDList(dataList));
        NDArray y = NDArrays.stack(new NDList(labelList));

        return new ArrayDataset.Builder()
                .setData(X)
                .optLabels(y)
                .setSampling(32, true)
                .build();
    }

    // =========================
    // DIRECTORY LOADER
    // =========================
    private static void loadImagesFromDir(
            File dir,
            float label,
            NDManager manager,
            List<NDArray> dataList,
            List<NDArray> labelList) throws IOException {

        File[] files = dir.listFiles();
        if (files == null) return;
        System.out.println("count " + label + " " + files.length);
        for (File file : files) {

            if (!file.getName().endsWith(".jpg") &&
                    !file.getName().endsWith(".jpeg") &&
                    !file.getName().endsWith(".png")) {
                continue;
            }

            NDArray image = loadImageAsNDArray(file, manager);

            dataList.add(image);
            labelList.add(manager.create(new float[]{label}));
        }
    }

    // =========================
    // IMAGE → NDARRAY
    // =========================
    public static NDArray loadImageAsNDArray(File file, NDManager manager) throws IOException {

        BufferedImage img = ImageIO.read(file);

        // Resize
        BufferedImage resized = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, WIDTH, HEIGHT, null);
        g.dispose();

        float[] data = new float[3 * WIDTH * HEIGHT];

        int idx = 0;

        // CHW format (IMPORTANT for DJL + CNNs)
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {

                    int rgb = resized.getRGB(x, y);

                    float value;
                    switch (c) {
                        case 0 -> value = ((rgb >> 16) & 0xFF);
                        case 1 -> value = ((rgb >> 8) & 0xFF);
                        default -> value = (rgb & 0xFF);
                    }

                    // Normalize for pretrained models (important for DenseNet)
                    value = (value / 255f);

                    data[idx++] = value;
                }
            }
        }

        return manager.create(data).reshape(3, HEIGHT, WIDTH);
    }
}