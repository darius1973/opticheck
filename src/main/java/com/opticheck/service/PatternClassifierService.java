package com.opticheck.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.TranslateException;
import com.opticheck.config.DJLConfigComponent;
import com.opticheck.trainer.CNNTrainer;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;


import com.opticheck.pojo.FilePrediction;

import javax.imageio.ImageIO;

import static com.opticheck.utils.ImageDatasetLoader.loadImageAsTensor;

@Component
public class PatternClassifierService {
    private final NDManager manager;
    private Model model;
    private final CNNTrainer trainer;

    public PatternClassifierService(DJLConfigComponent djlConfig, CNNTrainer trainer) {
        this.manager = djlConfig.getManager();
        this.trainer = trainer;
    }

    public void createCnnModel(
            int conv1Filters,
            int conv2Filters,
            int denseUnits,
            int outputClasses) {

        Model model = Model.newInstance("opticheck-cnn");

        SequentialBlock block = new SequentialBlock();

        block
                // Conv block 1
                .add(Conv2d.builder()
                        .setFilters(conv1Filters)
                        .setKernelShape(new Shape(3,3))
                        .optPadding(new Shape(1,1))
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2,2)))

                // Conv block 2
                .add(Conv2d.builder()
                        .setFilters(conv2Filters)
                        .setKernelShape(new Shape(3,3))
                        .optPadding(new Shape(1,1))
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2,2)))

                // Classifier
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(denseUnits).build())
                .add(Activation.reluBlock())

                // Output layer
                .add(Linear.builder().setUnits(outputClasses).build());

        model.setBlock(block);
        this.model = model;
    }

    // ----------------------
    // TRAINING
    // ----------------------
    public void train(ArrayDataset dataset, int epochs) throws IOException, TranslateException {
        trainer.train(this.model,dataset,epochs);
        Files.list(Paths.get("models")).forEach(p -> p.toFile().delete());
        model.save(Paths.get("models"), "opticheck-model");
    }


    public List<FilePrediction> filePredictions() throws IOException, TranslateException {
        List<FilePrediction> filePredictions = new ArrayList<>();
        var predictDir = new File("predict");
        if(Arrays.stream(Objects.requireNonNull(predictDir.listFiles())).toList().isEmpty()) {
            throw new IOException("No file found in predict folder");
        }
        for(File imgFile : Objects.requireNonNull(predictDir.listFiles())) {
            filePredictions.add(predictSingleFile(imgFile));
        }
        return filePredictions;
    }



    // ----------------------
    // PREDICT FROM FILE
    // ----------------------
    public FilePrediction predictSingleFile(File imageFile) throws IOException {

        // Convert file → float[3][64][64]
        float[][][] image = loadImageAsTensor(imageFile);

        int predictedClass = cnnPredict(model, image);

        return new FilePrediction(imageFile.getName(), predictedClass);
    }



    private int cnnPredict(Model model, float[][][] image) {

        try (NDManager manager = NDManager.newBaseManager()) {

            float[] flat = flattenImage(image);

            NDArray input = manager.create(flat)
                    .reshape(1, 3, 64, 64); // CNN tensor

            ParameterStore ps = new ParameterStore(manager, false);

            NDList output = model.getBlock()
                    .forward(ps, new NDList(input), false);

            return (int) output.singletonOrThrow()
                    .argMax()
                    .getLong(); // 0=OK, 1=NOK
        }
    }

    private float[] flattenImage(float[][][] image) {

        int c = image.length;
        int h = image[0].length;
        int w = image[0][0].length;

        float[] flat = new float[c * h * w];

        int idx = 0;

        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    flat[idx++] = image[ch][y][x];

        return flat;
    }

    public Model getModel() {
        return model;
    }

    @PostConstruct
    public NDManager getManager() {
        return manager;
    }

    public void loadModel(int firstConvFilters, int secondConvFilters, int denseNeurons, int outputClasses) throws IOException, MalformedModelException {

        Model modelToLoad = Model.newInstance("opticheck-cnn");

        // Rebuild SAME CNN architecture used during training
        SequentialBlock block = new SequentialBlock();

        block
                .add(Conv2d.builder()
                        .setFilters(firstConvFilters)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2, 2)))

                .add(Conv2d.builder()
                        .setFilters(secondConvFilters)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2, 2)))

                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(denseNeurons).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(outputClasses).build());

        modelToLoad.setBlock(block);

        // Load weights from file
        modelToLoad.load(Paths.get("models"), "opticheck-cnn");

        this.model = modelToLoad;
    }

}

