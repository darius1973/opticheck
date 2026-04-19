package com.opticheck.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.TranslateException;
import com.opticheck.trainer.CNNTrainer;
import com.opticheck.utils.ImageDatasetLoader;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


import com.opticheck.pojo.FilePrediction;

@Component
public class PatternClassifierService {

    private Model model;
    private final CNNTrainer trainer;

    private static final String MODEL_NAME = "xray-model";

    public PatternClassifierService(CNNTrainer trainer) {
        this.trainer = trainer;
    }

    // =========================================================
    // MODEL ARCHITECTURE (shared for train + load)
    // SequentialBlock  - is a pipeline of layers executed one after another
    // For example: Image → Conv → ReLU → Pool → Conv → ReLU → Dense → Output
    // So this is stack of operations the NN does
    // NN structure
    // Input (3,224,224)
    // → Conv(32)
    // → Pool
    // → Conv(64)
    // → Pool
    // → Conv(128)
    // → Pool
    // → Flatten
    // → Dense(256)
    // → Dense(1)
    // =========================================================
    private SequentialBlock buildModel(int outputClasses) {

        return new SequentialBlock()
                // apply filters
                .add(createBackbone())
                //then flatten-ize
                .add(Blocks.batchFlattenBlock())
                //then dense layers
                .add(Linear.builder().setUnits(256).build())
                .add(Activation.reluBlock())
                //and the output
                .add(Linear.builder().setUnits(outputClasses).build());
    }

    private Block createBackbone() {
        // layer stack:
        // Conv1 → edges
        // Conv2 → shapes
        // Conv3 → structures
        // Dense → decision
        return new SequentialBlock()
                // Conv block 1
                // scan the image with 32 different 3×3 detectors and produce 32 feature maps
                .add(Conv2d.builder()
                        .setFilters(32) // -> 32 3x3 detectors
                        .setKernelShape(new Shape(3, 3)) // -> what it looks at : 3 x 3 window
                        .optPadding(new Shape(1, 1)) // -> don't shrink the image
                        .build())
                .add(Activation.reluBlock()) //apply ReLu activation - for non-linearity
                .add(Pool.maxPool2dBlock(new Shape(2, 2)))

                // Conv block 2
                // scan the image with 64 different 3×3 detectors
                .add(Conv2d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2, 2)))

                // Conv block 3
                // scan the image with 128 different 3×3 detectors
                .add(Conv2d.builder()
                        .setFilters(128)
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2, 2)));

    }

    // =========================================================
    // CREATE MODEL (before training)
    // =========================================================
    public void createModel(int outputClasses) {

        Model m = Model.newInstance(MODEL_NAME);
        m.setBlock(buildModel(outputClasses));

        this.model = m;
    }

    // =========================================================
    // TRAIN + SAVE
    // =========================================================
    public void train(ArrayDataset dataset, int epochs)
            throws IOException, TranslateException {

        trainer.train(model, dataset, epochs);

        model.save(Paths.get("models"), MODEL_NAME);
    }

    // =========================================================
    // LOAD MODEL (after restart)
    // =========================================================
    public void loadModel(int outputClasses)
            throws IOException, MalformedModelException {

        Model m = Model.newInstance(MODEL_NAME);

        m.setBlock(buildModel(outputClasses));

        m.load(Paths.get("models"), MODEL_NAME);

        this.model = m;
    }

    // =========================================================
    // PREDICT MULTIPLE FILES
    // =========================================================
    public List<FilePrediction> filePredictions() throws IOException, TranslateException {

        File dir = new File("predict");

        File[] files = dir.listFiles();
        if (files == null || files.length == 0) {
            throw new IOException("No image files found in predict folder");
        }

        List<FilePrediction> results = new ArrayList<>();

        for (File file : files) {
            if (isImageFile(file)) {
                results.add(predictSingleFile(file));
            }
        }

        return results;
    }

    // =========================================================
    // PREDICT SINGLE FILE
    // =========================================================
    public FilePrediction predictSingleFile(File imageFile) throws IOException {

        try (NDManager manager = NDManager.newBaseManager()) {

            NDArray image = ImageDatasetLoader.loadImageAsNDArray(imageFile, manager);

            float prob = predict(image, manager);

            // -----------------------
            // THRESHOLDS (tunable)
            // -----------------------
            float low = 0.4f;
            float high = 0.7f;

            int predictedClass;
            String confidenceLabel;

            if (prob < low) {
                predictedClass = 0; // OK
                confidenceLabel = "CONFIDENT OK";
            } else if (prob > high) {
                predictedClass = 1; // NOT OK
                confidenceLabel = "CONFIDENT NOT OK";
            } else {
                predictedClass = 1; // or 0 depending on strategy
                confidenceLabel = "UNCERTAIN";
            }

            // -----------------------
            // DEBUG LOG (IMPORTANT)
            // -----------------------
            System.out.println(
                    imageFile.getName() +
                            " → prob=" + prob +
                            " → " + confidenceLabel
            );


            return new FilePrediction(
                    imageFile.getName(),
                    prob,
                    prob > 0.7f ? 1 : 0
            );
        }
    }

    // =========================================================
    // CORE INFERENCE
    // =========================================================
    private float predict(NDArray image, NDManager manager) {

        NDArray input = image.expandDims(0);

        ParameterStore ps = new ParameterStore(manager, false);

        NDArray logits = model.getBlock()
                .forward(ps, new NDList(input), false)
                .singletonOrThrow();

        // sigmoid (stable for binary classification)
        NDArray probs = logits.exp().div(logits.exp().add(1));

        return probs.getFloat(0);
    }

    public Model getModel() {
        return model;
    }

    // optional helper if you use it
    private boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg");
    }
}
