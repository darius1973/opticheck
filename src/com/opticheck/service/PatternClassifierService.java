package com.opticheck.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.tracker.CosineTracker;
import ai.djl.training.tracker.ParameterTracker;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.opticheck.config.DJLConfigComponent;
import com.opticheck.utils.ImageDatasetLoader;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;

@Component
public class PatternClassifierService {

    private static final int INPUT_SIZE = 64 * 64 * 3;
    private static final int NUM_CLASSES = 2;
    private final NDManager manager;
    private Model model;

    public PatternClassifierService(DJLConfigComponent djlConfig) {
        this.manager = djlConfig.getManager();
    }

    public void createMLP(int inputSize, int hiddenNodes, int outputClasses) {
        model = Model.newInstance("opticheck-model");

        SequentialBlock block = new SequentialBlock()
                .add(Linear.builder().setUnits(hiddenNodes).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(outputClasses).build());

        model.setBlock(block);
    }



    // ----------------------
    // TRAINING
    // ----------------------
    public void train(ArrayDataset dataset, int epochs) throws IOException, TranslateException {
        Tracker lrTracker = Tracker.fixed(0.001f);
        Optimizer optimizer = Optimizer.adam().optLearningRateTracker(lrTracker).build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(optimizer)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        try (Trainer trainer = model.newTrainer(config)) {

            trainer.initialize(new Shape(1, INPUT_SIZE));

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (Batch batch : trainer.iterateDataset(dataset)) {

                    try (GradientCollector gc = trainer.newGradientCollector()) {

                        NDArray data = batch.getData().head();
                        NDArray label = batch.getLabels().head().toType(DataType.INT32, false);

                        // forward() uses model block
                        NDArray pred = trainer.forward(new NDList(data)).singletonOrThrow();

                        NDArray loss = trainer.getLoss().evaluate(new NDList(label), new NDList(pred));
                        System.out.println("Loss: " + loss.getFloat());
                        gc.backward(loss);
                    }

                    trainer.step();
                    batch.close();
                }

                System.out.println("Epoch " + epoch + " complete");
            }
        }
    }


    // ----------------------
    // TRANSLATOR FOR PREDICTION
    // ----------------------
    private static class FloatArrayTranslator implements Translator<float[], Integer> {

        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            return new NDList(
                    ctx.getNDManager().create(input).reshape(1, INPUT_SIZE)
            );
        }

        @Override
        public Integer processOutput(TranslatorContext ctx, NDList list) {
            NDArray out = list.singletonOrThrow();
            return out.argMax().getInt();
        }
    }


    // ----------------------
    // PREDICT FLOAT ARRAY
    // ----------------------
    public int predict(float[] input) throws TranslateException {
        try (Predictor<float[], Integer> predictor = model.newPredictor(new FloatArrayTranslator())) {
            return predictor.predict(input);
        }
    }


    // ----------------------
    // PREDICT FROM FILE
    // ----------------------
    public int predictSingleFile() throws IOException, TranslateException {
        File predictDir = new File("predict");
        File imageFile = Arrays.stream(predictDir.listFiles())
                .findFirst()
                .orElseThrow(() -> new IOException("No file found in predict folder"));

        // Load image as float[] (normalized)
        float[] input = loadImageAsFloatArray(imageFile);

        // Use the service's long-living NDManager
        NDManager manager = getManager();

        NDArray array = manager.create(input).reshape(1, input.length); // shape [1, 64*64*3]
        NDList inputList = new NDList(array);

        // Forward pass through model
        ParameterStore ps = new ParameterStore(manager, false);
        NDList output = model.getBlock().forward(ps, inputList, false, null);

        // Get prediction: cast to INT32 before argMax to fix mismatch
        NDArray result = output.singletonOrThrow();
        int predictedClass = (int) result.argMax().getLong();
        return predictedClass; // argMax along class dimension, returns int
    }


    // ----------------------
    // IMAGE LOADING — FIXED (64×64, RGB, normalized)
    // ----------------------
    private float[] loadImageAsFloatArray(File file) throws IOException {

        try (NDManager m = NDManager.newBaseManager()) {

            Image img = ImageFactory.getInstance().fromFile(file.toPath());

            img = img.resize(64, 64,false); // IMPORTANT FIX

            NDArray arr = img.toNDArray(m)
                    .toType(DataType.FLOAT32, true)
                    .div(255f);

            return arr.toFloatArray();
        }
    }


    public void saveModel(File dir) throws IOException {
        model.save(dir.toPath(), "opticheck-model");
    }

    public Model getModel() {
        return model;
    }

    @PostConstruct
    public NDManager getManager() {
        return manager;
    }
}

