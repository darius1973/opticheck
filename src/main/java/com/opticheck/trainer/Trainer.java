package com.opticheck.trainer;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.opticheck.interfaces.TrainingListenerUI;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class Trainer {

    private static final int INPUT_SIZE = 64 * 64 * 3;


    private TrainingListenerUI uiLogger;

    public void setUiLogger(@Autowired TrainingListenerUI uiLogger) {
        this.uiLogger = uiLogger;
    }

    private void log(String msg) {
        if (uiLogger != null) uiLogger.onLog(msg);
    }

    private void error(String msg) {
        if (uiLogger != null) uiLogger.onError(msg);
    }

    // ----------------------
    // TRAINING
    // ----------------------
    public void train(Model model, ArrayDataset dataset, int epochs) throws IOException, TranslateException {
        Tracker lrTracker = Tracker.multiFactor()
                .setBaseValue(0.001f)
                .optFactor(0.2f)
                .setSteps(new int[]{5, 10})
                .build();
        Optimizer optimizer = Optimizer.adam()
                .optLearningRateTracker(lrTracker)
                .optWeightDecays(1e-5f)
                .build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(optimizer)
                .addEvaluator(new Accuracy());

        try (ai.djl.training.Trainer trainer = model.newTrainer(config)) {

            trainer.initialize(new Shape(1, INPUT_SIZE));

            for (int epoch = 0; epoch < epochs; epoch++) {
                log("▶ Starting epoch " + epoch);
                for (Batch batch : trainer.iterateDataset(dataset)) {

                    try (GradientCollector gc = trainer.newGradientCollector()) {

                        NDArray data = batch.getData().head();
                        NDArray label = batch.getLabels().head().toType(DataType.INT32, false);

                        // forward() uses model block
                        NDArray pred = trainer.forward(new NDList(data)).singletonOrThrow();

                        NDArray loss = trainer.getLoss().evaluate(new NDList(label), new NDList(pred));
                        log("   • Loss: " + loss.getFloat());
                        System.out.println("Loss: " + loss.getFloat());
                        gc.backward(loss);
                    }

                    trainer.step();
                    batch.close();
                }

                log("✔ Epoch " + epoch + " complete\n");
                System.out.println("✔ Epoch " + epoch + " complete\n");
            }
        }
    }
}
