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
public class CNNTrainer {

    private static final int IMAGE_SIZE = 64;
    private static final int CHANNELS = 3;

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
    public void train(Model model, ArrayDataset dataset, int epochs)
            throws IOException, TranslateException {

        // Learning rate schedule
        var lrTracker = Tracker.multiFactor()
                .setBaseValue(0.001f)      // starting learning rate
                .optFactor(0.2f)           // multiply LR by 0.2 at steps
                .setSteps(new int[]{5,10}) // epochs where LR drops
                .build();


        /*
           Adam optimizer (adaptive gradient descent)
           Adam adjusts the learning rate for each parameter individually based on the history of gradients.
           Instead of using just the current gradient, Adam keeps track of:
           First moment (mean of gradients) → like momentum
           Second moment (uncentered variance of gradients) → controls step size
           This helps training be:
                       - faster
                       - more stable
                       - less sensitive to learning-rate tuning
           Adam combines two powerful ideas:
            - Momentum
              Uses past gradients to smooth updates.
              Without momentum:  update = gradient
              With momentum:  update = running_average(gradients)
           - Adaptive Learning Rate
              Large gradients → smaller steps
              Small gradients → larger steps
              So parameters with noisy gradients automatically get smaller learning rates.
           Adam = Momentum + RMSProp + Bias Correction
         */
        var optimizer = Optimizer.adam()
                .optLearningRateTracker(lrTracker)
                .optWeightDecays(1e-5f) // L2 regularization
                .build();

        var config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(optimizer)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        try (ai.djl.training.Trainer trainer = model.newTrainer(config)) {

            // CNN input shape
            trainer.initialize(new Shape(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE));

            for (int epoch = 0; epoch < epochs; epoch++) {

                log("▶ Starting epoch " + epoch);

                for (Batch batch : trainer.iterateDataset(dataset)) {

                    try (GradientCollector gc = trainer.newGradientCollector()) {

                        NDArray data = batch.getData().head();
                        NDArray label = batch.getLabels().head()
                                .toType(DataType.INT32, false);

                        // Forward pass
                        NDArray pred =
                                trainer.forward(new NDList(data)).singletonOrThrow();

                        // Compute loss
                        NDArray loss = trainer.getLoss()
                                .evaluate(new NDList(label), new NDList(pred));

                        log("   • Loss: " + loss.getFloat());

                        // Backpropagation
                        gc.backward(loss);
                    }

                    // Update weights
                    trainer.step();

                    batch.close();
                }

                log("✔ Epoch " + epoch + " complete\n");
            }
        }
    }
}