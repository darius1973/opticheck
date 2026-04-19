package com.opticheck.trainer;


import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.opticheck.interfaces.TrainingListenerUI;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Component
public class CNNTrainer {

    private static final int IMAGE_SIZE = 224;
    private static final int CHANNELS = 3;

    private TrainingListenerUI uiLogger;

    public void setUiLogger(@Autowired TrainingListenerUI uiLogger) {
        this.uiLogger = uiLogger;
    }

    private void log(String msg) {
        if (uiLogger != null) uiLogger.onLog(msg);
    }

    // =========================
    // MAIN TRAINING LOOP
    // =========================
    public void train(Model model, ArrayDataset dataset, int epochs)
            throws IOException, TranslateException {

        for (int epoch = 0; epoch < epochs; epoch++) {

            float lr = (epoch < 3) ? 1e-4f : 1e-5f;

            log("\n▶ Epoch " + epoch + " | LR=" + lr);

            var optimizer = Optimizer.adam()
                    .optLearningRateTracker(Tracker.fixed(lr))
                    .optWeightDecays(1e-5f)
                    .build();

            var config = new DefaultTrainingConfig(
                    Loss.sigmoidBinaryCrossEntropyLoss()
            )
                    .optOptimizer(optimizer)
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            List<Float> allPreds = new ArrayList<>();
            List<Float> allLabels = new ArrayList<>();

            try (Trainer trainer = model.newTrainer(config)) {

                trainer.initialize(new Shape(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE));

                for (Batch batch : trainer.iterateDataset(dataset)) {

                    try (GradientCollector gc = trainer.newGradientCollector()) {

                        NDArray data = batch.getData().head();

                        System.out.println("IMAGE SHAPE: " + data.getShape());
                        System.out.println("MIN: " + data.min().getFloat());
                        System.out.println("MAX: " + data.max().getFloat());

                        NDArray label = batch.getLabels().head()
                                .toType(DataType.FLOAT32, false)
                                .reshape(-1, 1);

                        NDArray logits = trainer.forward(new NDList(data))
                                .singletonOrThrow();
                        System.out.println("LOGITS MEAN: " + logits.mean().getFloat());
                        System.out.println("LOGITS MIN: " + logits.min().getFloat());
                        System.out.println("LOGITS MAX: " + logits.max().getFloat());
                        // sigmoid
                        NDArray exp = logits.exp();
                        NDArray probs = exp.div(exp.add(1));

                        // store metrics ONLY ONCE
                        for (int i = 0; i < probs.getShape().get(0); i++) {

                            float predVal = probs.getFloat(i, 0);
                            float labelVal = label.getFloat(i, 0);

                            allPreds.add(predVal);
                            allLabels.add(labelVal);
                        }

                        // DEBUG (batch mean only)
                        System.out.println("batch mean prob = " + probs.mean().getFloat());

                        NDArray loss = trainer.getLoss()
                                .evaluate(new NDList(label), new NDList(logits));

                        gc.backward(loss);
                    }

                    trainer.step();
                    batch.close();
                }
            }

            // =========================
            // METRICS (per epoch)
            // =========================
            float threshold = 0.5f;

            float sensitivity = computeSensitivity(allPreds, allLabels, threshold);
            float specificity = computeSpecificity(allPreds, allLabels, threshold);
            float auc = computeAUC(allPreds, allLabels);

            log("📊 Sensitivity (Recall): " + sensitivity);
            log("📉 Specificity: " + specificity);
            log("📈 AUC: " + auc);
        }
    }

    // =========================
    // SENSITIVITY (RECALL)
    // =========================
    private float computeSensitivity(List<Float> preds, List<Float> labels, float threshold) {

        int tp = 0, fn = 0;

        for (int i = 0; i < preds.size(); i++) {

            if (labels.get(i) == 1f) {
                if (preds.get(i) >= threshold) tp++;
                else fn++;
            }
        }

        return tp / (float) (tp + fn + 1e-6);
    }

    // =========================
    // SPECIFICITY
    // =========================
    private float computeSpecificity(List<Float> preds, List<Float> labels, float threshold) {

        int tn = 0, fp = 0;

        for (int i = 0; i < preds.size(); i++) {

            if (labels.get(i) == 0f) {
                if (preds.get(i) < threshold) tn++;
                else fp++;
            }
        }

        return tn / (float) (tn + fp + 1e-6);
    }

    // =========================
    // AUC (Rank-based)
    // =========================
    private float computeAUC(List<Float> preds, List<Float> labels) {

        int n = preds.size();

        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < n; i++) idx.add(i);

        idx.sort(Comparator.comparing(preds::get));

        float rankSum = 0;
        int pos = 0, neg = 0;

        for (int i = 0; i < n; i++) {

            int id = idx.get(i);

            if (labels.get(id) == 1f) {
                rankSum += (i + 1);
                pos++;
            } else {
                neg++;
            }
        }

        if (pos == 0 || neg == 0) return 0.5f;

        return (rankSum - pos * (pos + 1) / 2f)
                / (pos * neg);
    }
}