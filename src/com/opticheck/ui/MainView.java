package com.opticheck.ui;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.translate.Pipeline;
import com.opticheck.utils.ImageDatasetLoader;
import com.opticheck.service.PatternClassifierService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.upload.Upload;
import com.vaadin.flow.component.upload.receivers.MultiFileMemoryBuffer;
import com.vaadin.flow.router.Route;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.TranslateException;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.Arrays;

@Route("")
@Component
public class MainView extends Div {

    @Autowired
    private PatternClassifierService classifierService;

    public MainView() {
        add(new H2("OptiCheck — Industrial Defect Classifier"));

        Button trainButton = new Button("Train Model", e -> trainModel());
        Button saveButton = new Button("Save Model", e -> saveModel());
        Button predictButton = new Button("Predict");
        predictButton.addClickListener(event -> {
            try {
                var result = classifierService.predictSingleFile();

                // Build a fresh message each time (avoid reusing components)
                String message = result == 0 ? "Pattern is RIGHT ✅" : "Pattern is WRONG ❌";
                Notification notification = new Notification(message, 3000); // visible for 3s
                notification.setPosition(Notification.Position.TOP_CENTER);
                notification.open();

            } catch (Exception e) {
                Notification error = new Notification("Prediction failed: " + e.getMessage(), 5000);
                error.setPosition(Notification.Position.TOP_CENTER);
                error.open();
            }
        });

        add(trainButton, saveButton, predictButton);
    }

    private void trainModel() {
        System.out.println("Loading dataset...");

        try {
            // Use a long-living global manager from your classifierService
            NDManager manager = classifierService.getManager();

            // Load training data
            ArrayDataset dataset = ImageDatasetLoader.loadDataset("training-data", manager);

            // Create the model (input size = 64*64*3, hidden = 32, output = 2 classes)
            classifierService.createMLP(64*64*3, 256, 2);;

            // Train for 30 epochs
            classifierService.train(dataset, 60);

            Notification.show("Training complete!");

        } catch (Exception ex) {
            ex.printStackTrace();
            Notification.show("Training failed: " + ex.getMessage());
        }
    }

    private void saveModel() {
        try {
            classifierService.saveModel(new File("models"));
            Notification.show("Model saved!");
        } catch (IOException ex) {
            Notification.show("Failed to save model: " + ex.getMessage());
        }
    }

}
