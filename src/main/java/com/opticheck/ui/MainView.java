package com.opticheck.ui;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import com.opticheck.interfaces.TrainingListenerUI;
import com.opticheck.pojo.FilePrediction;
import com.opticheck.utils.ImageDatasetLoader;
import com.opticheck.service.PatternClassifierService;
import com.vaadin.flow.component.UI;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.Image;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.Route;
import org.springframework.beans.factory.annotation.Autowired;

import java.io.IOException;
import com.opticheck.trainer.CNNTrainer;

@Route("")
//@Component
public class MainView extends VerticalLayout implements TrainingListenerUI {

    @Autowired
    private CNNTrainer trainer;

    @Autowired
    private PatternClassifierService classifierService;

    private Div statusBox;
    private UI uiRef;   // ← store UI reference for background thread

    //convoluted neural net
    private static final int FIRST_CONV_FILTERS = 16;
    private static final int SECOND_CONV_FILTERS = 32;
    private static final int DENSE_NEURONS = 64;
    private static final int OUTPUT_CLASSES = 1;

    public MainView() {

        add(new H2("OptiCheck — Neural Network Classifier"));

        // -------------------------
        // Buttons
        // -------------------------
        var trainButton = new Button("Train Model", e -> trainModel());
        var saveButton = new Button("Load Model", e -> loadModel());
        var predictButton = new Button("Test(Predict)");

        predictButton.addClickListener(event -> {
            try {
                var filePredictions = classifierService.filePredictions();

                StringBuilder message = new StringBuilder();

                for (FilePrediction fp : filePredictions) {

                    float prob = fp.probability();
                    int predicted = fp.predictedClass();

                    String status;
                    String icon;

                    if (predicted == 0) {
                        status = "OK";
                        icon = "✅";
                    } else {
                        status = "NOT OK";
                        icon = "❌";
                    }

                    message.append("📄 ")
                            .append(fp.fileName())
                            .append(" → ")
                            .append(status)
                            .append(" ")
                            .append(icon)
                            .append(" (confidence: ")
                            .append(String.format("%.2f", prob))
                            .append(")\n");
                }

                Notification notification = Notification.show(
                        message.toString(),
                        8000,
                        Notification.Position.TOP_CENTER
                );

                notification.addThemeVariants(NotificationVariant.LUMO_PRIMARY);

            } catch (Exception e) {
                Notification notification = Notification.show(
                        "Prediction failed: " + e.getMessage(),
                        5000,
                        Notification.Position.TOP_CENTER
                );

                notification.addThemeVariants(NotificationVariant.LUMO_ERROR);
            }
        });

        var buttonRow = new HorizontalLayout(trainButton, saveButton, predictButton);
        buttonRow.setSpacing(true);
        buttonRow.setPadding(true);

        add(buttonRow);

        // -------------------------
        // Status Box (Console)
        // -------------------------
        statusBox = new Div();
        statusBox.setId("statusBox");
        statusBox.getStyle().set("border", "1px solid #ccc");
        statusBox.getStyle().set("padding", "10px");
        statusBox.getStyle().set("width", "300px");
        statusBox.getStyle().set("height", "600px");
        statusBox.getStyle().set("overflow-y", "auto");
        statusBox.getStyle().set("white-space", "pre-wrap");

        // GIF
        var aiGif = new Image("frontend/images/anim.gif", "AI Face Animation");
        aiGif.setWidth("800px");
        aiGif.setHeight("600px");

        // Place console + GIF horizontally
        var contentRow = new HorizontalLayout(statusBox, aiGif);
        contentRow.setAlignItems(Alignment.START);

        add(contentRow);
    }

    // ------------------------------------------------------------
    // START TRAINING
    // ------------------------------------------------------------
    private void trainModel() {
        uiRef = UI.getCurrent();   // IMPORTANT: capture UI before thread starts

        statusBox.setText("Training started...\n");

        trainer.setUiLogger(this);

        // Run in background so UI stays responsive
        new Thread(() -> {
            uiRef.access(() -> appendMessage("Training thread started...", false));

            executeTraining();

            uiRef.access(() -> appendMessage("Training FINISHED ✔", false));
        }).start();
    }

    private void executeTraining() {

        try (NDManager manager = NDManager.newBaseManager()) {

            if (uiRef != null) {
                uiRef.access(() -> appendMessage("📂 Loading dataset...", false));
            }

            var dataset = ImageDatasetLoader.loadDataset("training-data", manager);

            if (uiRef != null) {
                uiRef.access(() -> appendMessage("🧠 Creating DenseNet model...", false));
            }

            classifierService.createModel(OUTPUT_CLASSES);

            if (uiRef != null) {
                uiRef.access(() -> appendMessage("🚀 Training started...", false));
            }

            classifierService.train(dataset, 40);

            if (uiRef != null) {
                uiRef.access(() -> appendMessage("✅ Training completed!", false));
            }

        } catch (Exception ex) {
            ex.printStackTrace();

            if (uiRef != null) {
                uiRef.access(() -> appendMessage(
                        "❌ ERROR: " + ex.getMessage(),
                        true
                ));
            }
        }
    }

    // ------------------------------------------------------------
    // TRAINER CALLBACKS
    // ------------------------------------------------------------
    @Override
    public void onLog(String message) {
        if (uiRef != null) {
            uiRef.access(() -> appendMessage(message, false));
        }
    }

    @Override
    public void onError(String message) {
        if (uiRef != null) {
            uiRef.access(() -> appendMessage(message, true));
        }
    }

    // ------------------------------------------------------------
    // UI LOGGING
    // ------------------------------------------------------------
    private void appendMessage(String msg, boolean error) {
        String coloredMsg = "";

        if (error) {
            coloredMsg = "<span style='color:red;'>" + escapeHtml(msg) + "</span>";
        } else if (msg.toLowerCase().contains("loss")) {
            float lossValue = extractLoss(msg); // parse the number
            String color;
            if (lossValue < 0.29f) {
                color = "green";
            } else if (lossValue < 0.60f) {
                color = "orange";
            } else {
                color = "red";
            }
            coloredMsg = "<span style='color:" + color + ";'>" + escapeHtml(msg) + "</span>";
        } else {
            coloredMsg = "<span style='color:black;'>" + escapeHtml(msg) + "</span>";
        }

        // Append HTML instead of plain text
        statusBox.getElement().setProperty("innerHTML",
                statusBox.getElement().getProperty("innerHTML") + coloredMsg + "<br>");

        // Auto scroll
        UI.getCurrent().getPage().executeJs(
                "var el=document.getElementById('statusBox'); el.scrollTop=el.scrollHeight;");
    }

    // Helper to escape any special HTML chars in messages
    private String escapeHtml(String s) {
        return s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");
    }

    private void appendMessagePrev(String msg, boolean error) {
        if (error) {
            statusBox.getStyle().set("color", "red");
        } else {
            if (msg.toLowerCase().contains("loss")) {
                float lossValue = extractLoss(msg); // parse the number from the message
                if (lossValue < 0.19f) {
                    statusBox.getStyle().set("color", "green");
                } else if (lossValue < 0.60f) {
                    statusBox.getStyle().set("color", "orange");
                } else if (lossValue < 1.0f){
                    statusBox.getStyle().set("color", "pink");
                } else {
                    statusBox.getStyle().set("color", "red");
                }
            } else {
                statusBox.getStyle().set("color", "black");
            }
        }

        statusBox.setText(statusBox.getText() + msg + "\n");

        // Auto scroll
        UI.getCurrent().getPage().executeJs(
                "var el=document.getElementById('statusBox'); el.scrollTop=el.scrollHeight;");
    }

    private float extractLoss(String msg) {
        try {
            // Look for the first number in the string
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("([0-9]*\\.?[0-9]+)").matcher(msg);
            if (m.find()) {
                return Float.parseFloat(m.group(1));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return -1f; // invalid number
    }


    private void loadModel() {
        try {
            classifierService.loadModel(OUTPUT_CLASSES);

            Notification notification = Notification.show(
                    "✅ Model loaded successfully!",
                    4000,
                    Notification.Position.TOP_CENTER
            );

            notification.addThemeVariants(NotificationVariant.LUMO_SUCCESS);

        } catch (IOException | MalformedModelException ex) {

            Notification notification = Notification.show(
                    "❌ Failed to load model: " + ex.getMessage(),
                    5000,
                    Notification.Position.TOP_CENTER
            );

            notification.addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }
}
