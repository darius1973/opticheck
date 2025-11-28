package com.opticheck.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.TranslateException;
import com.opticheck.config.DJLConfigComponent;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


import com.opticheck.pojo.FilePrediction;

@Component
public class PatternClassifierService {

    private static final int INPUT_SIZE = 64 * 64 * 3;
    private static final int NUM_CLASSES = 2;
    private final NDManager manager;
    private Model model;
    private final com.opticheck.trainer.Trainer trainer;

    public PatternClassifierService(DJLConfigComponent djlConfig, com.opticheck.trainer.Trainer trainer) {
        this.manager = djlConfig.getManager();
        this.trainer = trainer;
    }

    public void createMLP(int hiddenNodes, int outputClasses) {
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


        // Load image as float[] (normalized)
        float[] input = loadImageAsFloatArray(imageFile);

        // Use the service's long-living NDManager
        var manager = getManager();

        var array = manager.create(input).reshape(1, input.length); // shape [1, 64*64*3]
        var inputList = new NDList(array);

        // Forward pass through model
        var ps = new ParameterStore(manager, false);
        var output = model.getBlock().forward(ps, inputList, false, null);

        // Get prediction: cast to INT32 before argMax to fix mismatch
        var result = output.singletonOrThrow();
        int predictedClass = (int) result.argMax().getLong();
        return new FilePrediction(imageFile.getName(), predictedClass); // argMax along class dimension, returns int
    }


    // ----------------------
    // IMAGE LOADING — FIXED (64×64, RGB, normalized)
    // ----------------------
    private float[] loadImageAsFloatArray(File file) throws IOException {

        try (var m = NDManager.newBaseManager()) {

            var img = ImageFactory.getInstance().fromFile(file.toPath());
            // IMPORTANT FIX  - resize image to 64 X 64 pixels, no copy
            img = img.resize(64, 64,false);

            /*
             * ===========================================================================
             *     IMAGE → NDARRAY CONVERSION (THE CORE INPUT PIPELINE OF OptiCheck)
             * ===========================================================================
             *
             *  This operation converts a normal Java BufferedImage into a numeric NDArray
             *  that the neural network can understand. The entire classifier depends on
             *  this step to transform visual data → mathematical tensors.
             *
             * ---------------------------------------------------------------------------
             *  WHAT EXACTLY HAPPENS INSIDE image.toNDArray(manager)
             * ---------------------------------------------------------------------------
             *
             *  Step 1: Extract pixel RGB values from the Java image
             *  -----------------------------------------------
             *  Suppose we have a tiny 2×2 RGB image:
             *
             *        [ (R=10,G=20,B=30),  (R=40,G=50,B=60) ]
             *        [ (R=70,G=80,B=90),  (R=100,G=110,B=120) ]
             *
             *  These are raw byte values (0–255) from the BufferedImage raster.
             *
             *
             *  Step 2: Convert all pixels → float32
             *  -----------------------------------------------
             *  DJL converts these integers into floats:
             *
             *      10f, 20f, 30f, 40f, 50f, 60f, ...
             *
             *  Neural networks only operate on floating-point tensors, not bytes.
             *
             *
             *  Step 3: Construct a tensor with shape (height, width, channels)
             *  -----------------------------------------------
             *  For our 2×2 example, the NDArray shape becomes:
             *
             *      (2, 2, 3)
             *
             *  Meaning:
             *      - height = 2
             *      - width  = 2
             *      - channels = 3 (RGB)
             *
             *  Stored as:
             *
             *      [
             *        [ [10,20,30], [40,50,60] ],
             *        [ [70,80,90], [100,110,120] ]
             *      ]
             *
             *  For this real case:
             *
             *      64 × 64 RGB → shape (64, 64, 3)
             *
             *
             *  Step 4: Store the NDArray in the given NDManager
             *  -----------------------------------------------
             *  The NDManager manages GPU/CPU memory, lifecycle, and ensures tensors
             *  persist long enough for training or inference.
             *
             *
             * ---------------------------------------------------------------------------
             *  WHY THIS IS THE HEART OF THE MODEL
             * ---------------------------------------------------------------------------
             *
             *  After this conversion:
             *
             *      NDArray image = image.toNDArray(manager);
             *
             *  the NDArray becomes the *input features* for the neural network.
             *
             *  Each pixel channel is one feature.
             *
             *  Real-number count:
             *
             *      64 × 64 × 3 = 12,288 input features
             *
             *  So the 1D vector passed to the neural network looks like:
             *
             *      [12.0, 24.0, 18.0, ..., 128.0, 64.0]  // 12,288 values
             *
             *  This is analogous to “price”, “size”, “rooms” in a real-estate dataset,
             *  except here you have *thousands of features* extracted from pixels.
             *
             *
             * ---------------------------------------------------------------------------
             *  FULL PIPELINE EXAMPLE
             * ---------------------------------------------------------------------------
             *
             *  BufferedImage (e.g., a picture of a defect)
             *       ↓
             *  Convert to NDArray (H,W,C)
             *       ↓
             *  Resize to 64×64 if necessary:
             *       image = ImagePreprocessor.resize(image, 64, 64);
             *       ↓
             *  Convert to flat vector:
             *       NDArray flat = array.reshape(1, 64*64*3);
             *       ↓
             *  Model forward pass:
             *       NDList out = model.getBlock().forward(store, new NDList(flat), false, null);
             *       ↓
             *  Softmax:
             *       [0.97, 0.03]  →  class 0 (“OK”)
             *
             *
             * ---------------------------------------------------------------------------
             *  SUMMARY
             * ---------------------------------------------------------------------------
             *
             *  image.toNDArray(manager) performs:
             *      ✔ Extract RGB pixel data
             *      ✔ Convert to float32 tensor
             *      ✔ Build tensor with shape (H,W,C)
             *      ✔ Place tensor under NDManager memory
             *      ✔ Produce the neural network's actual input features
             *
             *  This is the single most important transformation in your entire system.
             *
             * ===========================================================================
             */

            NDArray arr = img.toNDArray(m)
                    .toType(DataType.FLOAT32, true)
                    .div(255f);

            return arr.toFloatArray();
        }
    }

    public Model getModel() {
        return model;
    }

    @PostConstruct
    public NDManager getManager() {
        return manager;
    }

    public void loadModel(int hiddenNodes, int outputClasses) throws IOException, MalformedModelException {
        // Rebuild the SAME block used when training
        var block = new SequentialBlock()
                // within Linear, the default weights and bias-ses are initialized as follows:
                // bias = 0
                // weights (Xavier formula)
                // w∼U(−limit,limit)  , where limit = SQRT (6/(fan_in + fan_out))
                // U meaning random distribution
                // fan_in = number of input units to the layer,
                // fan_out = number of output units from the layer
                // If weights are too large, activations explode layer-to-layer; if too small, they vanish.
                // To prevent this  - we initialize with Xavier formula.
                .add(Linear.builder().setUnits(hiddenNodes).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(outputClasses).build());

        var modelToLoad = Model.newInstance("opticheck-model");
        modelToLoad.setBlock(block);

        // Load the params file
        modelToLoad.load(Paths.get("models"), "opticheck-model");

        this.model = modelToLoad;

    }

}

