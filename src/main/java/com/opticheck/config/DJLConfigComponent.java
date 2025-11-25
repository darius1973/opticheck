package com.opticheck.config;


import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;

@Component
public class DJLConfigComponent {

    private NDManager manager;

    @PostConstruct
    public void init() {
        // 1️⃣ Tune PyTorch threads for CPU
        System.setProperty("ai.djl.pytorch.num_threads", "8");         // intra-op threads
        System.setProperty("ai.djl.pytorch.num_interop_threads", "8"); // inter-op threads

        // 2️⃣ Keep graph executor enabled
        System.setProperty("ai.djl.pytorch.enable_graph_executor", "true");

        // 3️⃣ Verify engine and device
        Engine engine = Engine.getEngine("PyTorch");
        System.out.println("Engine: " + engine.getEngineName());

        Device device = Device.cpu(); // change to Device.gpu() if you have GPU
        manager = NDManager.newBaseManager(device);

        System.out.println("DJL PyTorch configured for faster CPU training.");
    }

    public NDManager getManager() {
        return manager;
    }
}
