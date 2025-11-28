package com.opticheck.utils;


import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class NumberImageGenerator {

    private static final int WIDTH = 64;
    private static final int HEIGHT = 64;
    private static final Random rnd = new Random();

    public static void generateDataset(File okFolder, File notOkFolder, int count) throws IOException {
        okFolder.mkdirs();
        notOkFolder.mkdirs();

        for (int i = 0; i < count; i++) {
            var okImg = generateRealisticNumberImage("1", true);
            ImageIO.write(okImg, "jpg", new File(okFolder, "ok_" + i + ".jpg"));

            int wrongDigit = 2 + rnd.nextInt(8); // 2…9
            var notOkImg = generateRealisticNumberImage(String.valueOf(wrongDigit), false);
            ImageIO.write(notOkImg, "jpg", new File(notOkFolder, "notok_" + i + ".jpg"));
        }
    }

    private static BufferedImage generateRealisticNumberImage(String digit, boolean ok) {
        var img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();

        // Background variation
        int bg = 230 + rnd.nextInt(25);
        g.setColor(new Color(bg, bg, bg));
        g.fillRect(0, 0, WIDTH, HEIGHT);

        // Thick, dark number
        g.setColor(ok ? Color.BLACK : Color.DARK_GRAY);
        int fontSize = 35 + rnd.nextInt(20); // bigger numbers
        g.setFont(new Font("Arial", Font.BOLD, fontSize));

        // Random rotation
        double angle = Math.toRadians(rnd.nextInt(30) - 15);
        var old = g.getTransform();
        g.rotate(angle, WIDTH / 2.0, HEIGHT / 2.0);

        // Random placement (center-ish)
        int x = rnd.nextInt(10) - 5;
        int y = rnd.nextInt(10) - 5;
        g.drawString(digit, 20 + x, 40 + y);

        g.setTransform(old);

        // Add some noise
        for (int i = 0; i < 150; i++) {
            int nx = rnd.nextInt(WIDTH);
            int ny = rnd.nextInt(HEIGHT);
            int gray = 200 + rnd.nextInt(55);
            img.setRGB(nx, ny, new Color(gray, gray, gray).getRGB());
        }

        g.dispose();
        return img;
    }

    public static void main(String[] args) throws Exception {
        var ok = new File("training-data/ok");
        var notok = new File("training-data/notok");

        generateDataset(ok, notok, 200); // 200 OK and 200 NOT OK
        System.out.println("Dataset generated.");
    }
}
