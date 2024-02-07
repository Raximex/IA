import java.util.Arrays;
import java.util.Arrays;

public class Reseau {

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double[] hidden(double[] X, double[][] W_1, double[][] b_1) {
        double[] h = new double[b_1.length];
        for (int i = 0; i < b_1.length; i++) {
            double sum = 0;
            for (int j = 0; j < W_1[i].length; j++) {
                sum += W_1[i][j] * X[j];
            }
            h[i] = sigmoid(sum + b_1[i][0]);
        }
        return h;
    }

    private static double output(double[] H, double[][] W_2, double[][] b_2) {
        double sum = 0;
        for (int i = 0; i < H.length; i++) {
            sum += W_2[i][0] * H[i];
        }
        return sigmoid(sum + b_2[0][0]);
    }

    private static double delta_2(double o, double y) {
        return (o - y) * o * (1 - o);
    }

    private static double[] delta_1(double[][] W_2, double[] h, double d2) {
        double[] d1 = new double[h.length];
        for (int i = 0; i < h.length; i++) {
            d1[i] = W_2[i][0] * d2 * h[i] * (1 - h[i]);
        }
        return d1;
    }

    private static double[][] dJdW_2(double[] h, double d2) {
        double[][] dj = new double[h.length][1];
        for (int i = 0; i < h.length; i++) {
            dj[i][0] = d2 * h[i];
        }
        return dj;
    }

    private static double[][] dJdW_1(double[] X, double[] d1) {
        double[][] dj = new double[d1.length][X.length];
        for (int i = 0; i < d1.length; i++) {
            for (int j = 0; j < X.length; j++) {
                dj[i][j] = d1[i] * X[j];
            }
        }
        return dj;
    }

    private static void trainModel(double[][] X, double[][] W_1, double[][] W_2, double[][] b_1, double[][] b_2,
            double[] Y, double alpha, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Epoch: " + epoch);
            for (int i = 0; i < X.length; i++) {
                double[] h = hidden(X[i], W_1, b_1);
                double o = output(h, W_2, b_2);
                double d2 = delta_2(o, Y[i]);
                double[] d1 = delta_1(W_2, h, d2);
                double[][] dj2 = dJdW_2(h, d2);
                double[][] dj1 = dJdW_1(X[i], d1);

                // Print values for debugging
                System.out.println("h: " + Arrays.toString(h));
                System.out.println("o: " + o);
                System.out.println("d2: " + d2);
                System.out.println("d1: " + Arrays.toString(d1));
                System.out.println("dj2: " + Arrays.deepToString(dj2));
                System.out.println("dj1: " + Arrays.deepToString(dj1));

                // Update weights and biases
                for (int j = 0; j < W_2.length; j++) {
                    W_2[j][0] -= alpha * dj2[j][0];
                }
                for (int j = 0; j < W_1.length; j++) {
                    for (int k = 0; k < W_1[j].length; k++) {
                        W_1[j][k] -= alpha * dj1[j][k];
                    }
                }
                for (int j = 0; j < b_1.length; j++) {
                    b_1[j][0] -= alpha * d1[j];
                }
                b_2[0][0] -= alpha * d2;
            }

            // Print updated weights and biases
            System.out.println("Updated W_1: " + Arrays.deepToString(W_1));
            System.out.println("Updated W_2: " + Arrays.deepToString(W_2));
            System.out.println("Updated b_1: " + Arrays.deepToString(b_1));
            System.out.println("Updated b_2: " + Arrays.deepToString(b_2));
        }
    }

    private static double[] predictModel(double[][] X, double[][] W_1, double[][] W_2, double[][] b_1, double[][] b_2) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double[] h = hidden(X[i], W_1, b_1);
            predictions[i] = output(h, W_2, b_2);
        }
        return predictions;
    }

    public static void main(String[] args) {
        // Define input and output data
        double[][] X = { { -0.41675785, 0.26806408 } };
        double[][] W_1 = { { 0.68604422, -1.0 }, { -0.07099212, -0.5677824 }, { 0.21776441, -1.0 } };
        double[][] W_2 = { { -0.05148292 }, { -0.54286148 }, { -0.88762896 } };
        double[][] b_1 = { { -0.03349712 }, { -0.01 }, { -0.03712303 } };
        double[][] b_2 = { { 0.2799246 } };
        double[] y = { 0, 1, 0, 1, 1 };

        // Train the model
        trainModel(X, W_1, W_2, b_1, b_2, y, 0.1, 3);

        // Make predictions
        double[] predictions = predictModel(X, W_1, W_2, b_1, b_2);
        System.out.println(Arrays.toString(predictions));
    }
}
