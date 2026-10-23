namespace uni_machine_learning_cs.MachineLearning;

// Fully connected feed-forward network (multi-layer perceptron).
// Hidden layers use the sigmoid activation, the output layer uses softmax,
// trained with backpropagation and stochastic gradient descent on the cross-entropy loss.
public class NeuralNetwork
{
    private readonly int[] layerSizes;
    // weights[l][j, i]: weight from neuron i in layer l to neuron j in layer l + 1
    private readonly double[][,] weights;
    // biases[l][j]: bias of neuron j in layer l + 1
    private readonly double[][] biases;

    public NeuralNetwork(int[] layerSizes, int seed = 42)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("A network needs at least an input and an output layer.", nameof(layerSizes));

        this.layerSizes = layerSizes;
        Random random = new Random(seed);
        weights = new double[layerSizes.Length - 1][,];
        biases = new double[layerSizes.Length - 1][];

        for (int l = 0; l < layerSizes.Length - 1; l++)
        {
            int inputs = layerSizes[l];
            int outputs = layerSizes[l + 1];
            weights[l] = new double[outputs, inputs];
            biases[l] = new double[outputs];

            // Xavier initialisation: random weights scaled by the number of inputs
            double limit = Math.Sqrt(6.0 / (inputs + outputs));
            for (int j = 0; j < outputs; j++)
                for (int i = 0; i < inputs; i++)
                    weights[l][j, i] = (random.NextDouble() * 2 - 1) * limit;
        }
    }

    // Returns the activations of every layer, the last one being the output probabilities
    private double[][] Forward(double[] input)
    {
        double[][] activations = new double[layerSizes.Length][];
        activations[0] = input;

        for (int l = 0; l < weights.Length; l++)
        {
            double[] previous = activations[l];
            double[] current = new double[layerSizes[l + 1]];
            for (int j = 0; j < current.Length; j++)
            {
                double sum = biases[l][j];
                for (int i = 0; i < previous.Length; i++)
                    sum += weights[l][j, i] * previous[i];
                current[j] = sum;
            }

            bool isOutputLayer = l == weights.Length - 1;
            activations[l + 1] = isOutputLayer ? Softmax(current) : current.Select(Sigmoid).ToArray();
        }

        return activations;
    }

    // Probability for each class
    public double[] Predict(double[] input) => Forward(input)[^1];

    public int Classify(double[] input)
    {
        double[] output = Predict(input);
        return Array.IndexOf(output, output.Max());
    }

    // One backpropagation step for a single sample, returns its cross-entropy loss
    public double Train(double[] input, int expectedClass, double learningRate)
    {
        double[][] activations = Forward(input);

        // Softmax + cross-entropy: the error of the output layer is simply output - expected
        double[] delta = (double[])activations[^1].Clone();
        double loss = -Math.Log(Math.Max(delta[expectedClass], 1e-12));
        delta[expectedClass] -= 1;

        for (int l = weights.Length - 1; l >= 0; l--)
        {
            double[] previous = activations[l];

            // Error of the previous layer, computed with the weights before they are updated
            double[] previousDelta = new double[previous.Length];
            if (l > 0)
            {
                for (int i = 0; i < previous.Length; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < delta.Length; j++)
                        sum += weights[l][j, i] * delta[j];
                    // Derivative of the sigmoid: s * (1 - s)
                    previousDelta[i] = sum * previous[i] * (1 - previous[i]);
                }
            }

            for (int j = 0; j < delta.Length; j++)
            {
                for (int i = 0; i < previous.Length; i++)
                    weights[l][j, i] -= learningRate * delta[j] * previous[i];
                biases[l][j] -= learningRate * delta[j];
            }

            delta = previousDelta;
        }

        return loss;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double[] Softmax(double[] values)
    {
        // Subtract the maximum so Math.Exp can't overflow
        double max = values.Max();
        double[] exp = values.Select(v => Math.Exp(v - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }
}
