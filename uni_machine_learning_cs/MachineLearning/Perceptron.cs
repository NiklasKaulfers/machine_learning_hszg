namespace uni_machine_learning_cs.MachineLearning;

// How the output neurons of the perceptron learn from a wrongly classified vector
public enum PerceptronRule
{
    // Classic perceptron learning rule, every output neuron learns on its own "is it my concept or not?":
    // w += learningRate * (target - output) * x, with target / output being 0 or 1 (step function).
    // Several neurons can fire at once, the one with the largest weighted sum wins
    OneVsRest,

    // Multi-class perceptron: only when the winning neuron (largest weighted sum) is wrong, the correct neuron
    // is moved towards the vector and the wrongly winning one away from it. The neurons compete with each other
    MultiClass,
}

// Single-layer network: every input is connected straight to one output neuron per concept, there are no
// hidden layers. The neurons use the step function and learn with the perceptron learning rule
// (no gradients, no backpropagation). A neuron only changes its weights when it made a mistake.
public class Perceptron : IClassifier
{
    // weights[c, i]: weight from input i to the neuron of concept c
    private readonly double[,] weights;
    private readonly double[] biases;
    public readonly PerceptronRule rule;

    // The weights start at 0. With a start of 0 the learning rate only scales all weights by the same factor,
    // which does not change which neuron wins, so the learning rate has no influence on the result
    public Perceptron(int inputCount, int classCount, PerceptronRule rule)
    {
        weights = new double[classCount, inputCount];
        biases = new double[classCount];
        this.rule = rule;
    }

    private Perceptron(double[,] weights, double[] biases, PerceptronRule rule)
    {
        this.weights = weights;
        this.biases = biases;
        this.rule = rule;
    }

    public int InputCount => weights.GetLength(1);
    public int ClassCount => weights.GetLength(0);

    public double GetWeight(int concept, int input) => weights[concept, input];
    public double GetBias(int concept) => biases[concept];

    // Copy of the current weights, used to keep the best weights seen during training (pocket algorithm)
    public Perceptron Clone() => new Perceptron((double[,])weights.Clone(), (double[])biases.Clone(), rule);

    public void Write(BinaryWriter writer)
    {
        writer.Write((int)rule);
        writer.Write(ClassCount);
        writer.Write(InputCount);
        foreach (double weight in weights)
            writer.Write(weight);
        writer.WriteArray(biases);
    }

    public static Perceptron Read(BinaryReader reader)
    {
        PerceptronRule rule = (PerceptronRule)reader.ReadInt32();
        int classCount = reader.ReadInt32();
        int inputCount = reader.ReadInt32();
        double[,] weights = new double[classCount, inputCount];
        for (int c = 0; c < classCount; c++)
            for (int i = 0; i < inputCount; i++)
                weights[c, i] = reader.ReadDouble();
        return new Perceptron(weights, reader.ReadDoubleArray(), rule);
    }

    // Weighted sum (net input) of every output neuron
    private double[] NetInputs(double[] input)
    {
        double[] net = new double[ClassCount];
        for (int c = 0; c < net.Length; c++)
        {
            double sum = biases[c];
            for (int i = 0; i < input.Length; i++)
                sum += weights[c, i] * input[i];
            net[c] = sum;
        }
        return net;
    }

    // The neuron with the largest weighted sum wins
    public int Classify(double[] input)
    {
        double[] net = NetInputs(input);
        return Array.IndexOf(net, net.Max());
    }

    // A perceptron has no probabilities, only the weighted sums. For printing they are turned into shares with
    // softmax: these show which concepts were close, but are not real probabilities
    public double[] Predict(double[] input)
    {
        double[] net = NetInputs(input);
        double max = net.Max();
        double[] exp = net.Select(n => Math.Exp(n - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }

    // One learning step for a single vector. Returns whether the vector was classified wrongly before the step
    public bool Train(double[] input, int expectedClass, double learningRate)
    {
        double[] net = NetInputs(input);
        int predicted = Array.IndexOf(net, net.Max());

        if (rule == PerceptronRule.MultiClass)
        {
            if (predicted != expectedClass)
            {
                Update(expectedClass, input, learningRate);
                Update(predicted, input, -learningRate);
            }
            return predicted != expectedClass;
        }

        // One-vs-rest: each neuron fires (output 1) when its weighted sum is above 0
        for (int c = 0; c < net.Length; c++)
        {
            int target = c == expectedClass ? 1 : 0;
            int output = net[c] > 0 ? 1 : 0;
            if (target != output)
                Update(c, input, learningRate * (target - output));
        }
        return predicted != expectedClass;
    }

    // w += step * x, the bias is a weight whose input is always 1
    private void Update(int concept, double[] input, double step)
    {
        for (int i = 0; i < input.Length; i++)
            weights[concept, i] += step * input[i];
        biases[concept] += step;
    }
}
