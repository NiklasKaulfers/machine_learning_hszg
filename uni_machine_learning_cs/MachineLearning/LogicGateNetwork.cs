namespace uni_machine_learning_cs.MachineLearning;

// Differentiable logic gate network (Petersen et al., 2022, "Deep Differentiable Logic Gate Networks").
// Every neuron is a logic gate with exactly two inputs, which are picked at random when the network is built.
// What is learned is which of the 16 possible two-input gates (AND, OR, XOR, NAND, ...) each neuron is.
//
// A real gate can't be trained with gradients (its output jumps between 0 and 1), so during training every
// neuron is a mix of all 16 gates, weighted by a softmax over 16 learned values, and the gates work on
// probabilities: AND(a, b) = a * b, OR(a, b) = a + b - a * b, NOT a = 1 - a, ...
// After training every neuron simply becomes its most likely gate. With yes/no inputs (--binary)
// the network is then a pure logic circuit.
//
// The gates of the last layer are split into one group per class, a class's score is how many of its
// gates output 1 (the sum of their outputs).
public class LogicGateNetwork : IClassifier
{
    public const int GateTypes = 16;

    public static readonly string[] GateNames =
    [
        "FALSE", "A AND B", "A AND NOT B", "A", "NOT A AND B", "B", "A XOR B", "A OR B",
        "A NOR B", "A XNOR B", "NOT B", "A OR NOT B", "NOT A", "NOT A OR B", "A NAND B", "TRUE",
    ];

    // The same gates with {0} = input A and {1} = input B, for writing out the circuit
    private static readonly string[] GateFormats =
    [
        "FALSE", "{0} AND {1}", "{0} AND NOT {1}", "{0}", "NOT {0} AND {1}", "{1}", "{0} XOR {1}", "{0} OR {1}",
        "NOT ({0} OR {1})", "NOT ({0} XOR {1})", "NOT {1}", "{0} OR NOT {1}", "NOT {0}", "NOT {0} OR {1}",
        "NOT ({0} AND {1})", "TRUE",
    ];

    // Every gate written as c0 + cA * a + cB * b + cAB * a * b, which is exact for a, b in {0, 1}
    // and the probability of the gate being 1 when a and b are probabilities
    private static readonly double[] C0 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    private static readonly double[] CA = [0, 0, 1, 1, 0, 0, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0];
    private static readonly double[] CB = [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0];
    private static readonly double[] CAB = [0, 1, -1, 0, -1, 0, -2, -1, 1, 2, 0, 1, 0, 1, -1, 0];

    private readonly int inputCount;
    private readonly int classCount;
    private readonly double temperature;
    // inputA[l][g], inputB[l][g]: the two neurons of the previous layer (or inputs) gate g of layer l reads
    private readonly int[][] inputA;
    private readonly int[][] inputB;
    // chosen[l][g]: the gate type that gate g of layer l is after training (the most likely one)
    private readonly int[][] chosen;

    // Only while training: logits[l][g * 16 + type], plus the Adam optimizer state for every logit
    private readonly double[][]? logits;
    private readonly double[][]? adamM;
    private readonly double[][]? adamV;
    private int adamStep;

    public int Layers => inputA.Length;
    public int GatesPerLayer => inputA[0].Length;

    // gatesPerLayer is rounded up to a multiple of classCount, so every class gets the same number of gates
    public LogicGateNetwork(int inputCount, int classCount, int layers, int gatesPerLayer, int seed = 42)
    {
        if (layers < 1)
            throw new ArgumentOutOfRangeException(nameof(layers), "A gate network needs at least one layer.");

        gatesPerLayer = (gatesPerLayer + classCount - 1) / classCount * classCount;
        this.inputCount = inputCount;
        this.classCount = classCount;
        // Scales the group sums before the softmax: without it, the score differences of hundreds of gates
        // would make the network extremely sure of itself right from the start
        temperature = Math.Sqrt(gatesPerLayer / (double)classCount);

        Random random = new Random(seed);
        inputA = new int[layers][];
        inputB = new int[layers][];
        chosen = new int[layers][];
        logits = new double[layers][];
        adamM = new double[layers][];
        adamV = new double[layers][];

        for (int l = 0; l < layers; l++)
        {
            int previousCount = l == 0 ? inputCount : gatesPerLayer;
            inputA[l] = new int[gatesPerLayer];
            inputB[l] = new int[gatesPerLayer];
            chosen[l] = new int[gatesPerLayer];
            logits[l] = new double[gatesPerLayer * GateTypes];
            adamM[l] = new double[gatesPerLayer * GateTypes];
            adamV[l] = new double[gatesPerLayer * GateTypes];

            // Inputs are taken from shuffled lists of the previous neurons, so every previous neuron is used
            // (as long as there are enough gates) instead of some being picked many times and some never
            int[] order = [];
            for (int g = 0; g < gatesPerLayer; g++)
            {
                if (2 * g % previousCount == 0)
                    order = Enumerable.Range(0, previousCount).OrderBy(_ => random.Next()).ToArray();
                inputA[l][g] = order[2 * g % previousCount];
                inputB[l][g] = order[(2 * g + 1) % previousCount];
            }

            // Random start, so not all gates start as the same mix (normal distribution, Box-Muller)
            for (int i = 0; i < logits[l].Length; i++)
                logits[l][i] = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        }

        UpdateChosenGates();
    }

    // Used when loading a saved network, which only needs the chosen gates
    private LogicGateNetwork(int inputCount, int classCount, double temperature, int[][] inputA, int[][] inputB,
        int[][] chosen)
    {
        this.inputCount = inputCount;
        this.classCount = classCount;
        this.temperature = temperature;
        this.inputA = inputA;
        this.inputB = inputB;
        this.chosen = chosen;
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write(inputCount);
        writer.Write(classCount);
        writer.Write(temperature);
        writer.Write(Layers);
        for (int l = 0; l < Layers; l++)
        {
            writer.WriteArray(inputA[l]);
            writer.WriteArray(inputB[l]);
            writer.WriteArray(chosen[l]);
        }
    }

    public static LogicGateNetwork Read(BinaryReader reader)
    {
        int inputCount = reader.ReadInt32();
        int classCount = reader.ReadInt32();
        double temperature = reader.ReadDouble();
        int layers = reader.ReadInt32();
        int[][] inputA = new int[layers][];
        int[][] inputB = new int[layers][];
        int[][] chosen = new int[layers][];
        for (int l = 0; l < layers; l++)
        {
            inputA[l] = reader.ReadIntArray();
            inputB[l] = reader.ReadIntArray();
            chosen[l] = reader.ReadIntArray();
        }
        return new LogicGateNetwork(inputCount, classCount, temperature, inputA, inputB, chosen);
    }

    // Classifies with the chosen gates only (the finished circuit), not the mixes used for training
    public double[] Predict(double[] input)
    {
        double[] values = input;
        for (int l = 0; l < Layers; l++)
        {
            double[] next = new double[GatesPerLayer];
            for (int g = 0; g < next.Length; g++)
            {
                int type = chosen[l][g];
                double a = values[inputA[l][g]];
                double b = values[inputB[l][g]];
                next[g] = C0[type] + CA[type] * a + CB[type] * b + CAB[type] * a * b;
            }
            values = next;
        }
        return Softmax(GroupSums(values));
    }

    // One Adam step on a batch of samples, returns the summed cross-entropy loss of the batch
    public double TrainBatch(IReadOnlyList<(double[] Input, int Class)> batch, double learningRate)
    {
        if (logits == null || adamM == null || adamV == null)
            throw new InvalidOperationException("A loaded gate network can't be trained further.");

        // The softmax over the 16 gate types only changes after the update, so it is computed once per batch,
        // and with it the mixed coefficients of every gate
        double[][] probabilities = new double[Layers][];
        double[][] mixed = new double[Layers][]; // 4 coefficients per gate: c0, cA, cB, cAB
        for (int l = 0; l < Layers; l++)
        {
            probabilities[l] = new double[logits[l].Length];
            mixed[l] = new double[GatesPerLayer * 4];
            for (int g = 0; g < GatesPerLayer; g++)
            {
                Span<double> p = probabilities[l].AsSpan(g * GateTypes, GateTypes);
                SoftmaxInto(logits[l].AsSpan(g * GateTypes, GateTypes), p);
                for (int type = 0; type < GateTypes; type++)
                {
                    mixed[l][g * 4] += p[type] * C0[type];
                    mixed[l][g * 4 + 1] += p[type] * CA[type];
                    mixed[l][g * 4 + 2] += p[type] * CB[type];
                    mixed[l][g * 4 + 3] += p[type] * CAB[type];
                }
            }
        }

        // The samples of the batch are processed on all cores, each thread adds up its own gradients
        double[][] gradients = NewLayerArrays();
        double totalLoss = 0;
        object sync = new object();
        Parallel.ForEach(
            Partition(batch.Count, Environment.ProcessorCount),
            () => (Gradients: NewLayerArrays(), Loss: 0.0),
            (range, _, local) =>
            {
                for (int i = range.Start; i < range.End; i++)
                    local.Loss += Backpropagate(batch[i].Input, batch[i].Class, probabilities, mixed, local.Gradients);
                return local;
            },
            local =>
            {
                lock (sync)
                {
                    totalLoss += local.Loss;
                    for (int l = 0; l < Layers; l++)
                        for (int i = 0; i < gradients[l].Length; i++)
                            gradients[l][i] += local.Gradients[l][i];
                }
            });

        // Adam: every logit gets its own step size from the running mean (m) and variance (v) of its gradient
        const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
        adamStep++;
        double correction1 = 1 - Math.Pow(beta1, adamStep);
        double correction2 = 1 - Math.Pow(beta2, adamStep);
        for (int l = 0; l < Layers; l++)
        {
            for (int i = 0; i < logits[l].Length; i++)
            {
                double gradient = gradients[l][i] / batch.Count;
                adamM[l][i] = beta1 * adamM[l][i] + (1 - beta1) * gradient;
                adamV[l][i] = beta2 * adamV[l][i] + (1 - beta2) * gradient * gradient;
                logits[l][i] -= learningRate * (adamM[l][i] / correction1) / (Math.Sqrt(adamV[l][i] / correction2) + epsilon);
            }
        }

        UpdateChosenGates();
        return totalLoss;
    }

    // Forward pass with the gate mixes, then the gradients of the loss for every logit are added to `gradients`
    private double Backpropagate(double[] input, int expectedClass, double[][] probabilities, double[][] mixed,
        double[][] gradients)
    {
        double[][] activations = new double[Layers + 1][];
        activations[0] = input;
        for (int l = 0; l < Layers; l++)
        {
            double[] previous = activations[l];
            double[] current = new double[GatesPerLayer];
            double[] m = mixed[l];
            for (int g = 0; g < current.Length; g++)
            {
                double a = previous[inputA[l][g]];
                double b = previous[inputB[l][g]];
                current[g] = m[g * 4] + m[g * 4 + 1] * a + m[g * 4 + 2] * b + m[g * 4 + 3] * a * b;
            }
            activations[l + 1] = current;
        }

        double[] output = Softmax(GroupSums(activations[^1]));
        double loss = -Math.Log(Math.Max(output[expectedClass], 1e-12));

        // Error of every gate of the last layer: softmax + cross-entropy gives output - expected for the class
        // score, which every gate of the class's group adds to (divided by the temperature)
        int groupSize = GatesPerLayer / classCount;
        double[] delta = new double[GatesPerLayer];
        for (int g = 0; g < delta.Length; g++)
        {
            int gateClass = g / groupSize;
            delta[g] = (output[gateClass] - (gateClass == expectedClass ? 1 : 0)) / temperature;
        }

        for (int l = Layers - 1; l >= 0; l--)
        {
            double[] previous = activations[l];
            double[] previousDelta = l > 0 ? new double[previous.Length] : [];
            double[] m = mixed[l];
            double[] p = probabilities[l];
            double[] gradient = gradients[l];

            for (int g = 0; g < GatesPerLayer; g++)
            {
                double d = delta[g];
                if (d == 0)
                    continue;

                double a = previous[inputA[l][g]];
                double b = previous[inputB[l][g]];
                double outputValue = activations[l + 1][g];

                // Softmax derivative: d output / d logit_k = p_k * (gate_k(a, b) - output)
                int offset = g * GateTypes;
                for (int type = 0; type < GateTypes; type++)
                {
                    double gateValue = C0[type] + CA[type] * a + CB[type] * b + CAB[type] * a * b;
                    gradient[offset + type] += d * p[offset + type] * (gateValue - outputValue);
                }

                if (l > 0)
                {
                    previousDelta[inputA[l][g]] += d * (m[g * 4 + 1] + m[g * 4 + 3] * b);
                    previousDelta[inputB[l][g]] += d * (m[g * 4 + 2] + m[g * 4 + 3] * a);
                }
            }

            delta = previousDelta;
        }

        return loss;
    }

    // How often each gate type was chosen, over all layers
    public int[] GateUsage()
    {
        int[] usage = new int[GateTypes];
        foreach (int[] layer in chosen)
            foreach (int type in layer)
                usage[type]++;
        return usage;
    }

    // The finished circuit as text, one line per gate. inputNames: the name of every input of the network
    public IEnumerable<string> Describe(IReadOnlyList<string> inputNames, IReadOnlyList<string> concepts)
    {
        int groupSize = GatesPerLayer / classCount;
        for (int l = 0; l < Layers; l++)
        {
            yield return l == Layers - 1
                ? $"# Layer {l + 1} (output): a concept's score is the number of its gates that are 1"
                : $"# Layer {l + 1}";
            for (int g = 0; g < GatesPerLayer; g++)
            {
                string a = l == 0 ? $"[{inputNames[inputA[l][g]]}]" : $"L{l}G{inputA[l][g]}";
                string b = l == 0 ? $"[{inputNames[inputB[l][g]]}]" : $"L{l}G{inputB[l][g]}";
                string formula = string.Format(GateFormats[chosen[l][g]], a, b);
                string group = l == Layers - 1 ? $"   -> counts for {concepts[g / groupSize]}" : "";
                yield return $"L{l + 1}G{g} = {formula}{group}";
            }
        }
    }

    private void UpdateChosenGates()
    {
        for (int l = 0; l < Layers; l++)
            for (int g = 0; g < GatesPerLayer; g++)
            {
                ReadOnlySpan<double> row = logits![l].AsSpan(g * GateTypes, GateTypes);
                int best = 0;
                for (int type = 1; type < GateTypes; type++)
                    if (row[type] > row[best])
                        best = type;
                chosen[l][g] = best;
            }
    }

    private double[] GroupSums(double[] lastLayer)
    {
        int groupSize = lastLayer.Length / classCount;
        double[] scores = new double[classCount];
        for (int g = 0; g < lastLayer.Length; g++)
            scores[g / groupSize] += lastLayer[g];
        for (int c = 0; c < classCount; c++)
            scores[c] /= temperature;
        return scores;
    }

    private double[][] NewLayerArrays() => logits!.Select(layer => new double[layer.Length]).ToArray();

    private static IEnumerable<(int Start, int End)> Partition(int count, int parts)
    {
        int size = Math.Max(1, (count + parts - 1) / parts);
        for (int start = 0; start < count; start += size)
            yield return (start, Math.Min(start + size, count));
    }

    private static double[] Softmax(double[] values)
    {
        double[] result = new double[values.Length];
        SoftmaxInto(values, result);
        return result;
    }

    private static void SoftmaxInto(ReadOnlySpan<double> values, Span<double> result)
    {
        // Subtract the maximum so Math.Exp can't overflow
        double max = double.MinValue;
        foreach (double value in values)
            max = Math.Max(max, value);
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = Math.Exp(values[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < values.Length; i++)
            result[i] /= sum;
    }
}
