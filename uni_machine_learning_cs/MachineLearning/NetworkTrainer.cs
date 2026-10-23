using System.Text.Json;

namespace uni_machine_learning_cs.MachineLearning;

// Loads the feature vectors from the JSON file, trains a NeuralNetwork on them and reports the accuracy
public static class NetworkTrainer
{
    private const double TestShare = 0.7;
    private const int HiddenNeurons = 32;
    private const int Epochs = 200;
    private const double LearningRate = 0.01;
    private const int Seed = 50;

    // verbose: print the training progress, test accuracy and confusion matrix
    public static TrainedModel Run(string jsonPath, bool verbose = true)
    {
        FeatureFile featureFile = JsonSerializer.Deserialize<FeatureFile>(File.ReadAllText(jsonPath))
                                  ?? throw new InvalidDataException($"'{jsonPath}' contains no feature data.");

        string[] concepts = featureFile.Vectors.Select(v => v.Concept).Distinct().Order().ToArray();
        List<(double[] Input, int Class)> samples = featureFile.Vectors
            .Select(v => (FeatureExtractor.ToNumbers(v.Values), Array.IndexOf(concepts, v.Concept)))
            .ToList();

        // Shuffle, then split into training and test data
        Random random = new Random(Seed);
        samples = samples.OrderBy(_ => random.Next()).ToList();
        int testCount = (int)(samples.Count * TestShare);
        List<(double[] Input, int Class)> test = samples.Take(testCount).ToList();
        List<(double[] Input, int Class)> training = samples.Skip(testCount).ToList();

        // Scale every feature to 0-1, using only the training data to find min / max
        int featureCount = featureFile.Properties.Count;
        double[] min = new double[featureCount];
        double[] max = new double[featureCount];
        for (int f = 0; f < featureCount; f++)
        {
            min[f] = training.Min(s => s.Input[f]);
            max[f] = training.Max(s => s.Input[f]);
        }

        // Some concepts have far fewer images (e.g. 289 right turns vs 5780 left turns). Without balancing,
        // the network learns to never predict them, so smaller concepts are repeated up to the largest one
        int largestClass = training.CountBy(s => s.Class).Max(group => group.Value);
        training = training
            .GroupBy(s => s.Class)
            .SelectMany(group => Enumerable.Repeat(group.ToList(), largestClass / group.Count() + 1)
                .SelectMany(copy => copy)
                .Take(largestClass))
            .ToList();

        NeuralNetwork network = new NeuralNetwork([featureCount, HiddenNeurons, concepts.Length], Seed);
        TrainedModel model = new TrainedModel(network, concepts, min, max);
        training = training.Select(s => (model.Normalize(s.Input), s.Class)).ToList();
        test = test.Select(s => (model.Normalize(s.Input), s.Class)).ToList();

        if (verbose)
        {
            Console.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}");
            Console.WriteLine($"Concepts: {string.Join(", ", concepts)}");
            Console.WriteLine($"Layers: {featureCount} inputs -> {HiddenNeurons} hidden -> {concepts.Length} outputs");
        }

        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            double loss = 0;
            foreach (var (input, expected) in training.OrderBy(_ => random.Next()))
                loss += network.Train(input, expected, LearningRate);

            if (verbose && (epoch == 1 || epoch % 20 == 0))
                Console.WriteLine($"Epoch {epoch,3}: loss {loss / training.Count:F4}, " +
                                  $"training accuracy {Accuracy(network, training):P1}");
        }

        if (verbose)
            PrintEvaluation(network, concepts, test);

        return model;
    }

    private static void PrintEvaluation(NeuralNetwork network, string[] concepts, List<(double[] Input, int Class)> test)
    {
        Console.WriteLine($"Test accuracy: {Accuracy(network, test):P1}");

        // Confusion matrix: rows = actual concept, columns = predicted concept
        int[,] confusion = new int[concepts.Length, concepts.Length];
        foreach (var (input, expected) in test)
            confusion[expected, network.Classify(input)]++;

        Console.WriteLine("Confusion matrix (rows = actual, columns = predicted):");
        Console.WriteLine($"{"",12}" + string.Concat(concepts.Select(c => $"{c,12}")));
        for (int a = 0; a < concepts.Length; a++)
        {
            Console.Write($"{concepts[a],12}");
            for (int p = 0; p < concepts.Length; p++)
                Console.Write($"{confusion[a, p],12}");
            Console.WriteLine();
        }
    }

    // Classify only reads the network, so the samples can be checked in parallel
    private static double Accuracy(NeuralNetwork network, List<(double[] Input, int Class)> samples) =>
        samples.AsParallel().Count(s => network.Classify(s.Input) == s.Class) / (double)samples.Count;
}
