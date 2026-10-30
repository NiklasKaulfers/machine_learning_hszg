using System.Text.Json;

namespace uni_machine_learning_cs.MachineLearning;

public enum LearningMethod
{
    NeuralNetwork,
    KNearestNeighbors,
    DecisionTree,
}

// Loads the feature vectors from the JSON file, trains a classifier on them with the chosen learning method
// and reports the accuracy
public static class ModelTrainer
{
    private const double TestShare = 0.7;
    private const int Seed = 50;

    // Neural network
    private const int HiddenNeurons = 32;
    private const int Epochs = 200;
    private const double LearningRate = 0.01;

    // k-nearest neighbors: number of neighbors that vote, odd so two concepts tie less often
    private const int Neighbors = 5;

    // Decision tree
    private const int MaxDepth = 20;
    private const int MinSamplesSplit = 5;

    // Prints the training progress, test accuracy and confusion matrix
    public static TrainedModel Run(string jsonPath, LearningMethod method)
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
        training = training.Select(s => (TrainedModel.Normalize(s.Input, min, max), s.Class)).ToList();
        test = test.Select(s => (TrainedModel.Normalize(s.Input, min, max), s.Class)).ToList();

        Console.WriteLine($"Learning method: {method}");
        Console.WriteLine($"Concepts: {string.Join(", ", concepts)}");

        IClassifier classifier = method switch
        {
            LearningMethod.NeuralNetwork => TrainNeuralNetwork(training, test, featureCount, concepts.Length, random),
            LearningMethod.KNearestNeighbors => CreateKNearestNeighbors(training, test, concepts.Length),
            LearningMethod.DecisionTree => TrainDecisionTree(training, test, concepts.Length),
            _ => throw new ArgumentOutOfRangeException(nameof(method), method, null),
        };

        double testAccuracy = Accuracy(classifier, test);
        PrintEvaluation(classifier, concepts, test, testAccuracy);

        return new TrainedModel(classifier, method, concepts, min, max, DateTime.Now, testAccuracy);
    }

    private static NeuralNetwork TrainNeuralNetwork(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int featureCount, int classCount, Random random)
    {
        // Some concepts have far fewer images (e.g. 289 right turns vs 5780 left turns). Without balancing,
        // the network learns to never predict them, so smaller concepts are repeated up to the largest one
        int largestClass = training.CountBy(s => s.Class).Max(group => group.Value);
        training = training
            .GroupBy(s => s.Class)
            .SelectMany(group => Enumerable.Repeat(group.ToList(), largestClass / group.Count() + 1)
                .SelectMany(copy => copy)
                .Take(largestClass))
            .ToList();

        NeuralNetwork network = new NeuralNetwork([featureCount, HiddenNeurons, classCount], Seed);

        Console.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}");
        Console.WriteLine($"Layers: {featureCount} inputs -> {HiddenNeurons} hidden -> {classCount} outputs");

        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            double loss = 0;
            foreach (var (input, expected) in training.OrderBy(_ => random.Next()))
                loss += network.Train(input, expected, LearningRate);

            if (epoch == 1 || epoch % 20 == 0)
                Console.WriteLine($"Epoch {epoch,3}: loss {loss / training.Count:F4}, " +
                                  $"training accuracy {Accuracy(network, training):P1}");
        }

        return network;
    }

    // k-NN has no training phase, the (normalized) training vectors are the model. They are not balanced
    // like for the network: copies of a vector would only fill several of the k places with the same neighbor
    private static KNearestNeighbors CreateKNearestNeighbors(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int classCount)
    {
        Console.WriteLine($"Storing {training.Count} vectors, testing on {test.Count}, k = {Neighbors}");

        return new KNearestNeighbors(training, classCount, Neighbors);
    }

    private static DecisionTree TrainDecisionTree(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int classCount)
    {
        Console.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}, " +
                              $"max depth {MaxDepth}, min {MinSamplesSplit} vectors per split");

        DecisionTree tree = new DecisionTree(training, classCount, MaxDepth, MinSamplesSplit);

        Console.WriteLine($"Tree: depth {tree.Depth}, {tree.LeafCount} leaves, " +
                              $"training accuracy {Accuracy(tree, training):P1}");

        return tree;
    }

    private static void PrintEvaluation(IClassifier classifier, string[] concepts, List<(double[] Input, int Class)> test,
        double testAccuracy)
    {
        Console.WriteLine($"Test accuracy: {testAccuracy:P1}");

        // Confusion matrix: rows = actual concept, columns = predicted concept
        int[,] confusion = new int[concepts.Length, concepts.Length];
        foreach (var (actual, predicted) in test.AsParallel().Select(s => (s.Class, classifier.Classify(s.Input))).ToList())
            confusion[actual, predicted]++;

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

    // Classify only reads the classifier, so the samples can be checked in parallel
    private static double Accuracy(IClassifier classifier, List<(double[] Input, int Class)> samples) =>
        samples.AsParallel().Count(s => classifier.Classify(s.Input) == s.Class) / (double)samples.Count;
}
