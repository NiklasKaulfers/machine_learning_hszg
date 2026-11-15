using System.Text.Json;

namespace uni_machine_learning_cs.MachineLearning;

public enum LearningMethod
{
    NeuralNetwork,
    KNearestNeighbors,
    DecisionTree,
    LogicGateNetwork,
    Perceptron,
}

// Variants of the perceptron that can be compared with each other.
// Pocket: keep the weights with the best training accuracy seen after any epoch instead of the last ones. Without
// it, the weights keep jumping around if the concepts can't be separated by straight lines (hyperplanes)
public record PerceptronSettings(PerceptronRule Rule = PerceptronRule.MultiClass, bool Pocket = false)
{
    public string Name => (Rule == PerceptronRule.MultiClass ? "multiclass" : "one_vs_rest") + (Pocket ? "_pocket" : "");
}

// Loads the feature vectors from the JSON file, trains a classifier on them with the chosen learning method
// and reports the accuracy
public static class ModelTrainer
{
    // Share of the images used for training, the rest is test data
    private const double TrainingShare = 0.7;
    private const int Seed = 50;

    // Neural network: HiddenLayers hidden layers with HiddenNeurons neurons each
    private const int HiddenLayers = 1;
    private const int HiddenNeurons = 32;
    private const int Epochs = 200;
    private const double LearningRate = 0.01;

    // k-nearest neighbors: number of neighbors that vote, odd so two concepts tie less often
    private const int Neighbors = 5;

    // Decision tree
    private const int MaxDepth = 20;
    private const int MinSamplesSplit = 5;

    // Logic gate network: GateLayers layers with GatesPerLayer gates each, trained in batches with Adam
    private const int GateLayers = 4;
    private const int GatesPerLayer = 1000;
    private const int GateEpochs = 50;
    private const int GateBatchSize = 100;
    private const double GateLearningRate = 0.01;

    // Perceptron: maximum number of passes over the training data (stops earlier if an epoch has no mistakes).
    // The learning rate makes no difference because the weights start at 0 (see Perceptron)
    private const int PerceptronEpochs = 100;
    private const double PerceptronLearningRate = 1;

    // Prints the training progress, test accuracy and confusion matrix.
    // binary: every feature is turned into yes/no (above its average in the training data or not) first
    public static TrainedModel Run(string jsonPath, LearningMethod method, bool binary,
        PerceptronSettings? perceptron = null)
    {
        var (concepts, samples) = Load(jsonPath);
        return Train(concepts, samples, method, Seed, Console.Out, DefaultComplexity(method), binary,
            perceptron ?? new PerceptronSettings());
    }

    // The setting that makes each method simpler or more complex:
    // neural network = number of hidden layers (its depth), k-NN = k, decision tree = max depth,
    // logic gate network = number of gate layers, perceptron = number of epochs
    public static string ComplexityName(LearningMethod method) => method switch
    {
        LearningMethod.NeuralNetwork => "hidden layers",
        LearningMethod.KNearestNeighbors => "k",
        LearningMethod.LogicGateNetwork => "gate layers",
        LearningMethod.Perceptron => "epochs",
        _ => "max depth",
    };

    public static int DefaultComplexity(LearningMethod method) => method switch
    {
        LearningMethod.NeuralNetwork => HiddenLayers,
        LearningMethod.KNearestNeighbors => Neighbors,
        LearningMethod.LogicGateNetwork => GateLayers,
        LearningMethod.Perceptron => PerceptronEpochs,
        _ => MaxDepth,
    };

    // Values tried by Sweep when none are given
    public static int[] DefaultSweepValues(LearningMethod method) => method switch
    {
        LearningMethod.NeuralNetwork => [0, 1, 2, 3, 4, 5],
        LearningMethod.KNearestNeighbors => [1, 3, 5, 7, 9, 15, 21, 31, 51, 101],
        LearningMethod.LogicGateNetwork => [1, 2, 3, 4, 6, 8],
        LearningMethod.Perceptron => [1, 2, 5, 10, 20, 50, 100, 200],
        _ => [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30],
    };

    // Trains and tests the method `runs` times for every complexity value (seeds 1..runs, so every value is
    // tested on the same splits). Shows how the depth / k changes the test accuracy.
    // Returns the test accuracies of every run for every value
    public static List<(int Value, List<double> Accuracies)> Sweep(string jsonPath, LearningMethod method,
        int[] values, int runs, bool binary, PerceptronSettings? perceptron = null)
    {
        perceptron ??= new PerceptronSettings();
        var (concepts, samples) = Load(jsonPath);
        string name = ComplexityName(method);
        Console.WriteLine($"Learning method: {method}, {name} = {string.Join(", ", values)}, {runs} runs each");

        List<(int Value, List<double> Accuracies)> results = [];
        foreach (int value in values)
        {
            // The runs don't depend on each other, so they are trained on all CPU cores at once
            // (mostly for the neural network, whose training uses a single core)
            List<double> accuracies = Enumerable.Range(1, runs)
                .AsParallel()
                .AsOrdered()
                .Select(run => Train(concepts, samples, method, run, TextWriter.Null, value, binary, perceptron)
                    .testAccuracy)
                .ToList();

            double mean = accuracies.Average();
            double standardDeviation = Math.Sqrt(accuracies.Average(a => (a - mean) * (a - mean)));
            Console.WriteLine($"{name} = {value,3}: mean {mean:P2}, standard deviation {standardDeviation:P2}, " +
                              $"min {accuracies.Min():P2}, max {accuracies.Max():P2}");
            results.Add((value, accuracies));
        }
        return results;
    }

    // Trains and tests the method `runs` times, each time with a different random split into training and
    // test data (seed 1, 2, ...). Shows how much the accuracy depends on which images end up in the test data.
    // Returns the test accuracy of every run
    public static List<double> RunRepeated(string jsonPath, LearningMethod method, int runs, bool binary,
        PerceptronSettings? perceptron = null)
    {
        perceptron ??= new PerceptronSettings();
        var (concepts, samples) = Load(jsonPath);
        Console.WriteLine($"Learning method: {method}, {runs} runs with different training / test splits");

        // The runs don't depend on each other, so they are trained on all CPU cores at once (see Sweep)
        List<double> accuracies = Enumerable.Range(1, runs)
            .AsParallel()
            .AsOrdered()
            .Select(run => Train(concepts, samples, method, run, TextWriter.Null, DefaultComplexity(method), binary,
                perceptron).testAccuracy)
            .ToList();

        PrintRuns(accuracies);
        return accuracies;
    }

    // Tests an already trained model `runs` times on the test data of the same random splits that RunRepeated
    // uses (seed 1, 2, ...), so its accuracies can be compared run by run with freshly trained models.
    // Nothing is trained: this shows how good the model's current weights (e.g. edited by hand) are.
    // Note that the model may have seen some of these test images while it was trained on its own split.
    // Returns the test accuracy of every run
    public static List<double> EvaluateRepeated(string jsonPath, TrainedModel model, int runs)
    {
        var (concepts, samples) = Load(jsonPath);
        if (!concepts.SequenceEqual(model.concepts))
            throw new InvalidDataException("The model knows different concepts than the feature file.");
        Console.WriteLine($"Testing the saved {model.method} model {runs} times on different test splits");

        // Rows = actual concept, columns = predicted concept, summed over all runs
        int[,] confusion = new int[concepts.Length, concepts.Length];
        List<double> accuracies = [];
        for (int run = 1; run <= runs; run++)
        {
            var (_, test) = Split(samples, run);
            var predictions = test.AsParallel().Select(s => (s.Class, model.Classify(s.Input))).ToList();
            foreach (var (actual, predicted) in predictions)
                confusion[actual, predicted]++;
            accuracies.Add(predictions.Count(p => p.Class == p.Item2) / (double)test.Count);
        }

        PrintRuns(accuracies);
        Console.WriteLine("Confusion matrix of all runs (rows = actual, columns = predicted):");
        PrintConfusion(confusion, concepts, Console.Out);
        return accuracies;
    }

    private static void PrintRuns(List<double> accuracies)
    {
        for (int run = 0; run < accuracies.Count; run++)
            Console.WriteLine($"Run {run + 1,3}: test accuracy {accuracies[run]:P2}");

        double mean = accuracies.Average();
        double standardDeviation = Math.Sqrt(accuracies.Average(a => (a - mean) * (a - mean)));
        Console.WriteLine($"Mean {mean:P2}, standard deviation {standardDeviation:P2}, " +
                          $"min {accuracies.Min():P2}, max {accuracies.Max():P2}");
    }

    private static (string[] Concepts, List<(double[] Input, int Class)> Samples) Load(string jsonPath)
    {
        FeatureFile featureFile = JsonSerializer.Deserialize<FeatureFile>(File.ReadAllText(jsonPath))
                                  ?? throw new InvalidDataException($"'{jsonPath}' contains no feature data.");

        string[] concepts = featureFile.Vectors.Select(v => v.Concept).Distinct().Order().ToArray();
        List<(double[] Input, int Class)> samples = featureFile.Vectors
            .Select(v => (FeatureExtractor.ToNumbers(v.Values), Array.IndexOf(concepts, v.Concept)))
            .ToList();
        return (concepts, samples);
    }

    // log: where the training progress and evaluation are written (TextWriter.Null to stay quiet)
    // complexity: hidden layers (neural network), k (k-NN) or max depth (decision tree), see ComplexityName
    private static TrainedModel Train(string[] concepts, List<(double[] Input, int Class)> samples,
        LearningMethod method, int seed, TextWriter log, int complexity, bool binary, PerceptronSettings perceptron)
    {
        // Shuffle, then split into training and test data
        Random random = new Random(seed);
        var (training, test) = Split(samples, seed, random);

        // Scale every feature to 0-1, using only the training data to find min / max
        int featureCount = samples[0].Input.Length;
        double[] min = new double[featureCount];
        double[] max = new double[featureCount];
        for (int f = 0; f < featureCount; f++)
        {
            min[f] = training.Min(s => s.Input[f]);
            max[f] = training.Max(s => s.Input[f]);
        }
        training = training.Select(s => (TrainedModel.Normalize(s.Input, min, max), s.Class)).ToList();
        test = test.Select(s => (TrainedModel.Normalize(s.Input, min, max), s.Class)).ToList();

        // Yes/no features: "is this feature above its average in the training images?". The average works for
        // features that already are 0 / 1 (colour present) as well, a median could be 0 or 1 there
        double[]? thresholds = null;
        if (binary)
        {
            thresholds = Enumerable.Range(0, featureCount).Select(f => training.Average(s => s.Input[f])).ToArray();
            training = training.Select(s => (TrainedModel.Binarize(s.Input, thresholds), s.Class)).ToList();
            test = test.Select(s => (TrainedModel.Binarize(s.Input, thresholds), s.Class)).ToList();
        }

        log.WriteLine($"Learning method: {method}{(binary ? ", yes/no features" : "")}");
        log.WriteLine($"Concepts: {string.Join(", ", concepts)}");

        IClassifier classifier = method switch
        {
            LearningMethod.NeuralNetwork => TrainNeuralNetwork(training, test, featureCount, concepts.Length,
                complexity, random, seed, log),
            LearningMethod.KNearestNeighbors => CreateKNearestNeighbors(training, test, concepts.Length, complexity, log),
            LearningMethod.DecisionTree => TrainDecisionTree(training, test, concepts.Length, complexity, log),
            LearningMethod.LogicGateNetwork => TrainLogicGateNetwork(training, test, featureCount, concepts.Length,
                complexity, random, seed, log),
            LearningMethod.Perceptron => TrainPerceptron(training, test, featureCount, concepts.Length, complexity,
                perceptron, random, log),
            _ => throw new ArgumentOutOfRangeException(nameof(method), method, null),
        };

        double testAccuracy = Accuracy(classifier, test);
        // The confusion matrix classifies the test data a second time, skipped when nothing is written
        if (log != TextWriter.Null)
            PrintEvaluation(classifier, concepts, test, testAccuracy, log);

        return new TrainedModel(classifier, method, concepts, min, max, thresholds, DateTime.Now, testAccuracy);
    }

    // Shuffles the samples with the seed and splits them into training and test data. The same seed always
    // gives the same split. random: the generator to keep using afterwards (it has drawn one number per sample)
    private static (List<(double[] Input, int Class)> Training, List<(double[] Input, int Class)> Test) Split(
        List<(double[] Input, int Class)> samples, int seed, Random? random = null)
    {
        random ??= new Random(seed);
        samples = samples.OrderBy(_ => random.Next()).ToList();
        int trainingCount = (int)(samples.Count * TrainingShare);
        return (samples.Take(trainingCount).ToList(), samples.Skip(trainingCount).ToList());
    }

    private static NeuralNetwork TrainNeuralNetwork(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int featureCount, int classCount, int hiddenLayers, Random random,
        int seed, TextWriter log)
    {
        training = Balance(training);

        // 0 hidden layers: inputs connected straight to the outputs (multinomial logistic regression)
        int[] layerSizes = [featureCount, ..Enumerable.Repeat(HiddenNeurons, hiddenLayers), classCount];
        NeuralNetwork network = new NeuralNetwork(layerSizes, seed);

        log.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}");
        log.WriteLine($"Layers: {featureCount} inputs -> " +
                      string.Concat(Enumerable.Repeat($"{HiddenNeurons} hidden -> ", hiddenLayers)) +
                      $"{classCount} outputs");

        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            double loss = 0;
            foreach (var (input, expected) in training.OrderBy(_ => random.Next()))
                loss += network.Train(input, expected, LearningRate);

            if (log != TextWriter.Null && (epoch == 1 || epoch % 20 == 0))
                log.WriteLine($"Epoch {epoch,3}: loss {loss / training.Count:F4}, " +
                                  $"training accuracy {Accuracy(network, training):P1}");
        }

        return network;
    }

    // Perceptron learning rule: go through the (shuffled) training vectors, a neuron changes its weights only when
    // it was wrong. Stops early once a whole epoch has no mistakes: then the training data is linearly separable
    // and the perceptron convergence theorem says it found separating weights
    private static Perceptron TrainPerceptron(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int featureCount, int classCount, int epochs,
        PerceptronSettings settings, Random random, TextWriter log)
    {
        training = Balance(training);
        Perceptron perceptron = new Perceptron(featureCount, classCount, settings.Rule);
        Perceptron best = perceptron.Clone();
        double bestAccuracy = -1;

        log.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}");
        log.WriteLine($"Single layer: {featureCount} inputs -> {classCount} output neurons, " +
                      $"rule {settings.Rule}, pocket {(settings.Pocket ? "on" : "off")}, up to {epochs} epochs");

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            int mistakes = training.OrderBy(_ => random.Next())
                .Count(sample => perceptron.Train(sample.Input, sample.Class, PerceptronLearningRate));

            double accuracy = settings.Pocket || log != TextWriter.Null ? Accuracy(perceptron, training) : 0;
            if (settings.Pocket && accuracy > bestAccuracy)
            {
                bestAccuracy = accuracy;
                best = perceptron.Clone();
            }

            if (log != TextWriter.Null && (epoch == 1 || epoch % 10 == 0 || mistakes == 0))
                log.WriteLine($"Epoch {epoch,3}: {mistakes} mistakes while learning, " +
                              $"training accuracy {accuracy:P1}");

            if (mistakes == 0)
            {
                log.WriteLine($"No mistakes in epoch {epoch}, the training data is linearly separable");
                break;
            }
        }

        if (settings.Pocket)
            log.WriteLine($"Pocket: using the weights with training accuracy {bestAccuracy:P1}");
        return settings.Pocket ? best : perceptron;
    }

    private static LogicGateNetwork TrainLogicGateNetwork(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int featureCount, int classCount, int layers, Random random,
        int seed, TextWriter log)
    {
        training = Balance(training);
        LogicGateNetwork network = new LogicGateNetwork(featureCount, classCount, layers, GatesPerLayer, seed);

        log.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}");
        log.WriteLine($"Layers: {featureCount} inputs -> {layers} x {network.GatesPerLayer} gates " +
                      $"-> {classCount} groups of {network.GatesPerLayer / classCount}");

        for (int epoch = 1; epoch <= GateEpochs; epoch++)
        {
            List<(double[] Input, int Class)> shuffled = training.OrderBy(_ => random.Next()).ToList();
            double loss = 0;
            for (int start = 0; start < shuffled.Count; start += GateBatchSize)
                loss += network.TrainBatch(shuffled.GetRange(start, Math.Min(GateBatchSize, shuffled.Count - start)),
                    GateLearningRate);

            // The accuracy uses the finished gates (one per neuron), the loss the gate mixes used for training
            if (log != TextWriter.Null && (epoch == 1 || epoch % 5 == 0))
                log.WriteLine($"Epoch {epoch,3}: loss {loss / training.Count:F4}, " +
                              $"training accuracy {Accuracy(network, training):P1}");
        }

        if (log != TextWriter.Null)
        {
            int[] usage = network.GateUsage();
            log.WriteLine("Chosen gates: " + string.Join(", ", Enumerable.Range(0, usage.Length)
                .Where(type => usage[type] > 0)
                .OrderByDescending(type => usage[type])
                .Select(type => $"{LogicGateNetwork.GateNames[type]} {usage[type]}")));
        }

        return network;
    }

    // Some concepts have far fewer images (e.g. 289 right turns vs 5780 left turns). Without balancing,
    // a network learns to never predict them, so smaller concepts are repeated up to the largest one
    private static List<(double[] Input, int Class)> Balance(List<(double[] Input, int Class)> training)
    {
        int largestClass = training.CountBy(s => s.Class).Max(group => group.Value);
        return training
            .GroupBy(s => s.Class)
            .SelectMany(group => Enumerable.Repeat(group.ToList(), largestClass / group.Count() + 1)
                .SelectMany(copy => copy)
                .Take(largestClass))
            .ToList();
    }

    // k-NN has no training phase, the (normalized) training vectors are the model. They are not balanced
    // like for the network: copies of a vector would only fill several of the k places with the same neighbor
    private static KNearestNeighbors CreateKNearestNeighbors(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int classCount, int k, TextWriter log)
    {
        log.WriteLine($"Storing {training.Count} vectors, testing on {test.Count}, k = {k}");

        return new KNearestNeighbors(training, classCount, k);
    }

    private static DecisionTree TrainDecisionTree(List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> test, int classCount, int maxDepth, TextWriter log)
    {
        log.WriteLine($"Training on {training.Count} vectors, testing on {test.Count}, " +
                              $"max depth {maxDepth}, min {MinSamplesSplit} vectors per split");

        DecisionTree tree = new DecisionTree(training, classCount, maxDepth, MinSamplesSplit);

        if (log != TextWriter.Null)
            log.WriteLine($"Tree: depth {tree.Depth}, {tree.LeafCount} leaves, " +
                              $"training accuracy {Accuracy(tree, training):P1}");

        return tree;
    }

    private static void PrintEvaluation(IClassifier classifier, string[] concepts, List<(double[] Input, int Class)> test,
        double testAccuracy, TextWriter log)
    {
        log.WriteLine($"Test accuracy: {testAccuracy:P1}");

        // Confusion matrix: rows = actual concept, columns = predicted concept
        int[,] confusion = new int[concepts.Length, concepts.Length];
        foreach (var (actual, predicted) in test.AsParallel().Select(s => (s.Class, classifier.Classify(s.Input))).ToList())
            confusion[actual, predicted]++;

        log.WriteLine("Confusion matrix (rows = actual, columns = predicted):");
        PrintConfusion(confusion, concepts, log);
    }

    private static void PrintConfusion(int[,] confusion, string[] concepts, TextWriter log)
    {
        log.WriteLine($"{"",12}" + string.Concat(concepts.Select(c => $"{c,12}")));
        for (int a = 0; a < concepts.Length; a++)
        {
            log.Write($"{concepts[a],12}");
            for (int p = 0; p < concepts.Length; p++)
                log.Write($"{confusion[a, p],12}");
            log.WriteLine();
        }
    }

    // Classify only reads the classifier, so the samples can be checked in parallel
    private static double Accuracy(IClassifier classifier, List<(double[] Input, int Class)> samples) =>
        samples.AsParallel().Count(s => classifier.Classify(s.Input) == s.Class) / (double)samples.Count;
}
