using System.Text.Json;
using uni_machine_learning_cs.MachineLearning;

// Usage:
//   --train <method> [<image>..]  -> extract features of all images, train the method, save the model,
//                                    then classify the given images (or ask for images to classify)
//   <method> [<image>..]          -> load the saved model of the method and classify with it
//                                    (if there is no saved model yet, it is trained first)
//   --runs <n> <method>           -> train and test the method n times with different random training / test
//                                    splits, print every run's test accuracy and save them to evaluations/
//   --runs <n> <method> --tune    -> (--nn / --perceptron) every run trains from scratch on its split, but keeps
//                                    20 % of the training data back. The weights are then tuned on that part
//                                    (see WeightTuner) and the test accuracy before and after is saved
//   --runs <n> <method> --weights -> don't train: test the saved model (with the weights from its _weights.json
//                                    applied, see below) on the test data of the same n splits. Shows how good
//                                    weights edited by hand are, run by run comparable with --runs without it
//   --sweep <n> <method> [--values 1,2,3]
//                                 -> like --runs, but repeated for several complexity values: the number of
//                                    hidden layers (--nn), k (--knn), the max depth (--tree), the number of gate
//                                    layers (--gates) or the number of epochs (--perceptron). Saves the
//                                    accuracies per value to evaluations/ (without --values, defaults are used)
// <method> is --nn (neural network), --knn (k-nearest neighbors), --tree (decision tree) or
// --gates (logic gate network: every neuron learns to be one of the 16 two-input logic gates) or
// --perceptron (single-layer network trained with the perceptron learning rule).
// Perceptron variants: --ovr uses the classic rule where every output neuron learns "my concept or not" on its own
// (default: multi-class rule, only the correct and the wrongly winning neuron learn), --pocket keeps the weights
// with the best training accuracy instead of the last ones. For --sweep / --runs the value is the number of epochs.
// Every variant is saved as its own model, e.g. perceptron_one_vs_rest_pocket.model
// Without a method, it is asked for. The models are saved in the models folder of the project root.
// --binary (works with every method) turns every feature into yes/no before training: 1 if it is above its
// average in the training images, otherwise 0. Saved as its own model, e.g. neural_network_binary.model.
// Training the gate network also writes models/<model name>_circuit.txt with every gate of the finished circuit
// (with --binary it is a pure logic circuit, otherwise the gates work on the 0-1 values like probabilities)
// Training the neural network or the perceptron also writes models/<model name>_weights.json with the weights
// of every neuron. Changes made in that file are used on every following run, until the model is trained again

// Each folder holds the images of one concept (all subfolders included)
(string Folder, string Concept)[] conceptFolders =
[
    ("RightTurn", "right turn"),
    ("StopSign", "stop sign"),
    ("Vorfahrt von rechts", "right of way"),
    ("Fahrtrichtung links", "left turn"),
    ("Vorfahrt gewähren", "yield"),
];
string[] imageExtensions = [".bmp", ".jpg", ".jpeg"];

const string author = "Niklas Kaulfers";

// Write the output into the project root (the folder with the .sln), not into bin/Debug/...
DirectoryInfo? rootDir = new DirectoryInfo(AppContext.BaseDirectory);
while (rootDir != null && rootDir.GetFiles("*.sln").Length == 0)
    rootDir = rootDir.Parent;
string projectRoot = rootDir?.FullName ?? Directory.GetCurrentDirectory();
string outputFile = Path.Combine(projectRoot, "image_data.json");
string modelFolder = Path.Combine(projectRoot, "models");

string[] options = ["--train", "--runs", "--sweep", "--values", "--binary", "--nn", "--knn", "--tree", "--gates",
    "--perceptron", "--ovr", "--pocket", "--weights", "--tune"];
string? unknownOption = args.FirstOrDefault(arg => arg.StartsWith("--") && !options.Contains(arg));
if (unknownOption != null)
{
    Console.WriteLine($"Unknown option {unknownOption}. Use --train, --runs <n> or --sweep <n>, " +
                      "and one of --nn, --knn, --tree, --gates, --perceptron (optionally with --binary; " +
                      "--perceptron also takes --ovr and --pocket).");
    return 1;
}

int runsIndex = Array.IndexOf(args, "--runs");
int runs = 0;
if (runsIndex >= 0 && (runsIndex + 1 >= args.Length || !int.TryParse(args[runsIndex + 1], out runs) || runs < 1))
{
    Console.WriteLine("--runs needs the number of runs, e.g. --runs 100 --knn");
    return 1;
}

int sweepIndex = Array.IndexOf(args, "--sweep");
if (sweepIndex >= 0 && (sweepIndex + 1 >= args.Length || !int.TryParse(args[sweepIndex + 1], out runs) || runs < 1))
{
    Console.WriteLine("--sweep needs the number of runs per value, e.g. --sweep 10 --knn");
    return 1;
}

int valuesIndex = Array.IndexOf(args, "--values");
int[]? sweepValues = null;
if (valuesIndex >= 0)
{
    try
    {
        sweepValues = args[valuesIndex + 1].Split(',').Select(int.Parse).ToArray();
    }
    catch (Exception e) when (e is IndexOutOfRangeException or FormatException or OverflowException)
    {
        Console.WriteLine("--values needs a comma separated list of numbers, e.g. --values 1,3,5");
        return 1;
    }
}

bool train = args.Contains("--train");
bool binary = args.Contains("--binary");
bool savedWeights = args.Contains("--weights");
bool tune = args.Contains("--tune");
PerceptronSettings perceptron = new PerceptronSettings(
    args.Contains("--ovr") ? PerceptronRule.OneVsRest : PerceptronRule.MultiClass, args.Contains("--pocket"));
LearningMethod method = args.Contains("--perceptron") ? LearningMethod.Perceptron
    : args.Contains("--gates") ? LearningMethod.LogicGateNetwork
    : args.Contains("--tree") ? LearningMethod.DecisionTree
    : args.Contains("--knn") ? LearningMethod.KNearestNeighbors
    : args.Contains("--nn") ? LearningMethod.NeuralNetwork
    : AskLearningMethod();
int[] optionValueIndices = [runsIndex, sweepIndex, valuesIndex];
string[] images = args
    .Where((arg, i) => !options.Contains(arg) && !optionValueIndices.Any(index => index >= 0 && i == index + 1))
    .ToArray();
string modelName = method switch
{
    LearningMethod.NeuralNetwork => "neural_network",
    LearningMethod.KNearestNeighbors => "k_nearest_neighbors",
    LearningMethod.LogicGateNetwork => "logic_gate_network",
    LearningMethod.Perceptron => $"perceptron_{perceptron.Name}",
    _ => "decision_tree",
} + (binary ? "_binary" : "");
string modelFile = Path.Combine(modelFolder, modelName + ".model");
string weightsFile = Path.Combine(modelFolder, $"{modelName}_weights.json");

if (sweepIndex >= 0)
{
    if (!File.Exists(outputFile))
        WriteFeatureFile();

    var results = ModelTrainer.Sweep(outputFile, method, sweepValues ?? ModelTrainer.DefaultSweepValues(method), runs,
        binary, perceptron);

    // One line per value: summary first, then the accuracy of every run, so it can be charted in a spreadsheet
    var culture = System.Globalization.CultureInfo.InvariantCulture;
    string evaluationFolder = Path.Combine(projectRoot, "evaluations");
    string evaluationFile = Path.Combine(evaluationFolder,
        $"{Path.GetFileNameWithoutExtension(modelFile)}_sweep_{runs}_runs_{DateTime.Now:yyyy-MM-dd_HH-mm}.csv");
    Directory.CreateDirectory(evaluationFolder);
    File.WriteAllLines(evaluationFile, [
        $"{ModelTrainer.ComplexityName(method)},mean,standard deviation,min,max," +
        string.Join(",", Enumerable.Range(1, runs).Select(run => $"run {run}")),
        ..results.Select(result =>
        {
            double mean = result.Accuracies.Average();
            double standardDeviation = Math.Sqrt(result.Accuracies.Average(a => (a - mean) * (a - mean)));
            return string.Join(",", [result.Value.ToString(culture),
                ..new[] { mean, standardDeviation, result.Accuracies.Min(), result.Accuracies.Max() }
                    .Concat(result.Accuracies)
                    .Select(value => value.ToString("F4", culture))]);
        })]);
    Console.WriteLine($"Saved the results to {evaluationFile}");
    return 0;
}

if (runs > 0)
{
    if (!File.Exists(outputFile))
        WriteFeatureFile();

    var culture = System.Globalization.CultureInfo.InvariantCulture;
    if (tune)
    {
        if (method is not (LearningMethod.NeuralNetwork or LearningMethod.Perceptron))
        {
            Console.WriteLine("--tune only works with --nn or --perceptron.");
            return 1;
        }
        var tuned = ModelTrainer.TuneRepeated(outputFile, method, runs, binary, perceptron);
        string tuneFolder = Path.Combine(projectRoot, "evaluations");
        string tuneFile = Path.Combine(tuneFolder,
            $"{Path.GetFileNameWithoutExtension(modelFile)}_tuned_{runs}_runs_{DateTime.Now:yyyy-MM-dd_HH-mm}.csv");
        Directory.CreateDirectory(tuneFolder);
        File.WriteAllLines(tuneFile, [
            "run,seed,untuned test accuracy,tuned test accuracy,untuned validation accuracy," +
            "tuned validation accuracy,changes",
            ..tuned.Select((run, i) => string.Join(",", (i + 1).ToString(culture), run.Seed.ToString(culture),
                run.TestUntuned.ToString("F4", culture), run.TestTuned.ToString("F4", culture),
                run.ValidationUntuned.ToString("F4", culture), run.ValidationTuned.ToString("F4", culture),
                $"\"{run.Changes}\""))]);
        Console.WriteLine($"Saved the results to {tuneFile}");
        return 0;
    }

    List<double> accuracies;
    if (savedWeights)
    {
        if (!File.Exists(modelFile))
        {
            Console.WriteLine($"--weights needs a saved model, but there is none in {modelFolder}. Train one first.");
            return 1;
        }
        TrainedModel? saved = LoadModel();
        if (saved == null)
            return 1;
        if (!NetworkWeightsFile.Supports(saved))
        {
            Console.WriteLine("--weights only works with the neural network or the perceptron.");
            return 1;
        }
        if (!ApplyWeights(saved))
            return 1;
        accuracies = ModelTrainer.EvaluateRepeated(outputFile, saved, runs);
    }
    else
    {
        accuracies = ModelTrainer.RunRepeated(outputFile, method, runs, binary, perceptron);
    }

    // One line per run, so the results can be opened in a spreadsheet
    string evaluationFolder = Path.Combine(projectRoot, "evaluations");
    string evaluationFile = Path.Combine(evaluationFolder,
        $"{Path.GetFileNameWithoutExtension(modelFile)}{(savedWeights ? "_weights" : "")}_{runs}_runs_" +
        $"{DateTime.Now:yyyy-MM-dd_HH-mm}.csv");
    Directory.CreateDirectory(evaluationFolder);
    File.WriteAllLines(evaluationFile, ["run,seed,test accuracy",
        ..accuracies.Select((accuracy, i) =>
            $"{i + 1},{i + 1},{accuracy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}")]);
    Console.WriteLine($"Saved the results to {evaluationFile}");
    return 0;
}

TrainedModel? model = null;
if (!train)
{
    if (File.Exists(modelFile))
        model = LoadModel();
    else
        Console.WriteLine($"No saved {method} model found in {modelFolder}, training one now.");
}

if (model == null)
{
    WriteFeatureFile();

    // Train on the vectors that were just written and keep the result for the next runs
    Console.WriteLine();
    model = ModelTrainer.Run(outputFile, method, binary, perceptron);
    Directory.CreateDirectory(modelFolder);
    model.Save(modelFile);
    Console.WriteLine($"Saved the model to {modelFile}");

    if (NetworkWeightsFile.Supports(model))
    {
        NetworkWeightsFile.Write(model, weightsFile);
        Console.WriteLine($"Wrote the neuron weights to {weightsFile}, edit it to change them");
    }

    if (model.Gates != null)
    {
        string circuitFile = Path.Combine(modelFolder, $"{modelName}_circuit.txt");
        string[] inputNames = FeatureExtractor.Properties
            .Select(property => binary ? $"{property} above average" : property)
            .ToArray();
        File.WriteAllLines(circuitFile, model.Gates.Describe(inputNames, model.concepts));
        Console.WriteLine($"Wrote the gate circuit to {circuitFile}");
    }
}
else if (NetworkWeightsFile.Supports(model) && !ApplyWeights(model))
{
    return 1;
}

if (images.Length > 0)
{
    foreach (string image in images)
        PrintClassification(model, image);
}
else
{
    while (true)
    {
        Console.WriteLine();
        Console.Write("Image to classify (empty to quit): ");
        string? image = Console.ReadLine()?.Trim().Trim('"', '\'');
        if (string.IsNullOrEmpty(image))
            break;
        PrintClassification(model, image);
    }
}

return 0;

// A saved model that can't be read (e.g. the features changed since it was trained) is trained again
TrainedModel? LoadModel()
{
    try
    {
        TrainedModel loaded = TrainedModel.Load(modelFile);
        Console.WriteLine($"Loaded the {loaded.method} model from {modelFile}");
        Console.WriteLine($"Trained {loaded.trainedAt:yyyy-MM-dd HH:mm}, test accuracy {loaded.testAccuracy:P1}");
        return loaded;
    }
    catch (InvalidDataException e)
    {
        Console.WriteLine($"Could not load the saved model: {e.Message} Training it again.");
        return null;
    }
}

// Changes the model's weights to the ones in its _weights.json, if there is one. False if the file is broken
bool ApplyWeights(TrainedModel model)
{
    if (!File.Exists(weightsFile))
        return true;
    try
    {
        int changed = NetworkWeightsFile.Apply(model, weightsFile);
        Console.WriteLine($"Applied {changed} weights and biases from {weightsFile}");
        return true;
    }
    catch (InvalidDataException e)
    {
        Console.WriteLine($"Could not apply the neuron weights: {e.Message}");
        return false;
    }
}

LearningMethod AskLearningMethod()
{
    while (true)
    {
        Console.WriteLine("Learning method:");
        Console.WriteLine("  1) neural network");
        Console.WriteLine("  2) k-nearest neighbors");
        Console.WriteLine("  3) decision tree");
        Console.WriteLine("  4) logic gate network");
        Console.WriteLine("  5) perceptron (single layer)");
        Console.Write("Choice [1]: ");
        switch (Console.ReadLine()?.Trim().ToLowerInvariant())
        {
            case null or "" or "1" or "nn":
                return LearningMethod.NeuralNetwork;
            case "2" or "knn":
                return LearningMethod.KNearestNeighbors;
            case "3" or "tree":
                return LearningMethod.DecisionTree;
            case "4" or "gates":
                return LearningMethod.LogicGateNetwork;
            case "5" or "perceptron":
                return LearningMethod.Perceptron;
            default:
                Console.WriteLine("Please enter 1, 2, 3, 4 or 5.");
                break;
        }
    }
}

void WriteFeatureFile()
{
    var files = conceptFolders
        .SelectMany(entry => Directory.GetFiles(entry.Folder, "*", SearchOption.AllDirectories)
            .Where(file => imageExtensions.Contains(Path.GetExtension(file).ToLowerInvariant()))
            .Order()
            .Select(file => (File: file, entry.Concept)));

    // Images are read on all CPU cores; AsOrdered keeps the vectors in the same order as the files
    List<FeatureVector> vectors = files
        .AsParallel()
        .AsOrdered()
        .Select(entry =>
        {
            Console.WriteLine($"Reading {entry.File}");
            return new FeatureVector(entry.Concept, FeatureExtractor.Extract(entry.File));
        })
        .ToList();

    FeatureFile featureFile = new FeatureFile(author, FeatureExtractor.Properties, vectors);
    File.WriteAllText(outputFile,
        JsonSerializer.Serialize(featureFile, new JsonSerializerOptions { WriteIndented = true }));

    Console.WriteLine($"Wrote {vectors.Count} vectors to {outputFile}");
}

void PrintClassification(TrainedModel model, string image)
{
    try
    {
        var (concept, probabilities) = model.ClassifyImage(image);
        Console.WriteLine($"{image}: {concept}");
        for (int i = 0; i < model.concepts.Length; i++)
            Console.WriteLine($"  {model.concepts[i],-12} {probabilities[i]:P1}");
    }
    catch (Exception e) when (e is FileNotFoundException or InvalidDataException or NotSupportedException)
    {
        Console.WriteLine($"Could not classify {image}: {e.Message}");
    }
}
