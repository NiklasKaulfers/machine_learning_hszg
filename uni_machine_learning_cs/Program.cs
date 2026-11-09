using System.Text.Json;
using uni_machine_learning_cs.MachineLearning;

// Usage:
//   --train <method> [<image>..]  -> extract features of all images, train the method, save the model,
//                                    then classify the given images (or ask for images to classify)
//   <method> [<image>..]          -> load the saved model of the method and classify with it
//                                    (if there is no saved model yet, it is trained first)
//   --runs <n> <method>           -> train and test the method n times with different random training / test
//                                    splits, print every run's test accuracy and save them to evaluations/
//   --sweep <n> <method> [--values 1,2,3]
//                                 -> like --runs, but repeated for several complexity values: the number of
//                                    hidden layers (--nn), k (--knn) or the max depth (--tree). Saves the
//                                    accuracies per value to evaluations/ (without --values, defaults are used)
// <method> is --nn (neural network), --knn (k-nearest neighbors), --tree (decision tree) or
// --gates (logic gate network: every neuron learns to be one of the 16 two-input logic gates).
// Without a method, it is asked for. The models are saved in the models folder of the project root.
// --binary (works with every method) turns every feature into yes/no before training: 1 if it is above its
// average in the training images, otherwise 0. Saved as its own model, e.g. neural_network_binary.model.
// Training the gate network also writes models/<model name>_circuit.txt with every gate of the finished circuit
// (with --binary it is a pure logic circuit, otherwise the gates work on the 0-1 values like probabilities)
// Training the neural network also writes models/<model name>_weights.json with the weights of every neuron.
// Changes made in that file are used on every following run, until the network is trained again

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

string[] options = ["--train", "--runs", "--sweep", "--values", "--binary", "--nn", "--knn", "--tree", "--gates"];
string? unknownOption = args.FirstOrDefault(arg => arg.StartsWith("--") && !options.Contains(arg));
if (unknownOption != null)
{
    Console.WriteLine($"Unknown option {unknownOption}. Use --train, --runs <n> or --sweep <n>, " +
                      "and one of --nn, --knn, --tree, --gates (optionally with --binary).");
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
LearningMethod method = args.Contains("--gates") ? LearningMethod.LogicGateNetwork
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
    _ => "decision_tree",
} + (binary ? "_binary" : "");
string modelFile = Path.Combine(modelFolder, modelName + ".model");

if (sweepIndex >= 0)
{
    if (!File.Exists(outputFile))
        WriteFeatureFile();

    var results = ModelTrainer.Sweep(outputFile, method, sweepValues ?? ModelTrainer.DefaultSweepValues(method), runs,
        binary);

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

    List<double> accuracies = ModelTrainer.RunRepeated(outputFile, method, runs, binary);

    // One line per run, so the results can be opened in a spreadsheet
    string evaluationFolder = Path.Combine(projectRoot, "evaluations");
    string evaluationFile = Path.Combine(evaluationFolder,
        $"{Path.GetFileNameWithoutExtension(modelFile)}_{runs}_runs_{DateTime.Now:yyyy-MM-dd_HH-mm}.csv");
    Directory.CreateDirectory(evaluationFolder);
    File.WriteAllLines(evaluationFile, ["run,seed,test accuracy",
        ..accuracies.Select((accuracy, i) =>
            $"{i + 1},{i + 1},{accuracy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}")]);
    Console.WriteLine($"Saved the results to {evaluationFile}");
    return 0;
}

string weightsFile = Path.Combine(modelFolder, $"{modelName}_weights.json");

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
    model = ModelTrainer.Run(outputFile, method, binary);
    Directory.CreateDirectory(modelFolder);
    model.Save(modelFile);
    Console.WriteLine($"Saved the model to {modelFile}");

    if (model.Network != null)
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
else if (model.Network != null && File.Exists(weightsFile))
{
    try
    {
        int changed = NetworkWeightsFile.Apply(model, weightsFile);
        Console.WriteLine($"Applied {changed} weights and biases from {weightsFile}");
    }
    catch (InvalidDataException e)
    {
        Console.WriteLine($"Could not apply the neuron weights: {e.Message}");
        return 1;
    }
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

LearningMethod AskLearningMethod()
{
    while (true)
    {
        Console.WriteLine("Learning method:");
        Console.WriteLine("  1) neural network");
        Console.WriteLine("  2) k-nearest neighbors");
        Console.WriteLine("  3) decision tree");
        Console.WriteLine("  4) logic gate network");
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
            default:
                Console.WriteLine("Please enter 1, 2, 3 or 4.");
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
