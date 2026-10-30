using System.Text.Json;
using uni_machine_learning_cs.MachineLearning;

// Usage:
//   [--nn | --knn | --tree]                      -> extract features of all images, train, then ask for images to classify
//   [--nn | --knn | --tree] <image> [<image>..]  -> train on the existing image_data.json and classify the given images
// --nn uses the neural network, --knn k-nearest neighbors, --tree a decision tree. Without any, the method is asked for
// (with images given, the neural network is used)

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
string outputFile = Path.Combine(rootDir?.FullName ?? Directory.GetCurrentDirectory(), "image_data.json");

LearningMethod? chosenMethod = args.Contains("--tree") ? LearningMethod.DecisionTree
    : args.Contains("--knn") ? LearningMethod.KNearestNeighbors
    : args.Contains("--nn") ? LearningMethod.NeuralNetwork
    : null;
string[] images = args.Where(arg => arg is not ("--nn" or "--knn" or "--tree")).ToArray();

if (images.Length > 0)
{
    if (!File.Exists(outputFile))
        WriteFeatureFile();

    TrainedModel model = ModelTrainer.Run(outputFile, chosenMethod ?? LearningMethod.NeuralNetwork, verbose: false);
    foreach (string image in images)
        PrintClassification(model, image);
}
else
{
    LearningMethod method = chosenMethod ?? AskLearningMethod();
    WriteFeatureFile();

    // Train on the vectors that were just written
    Console.WriteLine();
    TrainedModel model = ModelTrainer.Run(outputFile, method);

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

LearningMethod AskLearningMethod()
{
    while (true)
    {
        Console.WriteLine("Learning method:");
        Console.WriteLine("  1) neural network");
        Console.WriteLine("  2) k-nearest neighbors");
        Console.WriteLine("  3) decision tree");
        Console.Write("Choice [1]: ");
        switch (Console.ReadLine()?.Trim().ToLowerInvariant())
        {
            case null or "" or "1" or "nn":
                return LearningMethod.NeuralNetwork;
            case "2" or "knn":
                return LearningMethod.KNearestNeighbors;
            case "3" or "tree":
                return LearningMethod.DecisionTree;
            default:
                Console.WriteLine("Please enter 1, 2 or 3.");
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
