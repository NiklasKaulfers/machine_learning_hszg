using System.Text.Json;
using uni_machine_learning_cs.MachineLearning;

// Usage:
//   --train <method> [<image>..]  -> extract features of all images, train the method, save the model,
//                                    then classify the given images (or ask for images to classify)
//   <method> [<image>..]          -> load the saved model of the method and classify with it
//                                    (if there is no saved model yet, it is trained first)
// <method> is --nn (neural network), --knn (k-nearest neighbors) or --tree (decision tree).
// Without a method, it is asked for. The models are saved in the models folder of the project root

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

string[] options = ["--train", "--nn", "--knn", "--tree"];
string? unknownOption = args.FirstOrDefault(arg => arg.StartsWith("--") && !options.Contains(arg));
if (unknownOption != null)
{
    Console.WriteLine($"Unknown option {unknownOption}. Use --train and/or one of --nn, --knn, --tree.");
    return 1;
}

bool train = args.Contains("--train");
LearningMethod method = args.Contains("--tree") ? LearningMethod.DecisionTree
    : args.Contains("--knn") ? LearningMethod.KNearestNeighbors
    : args.Contains("--nn") ? LearningMethod.NeuralNetwork
    : AskLearningMethod();
string[] images = args.Where(arg => !options.Contains(arg)).ToArray();
string modelFile = Path.Combine(modelFolder, method switch
{
    LearningMethod.NeuralNetwork => "neural_network.model",
    LearningMethod.KNearestNeighbors => "k_nearest_neighbors.model",
    _ => "decision_tree.model",
});

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
    model = ModelTrainer.Run(outputFile, method);
    Directory.CreateDirectory(modelFolder);
    model.Save(modelFile);
    Console.WriteLine($"Saved the model to {modelFile}");
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
