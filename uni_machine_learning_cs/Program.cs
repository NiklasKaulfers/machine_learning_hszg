using System.Text.Json;
using uni_machine_learning_cs.MachineLearning;

// Usage:
//   no arguments        -> extract features of all images, train the network, then ask for images to classify
//   <image> [<image>..] -> train on the existing image_data.json and classify the given images

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

if (args.Length > 0)
{
    if (!File.Exists(outputFile))
        WriteFeatureFile();

    TrainedModel model = NetworkTrainer.Run(outputFile, verbose: false);
    foreach (string image in args)
        PrintClassification(model, image);
}
else
{
    WriteFeatureFile();

    // Train the neural network on the vectors that were just written
    Console.WriteLine();
    TrainedModel model = NetworkTrainer.Run(outputFile);

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
