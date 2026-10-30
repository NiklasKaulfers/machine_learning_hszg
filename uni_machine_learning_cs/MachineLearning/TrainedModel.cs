namespace uni_machine_learning_cs.MachineLearning;

// A trained classifier together with what it needs to classify new images:
// the concept names and the min / max used to scale the features.
// Can be saved to a file and loaded again, so training only has to happen once
public class TrainedModel
{
    // Marks the file as a model file. The version changes whenever the layout changes,
    // so old files are rejected instead of misread
    private const string FileMagic = "uni_machine_learning_cs model";
    private const int FileVersion = 1;

    private readonly IClassifier classifier;
    private readonly double[] min;
    private readonly double[] max;
    public readonly LearningMethod method;
    public readonly string[] concepts;
    public readonly DateTime trainedAt;
    public readonly double testAccuracy;

    public TrainedModel(IClassifier classifier, LearningMethod method, string[] concepts, double[] min, double[] max,
        DateTime trainedAt, double testAccuracy)
    {
        this.classifier = classifier;
        this.method = method;
        this.concepts = concepts;
        this.min = min;
        this.max = max;
        this.trainedAt = trainedAt;
        this.testAccuracy = testAccuracy;
    }

    // Scales every feature to 0-1 with the min / max of the training data
    public double[] Normalize(double[] input) => Normalize(input, min, max);

    public static double[] Normalize(double[] input, double[] min, double[] max) => input
        .Select((value, f) => max[f] > min[f] ? (value - min[f]) / (max[f] - min[f]) : 0)
        .ToArray();

    // Classifies one image: the detected concept plus the probability of every concept
    public (string Concept, double[] Probabilities) ClassifyImage(string imagePath)
    {
        double[] input = Normalize(FeatureExtractor.ToNumbers(FeatureExtractor.Extract(imagePath)));
        double[] probabilities = classifier.Predict(input);
        int best = classifier.Classify(input);
        return (concepts[best], probabilities);
    }

    public void Save(string path)
    {
        using BinaryWriter writer = new BinaryWriter(File.Create(path));
        writer.Write(FileMagic);
        writer.Write(FileVersion);
        writer.Write((int)method);
        writer.Write(trainedAt.ToBinary());
        writer.Write(testAccuracy);
        writer.WriteArray(FeatureExtractor.Properties.ToArray());
        writer.WriteArray(concepts);
        writer.WriteArray(min);
        writer.WriteArray(max);
        classifier.Write(writer);
    }

    public static TrainedModel Load(string path)
    {
        using BinaryReader reader = new BinaryReader(File.OpenRead(path));
        try
        {
            if (reader.ReadString() != FileMagic || reader.ReadInt32() != FileVersion)
                throw new InvalidDataException($"'{path}' is not a model file of this program version.");

            LearningMethod method = (LearningMethod)reader.ReadInt32();
            DateTime trainedAt = DateTime.FromBinary(reader.ReadInt64());
            double testAccuracy = reader.ReadDouble();

            // A model only works with the features it was trained on
            string[] properties = reader.ReadStringArray();
            if (!properties.SequenceEqual(FeatureExtractor.Properties))
                throw new InvalidDataException($"'{path}' was trained on different features than the current ones.");

            string[] concepts = reader.ReadStringArray();
            double[] min = reader.ReadDoubleArray();
            double[] max = reader.ReadDoubleArray();
            IClassifier classifier = method switch
            {
                LearningMethod.NeuralNetwork => NeuralNetwork.Read(reader),
                LearningMethod.KNearestNeighbors => KNearestNeighbors.Read(reader),
                LearningMethod.DecisionTree => DecisionTree.Read(reader),
                _ => throw new InvalidDataException($"'{path}' contains an unknown learning method."),
            };

            return new TrainedModel(classifier, method, concepts, min, max, trainedAt, testAccuracy);
        }
        catch (EndOfStreamException)
        {
            throw new InvalidDataException($"'{path}' is incomplete.");
        }
    }
}
