namespace uni_machine_learning_cs.MachineLearning;

// A trained classifier together with what it needs to classify new images:
// the concept names and the min / max used to scale the features
public class TrainedModel
{
    private readonly IClassifier classifier;
    private readonly double[] min;
    private readonly double[] max;
    public readonly string[] concepts;

    public TrainedModel(IClassifier classifier, string[] concepts, double[] min, double[] max)
    {
        this.classifier = classifier;
        this.concepts = concepts;
        this.min = min;
        this.max = max;
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
}
