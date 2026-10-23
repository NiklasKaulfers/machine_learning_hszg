namespace uni_machine_learning_cs.MachineLearning;

// A trained network together with what it needs to classify new images:
// the concept names and the min / max used to scale the features
public class TrainedModel
{
    private readonly NeuralNetwork network;
    private readonly double[] min;
    private readonly double[] max;
    public readonly string[] concepts;

    public TrainedModel(NeuralNetwork network, string[] concepts, double[] min, double[] max)
    {
        this.network = network;
        this.concepts = concepts;
        this.min = min;
        this.max = max;
    }

    // Scales every feature to 0-1 with the min / max of the training data
    public double[] Normalize(double[] input) => input
        .Select((value, f) => max[f] > min[f] ? (value - min[f]) / (max[f] - min[f]) : 0)
        .ToArray();

    // Classifies one image: the detected concept plus the probability of every concept
    public (string Concept, double[] Probabilities) ClassifyImage(string imagePath)
    {
        double[] input = Normalize(FeatureExtractor.ToNumbers(FeatureExtractor.Extract(imagePath)));
        double[] probabilities = network.Predict(input);
        int best = Array.IndexOf(probabilities, probabilities.Max());
        return (concepts[best], probabilities);
    }
}
