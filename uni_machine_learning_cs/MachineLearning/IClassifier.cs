namespace uni_machine_learning_cs.MachineLearning;

// Anything that can turn a (normalized) feature vector into a concept
public interface IClassifier
{
    // Probability for each class
    double[] Predict(double[] input);

    // Index of the most likely class
    int Classify(double[] input)
    {
        double[] output = Predict(input);
        return Array.IndexOf(output, output.Max());
    }

    // Stores everything the classifier learned, so it can be loaded again without training
    void Write(BinaryWriter writer);
}
