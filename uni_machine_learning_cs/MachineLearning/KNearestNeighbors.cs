namespace uni_machine_learning_cs.MachineLearning;

// k-nearest-neighbors classifier: there is no real training, the training vectors are just stored.
// A new vector gets the concept that most of its k closest training vectors (Euclidean distance) have.
public class KNearestNeighbors : IClassifier
{
    private readonly List<(double[] Input, int Class)> training;
    private readonly int classCount;
    private readonly int k;

    public KNearestNeighbors(List<(double[] Input, int Class)> training, int classCount, int k)
    {
        if (k < 1 || k > training.Count)
            throw new ArgumentOutOfRangeException(nameof(k), $"k must be between 1 and {training.Count}.");

        this.training = training;
        this.classCount = classCount;
        this.k = k;
    }

    // Share of the k nearest neighbors that belong to each class
    public double[] Predict(double[] input) => Votes(NearestClasses(input));

    // Ties between classes are won by the class of the closest neighbor among them
    public int Classify(double[] input)
    {
        int[] nearestClasses = NearestClasses(input);
        double[] votes = Votes(nearestClasses);
        double best = votes.Max();
        return nearestClasses.First(c => votes[c] == best);
    }

    private double[] Votes(int[] nearestClasses)
    {
        double[] votes = new double[classCount];
        foreach (int nearestClass in nearestClasses)
            votes[nearestClass] += 1.0 / k;
        return votes;
    }

    // Classes of the k closest training vectors, closest first
    private int[] NearestClasses(double[] input)
    {
        // The k closest vectors found so far, sorted by distance
        double[] nearestDistances = Enumerable.Repeat(double.MaxValue, k).ToArray();
        int[] nearestClasses = new int[k];

        foreach (var (sample, sampleClass) in training)
        {
            double distance = SquaredDistance(input, sample, nearestDistances[^1]);
            if (distance >= nearestDistances[^1])
                continue;

            // Insert into the sorted list, the farthest one drops out
            int position = k - 1;
            while (position > 0 && nearestDistances[position - 1] > distance)
            {
                nearestDistances[position] = nearestDistances[position - 1];
                nearestClasses[position] = nearestClasses[position - 1];
                position--;
            }
            nearestDistances[position] = distance;
            nearestClasses[position] = sampleClass;
        }

        return nearestClasses;
    }

    // The square root is not needed to compare distances. Stops early once the sum reaches the limit,
    // because then the vector can't be one of the nearest anyway
    private static double SquaredDistance(double[] a, double[] b, double limit)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double difference = a[i] - b[i];
            sum += difference * difference;
            if (sum >= limit)
                return sum;
        }
        return sum;
    }
}
