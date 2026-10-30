namespace uni_machine_learning_cs.MachineLearning;

// Binary decision tree (CART). Every inner node asks "is feature f <= threshold?" and sends the vector
// left or right; every leaf holds the share of each class among the training vectors that ended there.
// Each split is chosen so that the two halves are as pure as possible, measured with the Gini impurity.
public class DecisionTree : IClassifier
{
    // A leaf has Probabilities, an inner node has Left and Right
    private class Node
    {
        public int Feature;
        public double Threshold;
        public Node? Left;
        public Node? Right;
        public double[]? Probabilities;
    }

    private readonly List<(double[] Input, int Class)> training;
    private readonly int classCount;
    private readonly int maxDepth;
    private readonly int minSamplesSplit;
    private readonly Node root;

    public int Depth { get; private set; }
    public int LeafCount { get; private set; }

    // maxDepth: the tree stops growing at this depth. minSamplesSplit: nodes with fewer vectors become leaves.
    // Both keep the tree from learning every single training vector by heart (overfitting)
    public DecisionTree(List<(double[] Input, int Class)> training, int classCount, int maxDepth, int minSamplesSplit)
    {
        this.training = training;
        this.classCount = classCount;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        root = Build(Enumerable.Range(0, training.Count).ToArray(), 0);
    }

    // Used when loading a saved tree, which only needs its nodes
    private DecisionTree(BinaryReader reader)
    {
        training = [];
        root = ReadNode(reader, 0);
    }

    // The nodes top-down, each inner node followed by its left and then its right subtree
    public void Write(BinaryWriter writer) => WriteNode(writer, root);

    public static DecisionTree Read(BinaryReader reader) => new DecisionTree(reader);

    private static void WriteNode(BinaryWriter writer, Node node)
    {
        bool isLeaf = node.Probabilities != null;
        writer.Write(isLeaf);
        if (isLeaf)
        {
            writer.WriteArray(node.Probabilities!);
            return;
        }
        writer.Write(node.Feature);
        writer.Write(node.Threshold);
        WriteNode(writer, node.Left!);
        WriteNode(writer, node.Right!);
    }

    private Node ReadNode(BinaryReader reader, int depth)
    {
        Depth = Math.Max(Depth, depth);
        if (reader.ReadBoolean())
        {
            LeafCount++;
            return new Node { Probabilities = reader.ReadDoubleArray() };
        }
        return new Node
        {
            Feature = reader.ReadInt32(),
            Threshold = reader.ReadDouble(),
            Left = ReadNode(reader, depth + 1),
            Right = ReadNode(reader, depth + 1),
        };
    }

    public double[] Predict(double[] input)
    {
        Node node = root;
        while (node.Probabilities == null)
            node = input[node.Feature] <= node.Threshold ? node.Left! : node.Right!;
        return node.Probabilities;
    }

    // samples: indices into the training list of the vectors that reach this node
    private Node Build(int[] samples, int depth)
    {
        Depth = Math.Max(Depth, depth);
        int[] counts = ClassCounts(samples);

        bool pure = counts.Count(c => c > 0) == 1;
        if (pure || depth >= maxDepth || samples.Length < minSamplesSplit)
            return Leaf(counts, samples.Length);

        var (feature, threshold, impurity) = BestSplit(samples);
        // No split makes the halves purer than the node itself
        if (feature < 0 || impurity >= Gini(counts, samples.Length))
            return Leaf(counts, samples.Length);

        int[] left = samples.Where(s => training[s].Input[feature] <= threshold).ToArray();
        int[] right = samples.Where(s => training[s].Input[feature] > threshold).ToArray();
        return new Node
        {
            Feature = feature,
            Threshold = threshold,
            Left = Build(left, depth + 1),
            Right = Build(right, depth + 1),
        };
    }

    private Node Leaf(int[] counts, int total)
    {
        LeafCount++;
        return new Node { Probabilities = counts.Select(c => c / (double)total).ToArray() };
    }

    // Tries every feature (in parallel) and every threshold between two neighbouring values,
    // returns the split with the lowest weighted Gini impurity of the two halves
    private (int Feature, double Threshold, double Impurity) BestSplit(int[] samples)
    {
        int featureCount = training[samples[0]].Input.Length;

        return Enumerable.Range(0, featureCount)
            .AsParallel()
            .Select(feature => BestSplitForFeature(samples, feature))
            .Aggregate((Feature: -1, Threshold: 0.0, Impurity: double.MaxValue),
                (best, candidate) => candidate.Impurity < best.Impurity ||
                                     (candidate.Impurity == best.Impurity && candidate.Feature < best.Feature)
                    ? candidate
                    : best);
    }

    private (int Feature, double Threshold, double Impurity) BestSplitForFeature(int[] samples, int feature)
    {
        // Sort by the feature, then move the vectors one by one from the right half to the left half
        int[] sorted = samples.OrderBy(s => training[s].Input[feature]).ToArray();
        int[] leftCounts = new int[classCount];
        int[] rightCounts = ClassCounts(sorted);
        (int Feature, double Threshold, double Impurity) best = (-1, 0, double.MaxValue);

        for (int i = 0; i < sorted.Length - 1; i++)
        {
            int moved = training[sorted[i]].Class;
            leftCounts[moved]++;
            rightCounts[moved]--;

            double value = training[sorted[i]].Input[feature];
            double next = training[sorted[i + 1]].Input[feature];
            // Only split between two different values, equal values have to end up on the same side
            if (value == next)
                continue;

            int leftTotal = i + 1;
            int rightTotal = sorted.Length - leftTotal;
            double impurity = (leftTotal * Gini(leftCounts, leftTotal) + rightTotal * Gini(rightCounts, rightTotal))
                              / sorted.Length;
            if (impurity < best.Impurity)
                best = (feature, (value + next) / 2, impurity);
        }

        return best;
    }

    private int[] ClassCounts(int[] samples)
    {
        int[] counts = new int[classCount];
        foreach (int s in samples)
            counts[training[s].Class]++;
        return counts;
    }

    // Probability that two random vectors of the group have different classes: 0 = pure, higher = more mixed
    private static double Gini(int[] counts, int total)
    {
        double sum = 0;
        foreach (int count in counts)
        {
            double share = count / (double)total;
            sum += share * share;
        }
        return 1 - sum;
    }
}
