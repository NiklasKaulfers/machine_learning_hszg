using System.Globalization;

namespace uni_machine_learning_cs.MachineLearning;

// Changes trained weights the way they were first tuned by hand in models/*_weights.json, but automatically and
// only with validation data: vectors the model was not trained on and that are not part of the test data.
// Every change is kept only if it gets more validation vectors right, and all of it is undone if the gain in the
// end is below MinGain (smaller gains are mostly luck with this many validation vectors).
// Returns what was changed, e.g. "cut hidden 1 neuron 19; bias stop sign +0.25"
public static class WeightTuner
{
    private const double MinGain = 0.001;
    private const int Rounds = 8;

    // Neural network: only the output layer is changed.
    // 1) Cut hidden neurons: set their output weights to 0 and add their average activation times the old weight
    //    to the output biases, so the outputs stay the same on average. Repeated while a cut helps.
    // 2) Shift the output biases, which moves the borders between the concepts
    public static string Tune(NeuralNetwork network, List<(double[] Input, int Class)> validation, string[] concepts)
    {
        int layer = network.LayerSizes.Count - 2;
        int hiddenCount = network.LayerSizes[layer];
        int classCount = concepts.Length;

        double[][] hidden = validation.AsParallel().AsOrdered()
            .Select(s => network.LastHiddenActivations(s.Input)).ToArray();
        int[] classes = validation.Select(s => s.Class).ToArray();
        double[,] weights = new double[classCount, hiddenCount];
        double[] biases = new double[classCount];
        for (int c = 0; c < classCount; c++)
        {
            biases[c] = network.GetBias(layer, c);
            for (int h = 0; h < hiddenCount; h++)
                weights[c, h] = network.GetWeight(layer, c, h);
        }
        double[] meanActivation = Enumerable.Range(0, hiddenCount).Select(h => hidden.Average(a => a[h])).ToArray();

        // Output sums of every validation vector, without the biases
        double[][] sums = hidden.Select(a => Enumerable.Range(0, classCount)
            .Select(c => Enumerable.Range(0, hiddenCount).Sum(h => weights[c, h] * a[h])).ToArray()).ToArray();

        int before = Correct(sums, biases, classes);
        int best = before;
        List<int> cut = [];
        while (true)
        {
            int bestNeuron = -1;
            double[][]? bestSums = null;
            double[]? bestBiases = null;
            for (int h = 0; h < hiddenCount; h++)
            {
                if (cut.Contains(h))
                    continue;
                double[][] trySums = sums.Select((row, n) => row.Select((sum, c) => sum - weights[c, h] * hidden[n][h])
                    .ToArray()).ToArray();
                double[] tryBiases = biases.Select((bias, c) => bias + weights[c, h] * meanActivation[h]).ToArray();
                int correct = Correct(trySums, tryBiases, classes);
                if (correct > best)
                    (best, bestNeuron, bestSums, bestBiases) = (correct, h, trySums, tryBiases);
            }
            if (bestNeuron < 0)
                break;
            cut.Add(bestNeuron);
            sums = bestSums!;
            biases = bestBiases!;
        }

        double[] shift = ShiftBiases(sums, biases, classes, [0.05, 0.1, 0.25, 0.5, 1], ref best);

        if (best - before < MinGain * validation.Count)
            return "no change";

        foreach (int h in cut)
            for (int c = 0; c < classCount; c++)
                network.SetWeight(layer, c, h, 0);
        for (int c = 0; c < classCount; c++)
            network.SetBias(layer, c, biases[c] + shift[c]);

        return string.Join("; ", [
            ..cut.Select(h => $"cut hidden {layer} neuron {h}"),
            ..Describe(shift, concepts, "F2")]);
    }

    // Perceptron:
    // 1) Arrow pattern: the left and right turn signs are mirror images. The difference of their average grid
    //    cells in the training data, made exactly mirror-antisymmetric, is added to the "left turn" weights and
    //    subtracted from the "right turn" weights.
    // 2) STOP lettering: "horizontal line 5 switches" (the middle line crosses the letters) is added to
    //    "stop sign" and subtracted from "yield".
    // 3) Shift the biases.
    // Every strength is tried (relative to the largest weight, the perceptron's weights have no fixed scale) and
    // the best one on the validation data is kept
    public static string Tune(Perceptron perceptron, List<(double[] Input, int Class)> training,
        List<(double[] Input, int Class)> validation, string[] concepts)
    {
        int classCount = concepts.Length;
        int inputCount = perceptron.InputCount;
        List<string> features = FeatureExtractor.Properties;
        int left = Array.IndexOf(concepts, "left turn"), right = Array.IndexOf(concepts, "right turn");
        int stop = Array.IndexOf(concepts, "stop sign"), yield = Array.IndexOf(concepts, "yield");
        int lettering = features.IndexOf("horizontal line 5 switches");

        double[] arrow = left >= 0 && right >= 0 ? ArrowPattern(training, left, right) : new double[inputCount];
        double scale = Enumerable.Range(0, classCount)
            .SelectMany(c => Enumerable.Range(0, inputCount).Select(i => Math.Abs(perceptron.GetWeight(c, i))))
            .DefaultIfEmpty(0).Max();
        if (scale == 0)
            return "no change";

        int[] classes = validation.Select(s => s.Class).ToArray();
        double[] biases = Enumerable.Range(0, classCount).Select(perceptron.GetBias).ToArray();
        double[][] baseSums = validation.Select(s => Enumerable.Range(0, classCount)
            .Select(c => Enumerable.Range(0, inputCount).Sum(i => perceptron.GetWeight(c, i) * s.Input[i]))
            .ToArray()).ToArray();
        double[] arrowValues = validation.Select(s => s.Input.Select((x, i) => x * arrow[i]).Sum()).ToArray();
        double[] letteringValues = validation.Select(s => lettering >= 0 ? s.Input[lettering] : 0).ToArray();

        int before = Correct(baseSums, biases, classes);
        (int Correct, double Arrow, double Lettering, double[] Shift) best = (before, 0, 0, new double[classCount]);
        double[] steps = new[] { 0.01, 0.02, 0.05, 0.1, 0.2, 0.4 }.Select(step => step * scale).ToArray();
        foreach (double arrowStrength in (double[])[0, 0.1, 0.2, 0.4, 0.8])
        foreach (double letteringStrength in (double[])[0, 0.1, 0.2, 0.4])
        {
            if ((arrowStrength > 0 && left < 0) || (letteringStrength > 0 && (stop < 0 || lettering < 0)))
                continue;
            double a = arrowStrength * scale, l = letteringStrength * scale;
            double[][] sums = baseSums.Select((row, n) => row.Select((sum, c) => sum
                + (c == left ? a * arrowValues[n] : c == right ? -a * arrowValues[n] : 0)
                + (c == stop ? l * letteringValues[n] : c == yield ? -l * letteringValues[n] : 0)).ToArray()).ToArray();
            int correct = Correct(sums, biases, classes);
            double[] shift = ShiftBiases(sums, biases, classes, steps, ref correct);
            if (correct > best.Correct)
                best = (correct, a, l, shift);
        }

        if (best.Correct - before < MinGain * validation.Count)
            return "no change";

        for (int i = 0; i < inputCount; i++)
        {
            if (left >= 0)
            {
                perceptron.SetWeight(left, i, perceptron.GetWeight(left, i) + best.Arrow * arrow[i]);
                perceptron.SetWeight(right, i, perceptron.GetWeight(right, i) - best.Arrow * arrow[i]);
            }
        }
        if (stop >= 0 && lettering >= 0)
        {
            perceptron.SetWeight(stop, lettering, perceptron.GetWeight(stop, lettering) + best.Lettering);
            perceptron.SetWeight(yield, lettering, perceptron.GetWeight(yield, lettering) - best.Lettering);
        }
        for (int c = 0; c < classCount; c++)
            perceptron.SetBias(c, biases[c] + best.Shift[c]);

        CultureInfo culture = CultureInfo.InvariantCulture;
        return string.Join("; ", [
            ..best.Arrow > 0 ? [$"arrow pattern x{best.Arrow.ToString("F0", culture)}"] : Array.Empty<string>(),
            ..best.Lettering > 0 ? [$"lettering x{best.Lettering.ToString("F0", culture)}"] : Array.Empty<string>(),
            ..Describe(best.Shift, concepts, "F0")]);
    }

    // Left minus right turn average of every grid cell feature, made mirror-antisymmetric (cell r,c gets the
    // opposite value of cell r,7-c) so it only reacts to which side the arrow points, scaled to a largest value of 1
    private static double[] ArrowPattern(List<(double[] Input, int Class)> training, int left, int right)
    {
        List<string> features = FeatureExtractor.Properties;
        double[] difference = Enumerable.Range(0, features.Count).Select(f =>
            training.Where(s => s.Class == left).Average(s => s.Input[f]) -
            training.Where(s => s.Class == right).Average(s => s.Input[f])).ToArray();

        double[] pattern = new double[features.Count];
        for (int f = 0; f < features.Count; f++)
        {
            string[] parts = features[f].Split(' ');
            // "cell r,c average x"
            if (parts.Length != 4 || parts[0] != "cell")
                continue;
            int[] rowColumn = parts[1].Split(',').Select(int.Parse).ToArray();
            int columns = features.Count(name => name.StartsWith($"cell {rowColumn[0]},") && name.EndsWith(parts[3]));
            int mirror = features.IndexOf($"cell {rowColumn[0]},{columns - 1 - rowColumn[1]} average {parts[3]}");
            pattern[f] = (difference[f] - difference[mirror]) / 2;
        }
        double largest = pattern.Max(Math.Abs);
        return largest > 0 ? pattern.Select(value => value / largest).ToArray() : pattern;
    }

    // Moves one bias at a time by one of the steps, as long as that gets more vectors right.
    // correct: the number right before, updated to the number right after. Returns the shift of every bias
    private static double[] ShiftBiases(double[][] sums, double[] biases, int[] classes, double[] steps, ref int correct)
    {
        double[] shift = new double[biases.Length];
        for (int round = 0; round < Rounds; round++)
        {
            bool improved = false;
            for (int c = 0; c < biases.Length; c++)
            foreach (double step in steps.Concat(steps.Select(step => -step)))
            {
                double[] tryBiases = biases.Select((bias, k) => bias + shift[k] + (k == c ? step : 0)).ToArray();
                int tryCorrect = Correct(sums, tryBiases, classes);
                if (tryCorrect > correct)
                {
                    correct = tryCorrect;
                    shift[c] += step;
                    improved = true;
                }
            }
            if (!improved)
                break;
        }
        return shift;
    }

    private static int Correct(double[][] sums, double[] biases, int[] classes)
    {
        int correct = 0;
        for (int n = 0; n < sums.Length; n++)
        {
            int predicted = 0;
            for (int c = 1; c < biases.Length; c++)
                if (sums[n][c] + biases[c] > sums[n][predicted] + biases[predicted])
                    predicted = c;
            if (predicted == classes[n])
                correct++;
        }
        return correct;
    }

    private static IEnumerable<string> Describe(double[] shift, string[] concepts, string format) => shift
        .Select((value, c) => (value, c))
        .Where(entry => entry.value != 0)
        .Select(entry => $"bias {concepts[entry.c]} {entry.value.ToString("+0.##;-0.##", CultureInfo.InvariantCulture)}");
}
