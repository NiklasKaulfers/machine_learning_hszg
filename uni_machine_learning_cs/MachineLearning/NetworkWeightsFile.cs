using System.Text.Json;
using System.Text.Json.Serialization;

namespace uni_machine_learning_cs.MachineLearning;

// The weights of a neural network or a perceptron as an editable JSON file. Every neuron is listed by name with
// its bias and one weight per input, e.g. the output neuron "stop sign" has a weight for "hidden 1 neuron 0"
// (network) or straight for every feature such as "red" (perceptron, it has no hidden layers).
// When the file is applied, only what is listed is changed: left out neurons and weights keep their value
public static class NetworkWeightsFile
{
    private static readonly JsonSerializerOptions Options = new JsonSerializerOptions { WriteIndented = true };

    public static bool Supports(TrainedModel model) => model.Network != null || model.Perceptron != null;

    public static void Write(TrainedModel model, string path)
    {
        List<WeightsLayer> layers = Layers(model)
            .Select(layer => new WeightsLayer(layer.Name, layer.NeuronNames
                .Select((name, j) => new WeightsNeuron(name, layer.GetBias(j),
                    layer.InputNames.Select((input, i) => (input, layer.GetWeight(j, i)))
                        .ToDictionary(w => w.input, w => w.Item2)))
                .ToList()))
            .ToList();

        File.WriteAllText(path, JsonSerializer.Serialize(new WeightsFile(layers), Options));
    }

    // Returns how many biases and weights the file set
    public static int Apply(TrainedModel model, string path)
    {
        List<EditableLayer> layers = Layers(model);
        WeightsFile file;
        try
        {
            file = JsonSerializer.Deserialize<WeightsFile>(File.ReadAllText(path))
                   ?? throw new InvalidDataException($"'{path}' is empty.");
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"'{path}' is not valid JSON: {e.Message}");
        }

        int changed = 0;
        foreach (WeightsLayer fileLayer in file.Layers ?? [])
        {
            EditableLayer layer = layers.FirstOrDefault(l => l.Name == fileLayer.Name)
                ?? throw new InvalidDataException($"'{path}': there is no layer \"{fileLayer.Name}\".");

            foreach (WeightsNeuron neuron in fileLayer.Neurons ?? [])
            {
                int j = Array.IndexOf(layer.NeuronNames, neuron.Name);
                if (j < 0)
                    throw new InvalidDataException($"'{path}': layer \"{layer.Name}\" has no neuron \"{neuron.Name}\".");

                if (neuron.Bias is double bias)
                {
                    layer.SetBias(j, bias);
                    changed++;
                }
                foreach (var (input, weight) in neuron.Weights ?? [])
                {
                    int i = Array.IndexOf(layer.InputNames, input);
                    if (i < 0)
                        throw new InvalidDataException(
                            $"'{path}': neuron \"{neuron.Name}\" has no input \"{input}\".");
                    layer.SetWeight(j, i, weight);
                    changed++;
                }
            }
        }

        return changed;
    }

    // One layer of weights that the file can read and change: its neurons, their inputs, and access to the
    // weight from input i to neuron j and the bias of neuron j
    private record EditableLayer(string Name, string[] NeuronNames, string[] InputNames,
        Func<int, double> GetBias, Action<int, double> SetBias,
        Func<int, int, double> GetWeight, Action<int, int, double> SetWeight);

    private static List<EditableLayer> Layers(TrainedModel model)
    {
        string[] features = FeatureExtractor.Properties.ToArray();

        // The perceptron is only the output layer, fed straight by the features
        if (model.Perceptron is Perceptron perceptron)
            return
            [
                new EditableLayer("output", model.concepts, features,
                    perceptron.GetBias, perceptron.SetBias, perceptron.GetWeight, perceptron.SetWeight),
            ];

        NeuralNetwork network = model.Network
            ?? throw new InvalidOperationException("Only a neural network or a perceptron has neuron weights.");

        // Layer l + 1 of the network gets its inputs from layer l. Layer 0 = the inputs (named after the
        // features), the last layer = the outputs (named after the concepts), everything in between is hidden
        int last = network.LayerSizes.Count - 1;
        string[] NeuronNames(int layer) => layer == 0 ? features
            : layer == last ? model.concepts
            : Enumerable.Range(0, network.LayerSizes[layer]).Select(j => $"hidden {layer} neuron {j}").ToArray();

        return Enumerable.Range(0, last)
            .Select(l => new EditableLayer(l + 1 == last ? "output" : $"hidden {l + 1}",
                NeuronNames(l + 1), NeuronNames(l),
                j => network.GetBias(l, j), (j, value) => network.SetBias(l, j, value),
                (j, i) => network.GetWeight(l, j, i), (j, i, value) => network.SetWeight(l, j, i, value)))
            .ToList();
    }

    // Layout of the JSON file. Bias and weights may be left out, then they keep their value
    private record WeightsFile(
        [property: JsonPropertyName("layers")] List<WeightsLayer>? Layers);

    private record WeightsLayer(
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("neurons")] List<WeightsNeuron>? Neurons);

    private record WeightsNeuron(
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("bias")] double? Bias,
        [property: JsonPropertyName("weights")] Dictionary<string, double>? Weights);
}
