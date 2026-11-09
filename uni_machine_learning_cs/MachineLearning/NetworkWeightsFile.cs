using System.Text.Json;
using System.Text.Json.Serialization;

namespace uni_machine_learning_cs.MachineLearning;

// The weights of a neural network as an editable JSON file. Every neuron is listed by name with its bias
// and one weight per input neuron, e.g. the output neuron "stop sign" has a weight for "hidden 1 neuron 0".
// When the file is applied, only what is listed is changed: left out neurons and weights keep their value
public static class NetworkWeightsFile
{
    private static readonly JsonSerializerOptions Options = new JsonSerializerOptions { WriteIndented = true };

    public static void Write(TrainedModel model, string path)
    {
        NeuralNetwork network = RequireNetwork(model);
        List<WeightsLayer> layers = [];
        for (int l = 0; l < network.LayerSizes.Count - 1; l++)
        {
            string[] inputNames = NeuronNames(model, l);
            List<WeightsNeuron> neurons = NeuronNames(model, l + 1)
                .Select((name, j) => new WeightsNeuron(name, network.GetBias(l, j),
                    inputNames.Select((input, i) => (input, network.GetWeight(l, j, i)))
                        .ToDictionary(w => w.input, w => w.Item2)))
                .ToList();
            layers.Add(new WeightsLayer(LayerName(network, l + 1), neurons));
        }

        File.WriteAllText(path, JsonSerializer.Serialize(new WeightsFile(layers), Options));
    }

    // Returns how many biases and weights the file set
    public static int Apply(TrainedModel model, string path)
    {
        NeuralNetwork network = RequireNetwork(model);
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
        foreach (WeightsLayer layer in file.Layers ?? [])
        {
            // Layer l + 1 of the network gets its inputs from layer l
            int l = Enumerable.Range(0, network.LayerSizes.Count - 1)
                .FirstOrDefault(l => LayerName(network, l + 1) == layer.Name, -1);
            if (l < 0)
                throw new InvalidDataException($"'{path}': there is no layer \"{layer.Name}\".");

            string[] neuronNames = NeuronNames(model, l + 1);
            string[] inputNames = NeuronNames(model, l);
            foreach (WeightsNeuron neuron in layer.Neurons ?? [])
            {
                int j = Array.IndexOf(neuronNames, neuron.Name);
                if (j < 0)
                    throw new InvalidDataException($"'{path}': layer \"{layer.Name}\" has no neuron \"{neuron.Name}\".");

                if (neuron.Bias is double bias)
                {
                    network.SetBias(l, j, bias);
                    changed++;
                }
                foreach (var (input, weight) in neuron.Weights ?? [])
                {
                    int i = Array.IndexOf(inputNames, input);
                    if (i < 0)
                        throw new InvalidDataException(
                            $"'{path}': neuron \"{neuron.Name}\" has no input \"{input}\".");
                    network.SetWeight(l, j, i, weight);
                    changed++;
                }
            }
        }

        return changed;
    }

    private static NeuralNetwork RequireNetwork(TrainedModel model) =>
        model.Network ?? throw new InvalidOperationException("Only a neural network has neuron weights.");

    // Layer 0 = the inputs, the last layer = the outputs, everything in between is hidden
    private static string LayerName(NeuralNetwork network, int layer) =>
        layer == network.LayerSizes.Count - 1 ? "output" : $"hidden {layer}";

    // Inputs are named after the features, outputs after the concepts
    private static string[] NeuronNames(TrainedModel model, int layer)
    {
        NeuralNetwork network = RequireNetwork(model);
        if (layer == 0)
            return FeatureExtractor.Properties.ToArray();
        if (layer == network.LayerSizes.Count - 1)
            return model.concepts;
        return Enumerable.Range(0, network.LayerSizes[layer]).Select(j => $"hidden {layer} neuron {j}").ToArray();
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
