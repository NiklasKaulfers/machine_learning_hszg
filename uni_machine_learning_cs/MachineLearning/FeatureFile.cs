using System.Text.Json.Serialization;

namespace uni_machine_learning_cs.MachineLearning;

// Layout of image_data.json
public record FeatureFile(
    [property: JsonPropertyName("author")] string Author,
    [property: JsonPropertyName("properties")] List<string> Properties,
    [property: JsonPropertyName("vectors")] List<FeatureVector> Vectors);

public record FeatureVector(
    [property: JsonPropertyName("concept")] string Concept,
    [property: JsonPropertyName("values")] List<string> Values);
