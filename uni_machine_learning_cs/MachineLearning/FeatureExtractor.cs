using System.Globalization;
using uni_machine_learning_cs.ImageAnalysis;

namespace uni_machine_learning_cs.MachineLearning;

// Turns an image into its feature vector
public static class FeatureExtractor
{
    private static readonly Colors[] PresenceColors =
        [Colors.Red, Colors.Blue, Colors.White, Colors.Yellow, Colors.Green, Colors.Orange, Colors.Black, Colors.Grey];

    // The image is split into GridSize x GridSize cells, each cell's average colour is a feature
    private const int GridSize = 8;

    // Number of evenly spaced vertical and horizontal lines along which the colour switches are counted
    private const int SwitchLines = 11;

    // Names of the values in each vector, in the same order as the values
    public static readonly List<string> Properties =
    [
        "width", "height",
        // "vertical line 0 switches" = leftmost column, "horizontal line 0 switches" = top row
        ..Enumerable.Range(0, SwitchLines).Select(line => $"vertical line {line} switches"),
        ..Enumerable.Range(0, SwitchLines).Select(line => $"horizontal line {line} switches"),
        "red", "blue", "white", "yellow", "green", "orange", "black", "grey",
        "average r", "average g", "average b",
        // "cell 0,0 average r" = top-left cell, "cell 1,1 average b" = bottom-right cell (row,column)
        ..Enumerable.Range(0, GridSize * GridSize).SelectMany(cell =>
            new[] { "r", "g", "b" }.Select(channel =>
                $"cell {cell / GridSize},{cell % GridSize} average {channel}")),
    ];

    // Invariant culture so decimals are always written with '.' (a German locale would use ',')
    private static readonly CultureInfo Culture = CultureInfo.InvariantCulture;

    public static List<string> Extract(string imagePath)
    {
        ReadImageFile fileContent = new ReadImageFile(imagePath);
        var average = fileContent.AverageColor();

        List<string> values =
        [
            fileContent.width.ToString(Culture),
            fileContent.height.ToString(Culture),
        ];
        // Color switches along each line
        foreach (int switches in fileContent.VerticalLinesColorSwitches(SwitchLines))
            values.Add(switches.ToString(Culture));
        foreach (int switches in fileContent.HorizontalLinesColorSwitches(SwitchLines))
            values.Add(switches.ToString(Culture));
        // presence of colors, 1 = present, 0 = not present
        foreach (Colors color in PresenceColors)
            values.Add(fileContent.ColorPresent(color) ? "1" : "0");
        // average color
        values.Add(average.R.ToString("F2", Culture));
        values.Add(average.G.ToString("F2", Culture));
        values.Add(average.B.ToString("F2", Culture));
        // average color of each grid cell
        foreach (var cell in fileContent.GridAverageColors(GridSize, GridSize))
        {
            values.Add(cell.R.ToString("F2", Culture));
            values.Add(cell.G.ToString("F2", Culture));
            values.Add(cell.B.ToString("F2", Culture));
        }

        return values;
    }

    public static double[] ToNumbers(IEnumerable<string> values) =>
        values.Select(value => double.Parse(value, Culture)).ToArray();
}
