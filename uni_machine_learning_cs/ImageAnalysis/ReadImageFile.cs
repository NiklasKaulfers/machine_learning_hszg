namespace uni_machine_learning_cs.ImageAnalysis;

public readonly record struct Rgb(byte R, byte G, byte B);

public class ReadImageFile
{
    public readonly int width;
    public readonly int height;

    // pixels[x, y], with (0, 0) being the top-left corner
    public readonly Rgb[,] pixels;

    public ReadImageFile(string path)
    {
        if (string.IsNullOrEmpty(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));

        // Relative paths are resolved against the current working directory
        string fullPath = Path.GetFullPath(path, Directory.GetCurrentDirectory());
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"Image not found: '{fullPath}'.", fullPath);

        byte[] data = File.ReadAllBytes(fullPath);

        if (data.Length < 54 || data[0] != 'B' || data[1] != 'M')
            throw new InvalidDataException($"'{fullPath}' is not a BMP file.");

        int pixelOffset = BitConverter.ToInt32(data, 10);
        width = BitConverter.ToInt32(data, 18);
        int rawHeight = BitConverter.ToInt32(data, 22);
        short bitsPerPixel = BitConverter.ToInt16(data, 28);
        int compression = BitConverter.ToInt32(data, 30);

        if (bitsPerPixel != 24 && bitsPerPixel != 32)
            throw new NotSupportedException($"Only 24/32-bit BMPs are supported, got {bitsPerPixel}-bit.");
        if (compression != 0 && compression != 3)
            throw new NotSupportedException($"Compressed BMPs are not supported (compression = {compression}).");

        // Positive height means rows are stored bottom-up, negative means top-down
        bool bottomUp = rawHeight > 0;
        height = Math.Abs(rawHeight);

        int bytesPerPixel = bitsPerPixel / 8;
        // Each row is padded to a multiple of 4 bytes
        int rowSize = (width * bytesPerPixel + 3) / 4 * 4;

        pixels = new Rgb[width, height];
        for (int y = 0; y < height; y++)
        {
            int fileRow = bottomUp ? height - 1 - y : y;
            int rowStart = pixelOffset + fileRow * rowSize;
            for (int x = 0; x < width; x++)
            {
                int i = rowStart + x * bytesPerPixel;
                // BMP stores pixels as BGR
                pixels[x, y] = new Rgb(data[i + 2], data[i + 1], data[i]);
            }
        }
    }

    public Rgb GetPixel(int x, int y) => pixels[x, y];


    public int VerticalLineColorSwitches()
    {
        return LineColorSwitches(width/2, false);
    }

    public int HorizontalLineColorSwitches()
    {
        return LineColorSwitches(height/2, true);
    }

    // A new colour has to last this many pixels in a row to count as a switch,
    // so single anti-aliased edge pixels don't count as an extra colour
    private const int MinRunLength = 1;

    // Counts how often the traffic sign colour changes along a line.
    // isHorizontal: walks along row y = coordinate, otherwise along column x = coordinate.
    private int LineColorSwitches(int coordinate, bool isHorizontal)
    {
        int length = isHorizontal ? width : height;
        Rgb PixelAt(int i) => isHorizontal ? pixels[i, coordinate] : pixels[coordinate, i];

        Colors current = ClosestColor(PixelAt(0));
        Colors candidate = current;
        int candidateRun = 0;
        int switches = 0;

        for (int i = 1; i < length; i++)
        {
            Colors color = ClosestColor(PixelAt(i));

            if (color.Equals(current))
            {
                candidateRun = 0;
                continue;
            }

            if (candidateRun > 0 && color.Equals(candidate))
            {
                candidateRun++;
            }
            else
            {
                candidate = color;
                candidateRun = 1;
            }

            if (candidateRun >= MinRunLength)
            {
                switches++;
                current = candidate;
                candidateRun = 0;
            }
        }

        return switches;
    }

    // Below this saturation (0-1) a pixel counts as white / grey / black instead of a colour
    private const double MinSaturation = 0.25;
    // Brightness (0-1) thresholds for the colourless pixels
    private const double WhiteMinBrightness = 0.75;
    private const double BlackMaxBrightness = 0.25;

    // Maps a pixel to a traffic sign colour by its hue, so lighter or darker
    // shades (e.g. a pale blue) still count as that colour
    private static Colors ClosestColor(Rgb pixel)
    {
        int max = Math.Max(pixel.R, Math.Max(pixel.G, pixel.B));
        int min = Math.Min(pixel.R, Math.Min(pixel.G, pixel.B));
        double brightness = max / 255.0;
        double saturation = max == 0 ? 0 : (max - min) / (double)max;

        if (saturation < MinSaturation)
        {
            if (brightness >= WhiteMinBrightness) return Colors.White;
            if (brightness <= BlackMaxBrightness) return Colors.Black;
            return Colors.Grey;
        }

        // Hue in degrees: 0 = red, 60 = yellow, 120 = green, 240 = blue
        double delta = max - min;
        double hue;
        if (max == pixel.R) hue = 60 * ((pixel.G - pixel.B) / delta);
        else if (max == pixel.G) hue = 60 * ((pixel.B - pixel.R) / delta + 2);
        else hue = 60 * ((pixel.R - pixel.G) / delta + 4);
        if (hue < 0) hue += 360;

        if (hue < 15 || hue >= 300) return Colors.Red;
        if (hue < 40) return Colors.Orange;
        if (hue < 70) return Colors.Yellow;
        if (hue < 170) return Colors.Green;
        return Colors.Blue;
    }

    public bool ColorPresent(Colors color)
    {
        foreach (Rgb pixel in pixels)
        {
            if (ClosestColor(pixel).Equals(color))
            {
                return true;
            }
        }
        return false;
    }
}
