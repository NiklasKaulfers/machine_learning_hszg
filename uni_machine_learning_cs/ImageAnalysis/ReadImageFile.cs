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
}
