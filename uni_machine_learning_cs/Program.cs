using uni_machine_learning_cs.ImageAnalysis;

ReadImageFile fileContent = new ReadImageFile("80x60+3/X-10Y-10.bmp");
Console.WriteLine($"Size: {fileContent.width}x{fileContent.height}");

for (int y = 0; y < fileContent.height; y++)
{
    for (int x = 0; x < fileContent.width; x++)
    {
        Rgb pixel = fileContent.GetPixel(x, y);
        Console.WriteLine($"({x}, {y}): R={pixel.R} G={pixel.G} B={pixel.B}");
    }
}
