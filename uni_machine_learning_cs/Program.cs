using uni_machine_learning_cs;
using uni_machine_learning_cs.ImageAnalysis;

string[] files = Directory.GetFiles("80x60+3", "*.bmp");
Array.Sort(files);

foreach (string file in files)
{
    ReadImageFile fileContent = new ReadImageFile(file);

    Console.WriteLine($"=== {Path.GetFileName(file)}");
    Console.WriteLine($"Size: {fileContent.width}x{fileContent.height}");
    // Color switches
    Console.WriteLine($"Vertical Color switches: " + fileContent.VerticalLineColorSwitches());
    Console.WriteLine($"Horizontal Color switches: " + fileContent.HorizontalLineColorSwitches());
    // presence of colors
    Console.WriteLine($"Red present: " + fileContent.ColorPresent(Colors.Red));
    Console.WriteLine($"Blue present: " + fileContent.ColorPresent(Colors.Blue));
    Console.WriteLine($"White present: " + fileContent.ColorPresent(Colors.White));
    Console.WriteLine($"Yellow present: " + fileContent.ColorPresent(Colors.Yellow));
    Console.WriteLine($"Green present: " + fileContent.ColorPresent(Colors.Green));
    Console.WriteLine();
}
