namespace uni_machine_learning_cs;

// Traffic sign colours (German StVO / RAL "traffic" colours), as approximate sRGB values
public struct Colors
{
    public static readonly Colors Red = new Colors(193, 18, 28);       // RAL 3020 traffic red
    public static readonly Colors Blue = new Colors(0, 91, 140);       // RAL 5017 traffic blue
    public static readonly Colors Yellow = new Colors(247, 181, 0);    // RAL 1023 traffic yellow
    public static readonly Colors Green = new Colors(0, 135, 84);      // RAL 6024 traffic green
    public static readonly Colors Orange = new Colors(226, 83, 3);     // RAL 2009 traffic orange
    public static readonly Colors White = new Colors(246, 246, 246);   // RAL 9016 traffic white
    public static readonly Colors Black = new Colors(30, 30, 30);      // RAL 9017 traffic black
    public static readonly Colors Grey = new Colors(143, 150, 149);    // RAL 7042 traffic grey A

    public static readonly Colors[] All = [Red, Blue, Yellow, Green, Orange, White, Black, Grey];

    public readonly byte R;
    public readonly byte G;
    public readonly byte B;

    private Colors(byte r, byte g, byte b)
    {
        R = r;
        G = g;
        B = b;
    }
}
