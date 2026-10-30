namespace uni_machine_learning_cs.MachineLearning;

// Writing and reading arrays in model files: the length first, then the values
public static class BinaryExtensions
{
    public static void WriteArray(this BinaryWriter writer, double[] values)
    {
        writer.Write(values.Length);
        foreach (double value in values)
            writer.Write(value);
    }

    public static double[] ReadDoubleArray(this BinaryReader reader)
    {
        double[] values = new double[reader.ReadInt32()];
        for (int i = 0; i < values.Length; i++)
            values[i] = reader.ReadDouble();
        return values;
    }

    public static void WriteArray(this BinaryWriter writer, int[] values)
    {
        writer.Write(values.Length);
        foreach (int value in values)
            writer.Write(value);
    }

    public static int[] ReadIntArray(this BinaryReader reader)
    {
        int[] values = new int[reader.ReadInt32()];
        for (int i = 0; i < values.Length; i++)
            values[i] = reader.ReadInt32();
        return values;
    }

    public static void WriteArray(this BinaryWriter writer, string[] values)
    {
        writer.Write(values.Length);
        foreach (string value in values)
            writer.Write(value);
    }

    public static string[] ReadStringArray(this BinaryReader reader)
    {
        string[] values = new string[reader.ReadInt32()];
        for (int i = 0; i < values.Length; i++)
            values[i] = reader.ReadString();
        return values;
    }
}
