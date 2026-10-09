namespace uni_machine_learning_cs.Vectors;

public class ValueVector
{
    public readonly float value;
    public readonly ValueContext context;
    public ValueVector(float value, ValueContext context)
    {
        this.value = value;
        this.context = context;
    }
}