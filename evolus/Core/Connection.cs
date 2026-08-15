namespace evolus.Core;

/// <summary>
/// Represents a connection between two neurons with a weight
/// </summary>
public class Connection
{
    public int FromNeuronId { get; set; }
    public int ToNeuronId { get; set; }
    public decimal Weight { get; set; }

    public Connection Clone()
    {
        return new Connection
        {
            FromNeuronId = FromNeuronId,
            ToNeuronId = ToNeuronId,
            Weight = Weight
        };
    }

    public override string ToString()
    {
        return $"{FromNeuronId} -> {ToNeuronId} : {Weight}";
    }
}
