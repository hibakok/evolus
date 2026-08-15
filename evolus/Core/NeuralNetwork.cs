namespace evolus.Core;

/// <summary>
/// Represents a neural network without layer structure - any neuron can connect to any other
/// </summary>
public class NeuralNetwork
{
    public List<Neuron> Neurons { get; set; } = new();
    public List<Connection> Connections { get; set; } = new();
    
    public int InputCount { get; set; }
    public int OutputCount { get; set; }
    
    // IDs of input neurons (first InputCount neurons)
    private List<int> InputNeuronIds => Neurons.Take(InputCount).Select(n => n.Id).ToList();
    
    // IDs of output neurons (last OutputCount neurons)
    private List<int> OutputNeuronIds => Neurons.Skip(Math.Max(0, Neurons.Count - OutputCount)).Take(OutputCount).Select(n => n.Id).ToList();

    /// <summary>
    /// Creates a completely empty neural network
    /// </summary>
    public static NeuralNetwork CreateEmpty()
    {
        return new NeuralNetwork();
    }

    /// <summary>
    /// Initializes the network with input and output neurons (no hidden neurons, no connections)
    /// </summary>
    public void Initialize(int inputCount, int outputCount)
    {
        Neurons.Clear();
        Connections.Clear();
        InputCount = inputCount;
        OutputCount = outputCount;
        
        // Create input neurons
        for (int i = 0; i < inputCount; i++)
        {
            Neurons.Add(new Neuron 
            { 
                Id = i, 
                ActivationFunction = ActivationFunctionType.Linear 
            });
        }
        
        // Create output neurons
        for (int i = 0; i < outputCount; i++)
        {
            Neurons.Add(new Neuron 
            { 
                Id = inputCount + i,
                ActivationFunction = ActivationFunctionType.Sigmoid 
            });
        }
    }

    /// <summary>
    /// Runs the neural network with given inputs and returns outputs
    /// </summary>
    public decimal[] Run(decimal[] inputs)
    {
        if (inputs.Length != InputCount)
            throw new ArgumentException($"Expected {InputCount} inputs, got {inputs.Length}");
        
        if (Neurons.Count == 0)
            return Enumerable.Repeat(0m, OutputCount).ToArray();
        
        // Set input values
        var neuronValues = new Dictionary<int, decimal>();
        for (int i = 0; i < InputCount; i++)
        {
            neuronValues[i] = inputs[i];
        }
        
        // Iterate until convergence or max iterations
        int maxIterations = 100;
        for (int iter = 0; iter < maxIterations; iter++)
        {
            bool changed = false;
            
            // Process non-input neurons in order
            for (int i = InputCount; i < Neurons.Count; i++)
            {
                var neuron = Neurons[i];
                
                // Sum all incoming connections
                decimal sum = 0m;
                foreach (var conn in Connections.Where(c => c.ToNeuronId == neuron.Id))
                {
                    if (neuronValues.ContainsKey(conn.FromNeuronId))
                    {
                        sum += neuronValues[conn.FromNeuronId] * conn.Weight;
                    }
                }
                
                // Apply activation function
                decimal newValue = neuron.Activate(sum);
                
                if (!neuronValues.ContainsKey(neuron.Id) || neuronValues[neuron.Id] != newValue)
                {
                    neuronValues[neuron.Id] = newValue;
                    changed = true;
                }
            }
            
            if (!changed)
                break;
        }
        
        // Get output values
        var outputs = new decimal[OutputCount];
        for (int i = 0; i < OutputCount; i++)
        {
            int outputNeuronId = Neurons.Count - OutputCount + i;
            if (outputNeuronId >= 0 && outputNeuronId < Neurons.Count && neuronValues.ContainsKey(outputNeuronId))
            {
                outputs[i] = neuronValues[outputNeuronId];
            }
            else
            {
                outputs[i] = 0m;
            }
        }
        
        return outputs;
    }

    /// <summary>
    /// Calculates computational complexity (number of connections)
    /// </summary>
    public int GetComplexity()
    {
        return Connections.Count;
    }

    /// <summary>
    /// Creates a deep clone of this neural network
    /// </summary>
    public NeuralNetwork Clone()
    {
        var clone = new NeuralNetwork
        {
            InputCount = InputCount,
            OutputCount = OutputCount
        };
        
        foreach (var neuron in Neurons)
        {
            clone.Neurons.Add(neuron.Clone());
        }
        
        foreach (var connection in Connections)
        {
            clone.Connections.Add(connection.Clone());
        }
        
        return clone;
    }

    /// <summary>
    /// Saves the neural network to a text file
    /// </summary>
    public void SaveToFile(string filePath)
    {
        var sb = new System.Text.StringBuilder();
        
        // Header: inputCount outputCount
        sb.AppendLine($"{InputCount} {OutputCount}");
        
        // Neurons section
        sb.AppendLine("# NEURONS");
        sb.AppendLine($"# Id ActivationFunction Bias");
        foreach (var neuron in Neurons)
        {
            sb.AppendLine($"{neuron.Id} {(int)neuron.ActivationFunction} {neuron.Bias}");
        }
        
        // Connections section
        sb.AppendLine("# CONNECTIONS");
        sb.AppendLine($"# FromNeuronId ToNeuronId Weight");
        foreach (var conn in Connections)
        {
            sb.AppendLine($"{conn.FromNeuronId} {conn.ToNeuronId} {conn.Weight}");
        }
        
        File.WriteAllText(filePath, sb.ToString());
    }

    /// <summary>
    /// Loads a neural network from a text file
    /// </summary>
    public static NeuralNetwork LoadFromFile(string filePath)
    {
        var lines = File.ReadAllLines(filePath);
        var network = new NeuralNetwork();
        
        string? section = null;
        
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            
            if (trimmed.StartsWith("#"))
            {
                if (trimmed.Contains("NEURONS"))
                    section = "neurons";
                else if (trimmed.Contains("CONNECTIONS"))
                    section = "connections";
                continue;
            }
            
            if (string.IsNullOrWhiteSpace(trimmed))
                continue;
            
            var parts = trimmed.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            
            // First line: inputCount outputCount
            if (section == null && parts.Length >= 2)
            {
                network.InputCount = int.Parse(parts[0]);
                network.OutputCount = int.Parse(parts[1]);
                continue;
            }
            
            if (section == "neurons" && parts.Length >= 3)
            {
                var neuron = new Neuron
                {
                    Id = int.Parse(parts[0]),
                    ActivationFunction = (ActivationFunctionType)int.Parse(parts[1]),
                    Bias = decimal.Parse(parts[2])
                };
                network.Neurons.Add(neuron);
            }
            else if (section == "connections" && parts.Length >= 3)
            {
                var conn = new Connection
                {
                    FromNeuronId = int.Parse(parts[0]),
                    ToNeuronId = int.Parse(parts[1]),
                    Weight = decimal.Parse(parts[2])
                };
                network.Connections.Add(conn);
            }
        }
        
        return network;
    }

    public override string ToString()
    {
        return $"NeuralNetwork(Neurons={Neurons.Count}, Connections={Connections.Count}, Inputs={InputCount}, Outputs={OutputCount})";
    }
}
