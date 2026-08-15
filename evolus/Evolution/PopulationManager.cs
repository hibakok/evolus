namespace evolus.Evolution;

/// <summary>
/// Represents an individual in the population with its neural network and fitness
/// </summary>
public class Individual
{
    public Core.NeuralNetwork Network { get; set; }
    public Core.FitnessResult? Fitness { get; set; }
    public int? ParentId { get; set; } // ID of the parent in inner population (null for inner population members)
    public int Id { get; set; }

    public Individual(Core.NeuralNetwork network, int id, int? parentId = null)
    {
        Network = network;
        Id = id;
        ParentId = parentId;
    }

    public Individual Clone()
    {
        return new Individual(Network.Clone(), Id, ParentId)
        {
            Fitness = Fitness != null ? new Core.FitnessResult { Error = Fitness.Error, Complexity = Fitness.Complexity } : null
        };
    }
}

/// <summary>
/// Manages the evolutionary process with inner and outer populations
/// </summary>
public class PopulationManager
{
    private readonly Core.FitnessEvaluator _fitnessEvaluator;
    private readonly Random _random;
    
    // Inner population - immutable individuals that produce mutants
    public List<Individual> InnerPopulation { get; set; } = new();
    
    // Outer population - mutant offspring that compete to replace their parents
    public List<Individual> OuterPopulation { get; set; } = new();
    
    private int _nextInnerId = 0;
    private int _nextOuterId = 0;

    public PopulationManager(Core.FitnessEvaluator fitnessEvaluator, int seed = 42)
    {
        _fitnessEvaluator = fitnessEvaluator;
        _random = new Random(seed);
    }

    /// <summary>
    /// Initializes the inner population with empty neural networks
    /// </summary>
    public void InitializeInnerPopulation(int size, int inputCount, int outputCount)
    {
        InnerPopulation.Clear();
        _nextInnerId = 0;
        
        for (int i = 0; i < size; i++)
        {
            var network = new Core.NeuralNetwork();
            network.Initialize(inputCount, outputCount);
            
            InnerPopulation.Add(new Individual(network, _nextInnerId++));
        }
    }

    /// <summary>
    /// Generates mutant offspring from inner population to outer population
    /// </summary>
    public void GenerateOffspring(int offspringPerIndividual, int mutationCount)
    {
        OuterPopulation.Clear();
        _nextOuterId = 10000; // Start outer IDs from a high number to distinguish
        
        foreach (var parent in InnerPopulation)
        {
            for (int i = 0; i < offspringPerIndividual; i++)
            {
                var offspring = parent.Clone();
                offspring.Id = _nextOuterId++;
                offspring.ParentId = parent.Id;
                
                // Apply mutations
                ApplyMutations(offspring.Network, mutationCount);
                
                OuterPopulation.Add(offspring);
            }
        }
    }

    /// <summary>
    /// Applies specified number of minimal mutations to a neural network
    /// </summary>
    private void ApplyMutations(Core.NeuralNetwork network, int mutationCount)
    {
        for (int i = 0; i < mutationCount; i++)
        {
            var mutationType = _random.Next(5); // 0-4: 5 types of mutations
            
            switch (mutationType)
            {
                case 0: // Change weight
                    MutateWeight(network);
                    break;
                case 1: // Add connection
                    AddConnection(network);
                    break;
                case 2: // Remove connection
                    RemoveConnection(network);
                    break;
                case 3: // Add neuron
                    AddNeuron(network);
                    break;
                case 4: // Remove neuron
                    RemoveNeuron(network);
                    break;
            }
        }
    }

    private void MutateWeight(Core.NeuralNetwork network)
    {
        if (network.Connections.Count == 0)
            return;
        
        var connIndex = _random.Next(network.Connections.Count);
        var connection = network.Connections[connIndex];
        
        // Small random change to weight
        decimal change = (decimal)(_random.NextDouble() * 2 - 1) * 0.5m; // -0.5 to 0.5
        connection.Weight += change;
    }

    private void AddConnection(Core.NeuralNetwork network)
    {
        if (network.Neurons.Count < 2)
            return;
        
        // Check if we already have maximum connections
        int maxConnections = network.Neurons.Count * (network.Neurons.Count - 1);
        if (network.Connections.Count >= maxConnections)
            return;
        
        int attempts = 0;
        while (attempts < 100)
        {
            var fromNeuron = network.Neurons[_random.Next(network.Neurons.Count)];
            var toNeuron = network.Neurons[_random.Next(network.Neurons.Count)];
            
            // No self-connections
            if (fromNeuron.Id == toNeuron.Id)
            {
                attempts++;
                continue;
            }
            
            // Check if connection already exists
            if (network.Connections.Any(c => c.FromNeuronId == fromNeuron.Id && c.ToNeuronId == toNeuron.Id))
            {
                attempts++;
                continue;
            }
            
            network.Connections.Add(new Core.Connection
            {
                FromNeuronId = fromNeuron.Id,
                ToNeuronId = toNeuron.Id,
                Weight = (decimal)(_random.NextDouble() * 2 - 1) // Random weight -1 to 1
            });
            return;
        }
    }

    private void RemoveConnection(Core.NeuralNetwork network)
    {
        if (network.Connections.Count == 0)
            return;
        
        var connIndex = _random.Next(network.Connections.Count);
        network.Connections.RemoveAt(connIndex);
    }

    private void AddNeuron(Core.NeuralNetwork network)
    {
        // Don't add too many neurons
        if (network.Neurons.Count >= 100)
            return;
        
        int newId = network.Neurons.Count > 0 ? network.Neurons.Max(n => n.Id) + 1 : 0;
        
        var activationFunctions = Enum.GetValues(typeof(Core.ActivationFunctionType));
        var randomActivation = (Core.ActivationFunctionType)_random.Next(activationFunctions.Length);
        
        network.Neurons.Add(new Core.Neuron
        {
            Id = newId,
            ActivationFunction = randomActivation,
            Bias = 0m
        });
    }

    private void RemoveNeuron(Core.NeuralNetwork network)
    {
        // Keep at least input + output neurons
        int minNeurons = network.InputCount + network.OutputCount;
        if (network.Neurons.Count <= minNeurons)
            return;
        
        // Don't remove input or output neurons
        var removableNeurons = network.Neurons
            .Where(n => n.Id >= network.InputCount && n.Id < network.Neurons.Count - network.OutputCount)
            .ToList();
        
        if (removableNeurons.Count == 0)
            return;
        
        var neuronToRemove = removableNeurons[_random.Next(removableNeurons.Count)];
        
        // Remove all connections to/from this neuron
        network.Connections.RemoveAll(c => c.FromNeuronId == neuronToRemove.Id || c.ToNeuronId == neuronToRemove.Id);
        network.Neurons.Remove(neuronToRemove);
    }

    /// <summary>
    /// Evaluates fitness of all individuals in outer population
    /// </summary>
    public void EvaluateOuterPopulation()
    {
        foreach (var individual in OuterPopulation)
        {
            individual.Fitness = _fitnessEvaluator.Evaluate(individual.Network);
        }
    }

    /// <summary>
    /// Replaces inner population members with better offspring from outer population
    /// Only replaces if offspring has strictly better fitness than parent
    /// </summary>
    public void UpdateInnerPopulation()
    {
        // Group offspring by parent
        var offspringByParent = OuterPopulation
            .Where(o => o.ParentId.HasValue)
            .GroupBy(o => o.ParentId.Value)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        foreach (var parent in InnerPopulation.ToList())
        {
            if (!offspringByParent.ContainsKey(parent.Id))
                continue;
            
            // Evaluate parent's current fitness
            parent.Fitness = _fitnessEvaluator.Evaluate(parent.Network);
            
            // Find the best offspring
            var bestOffspring = offspringByParent[parent.Id]
                .OrderByDescending(o => o.Fitness!.Error) // Lower error is better, so we want minimum
                .FirstOrDefault(o => o.Fitness!.IsMoreFitThan(parent.Fitness!));
            
            if (bestOffspring != null && bestOffspring.Fitness!.IsMoreFitThan(parent.Fitness!))
            {
                // Replace parent with offspring
                var index = InnerPopulation.IndexOf(parent);
                var newIndividual = bestOffspring.Clone();
                newIndividual.Id = parent.Id; // Keep parent's ID
                newIndividual.ParentId = null; // It's now part of inner population
                InnerPopulation[index] = newIndividual;
            }
        }
    }

    /// <summary>
    /// Gets the best individual from inner population
    /// </summary>
    public Individual? GetBestIndividual()
    {
        if (InnerPopulation.Count == 0)
            return null;
        
        // Evaluate all if not already done
        foreach (var individual in InnerPopulation.Where(i => i.Fitness == null))
        {
            individual.Fitness = _fitnessEvaluator.Evaluate(individual.Network);
        }
        
        return InnerPopulation.OrderBy(i => i.Fitness!.Error).ThenBy(i => i.Fitness!.Complexity).First();
    }

    /// <summary>
    /// Saves inner population to file
    /// </summary>
    public void SavePopulation(string directoryPath)
    {
        if (!Directory.Exists(directoryPath))
            Directory.CreateDirectory(directoryPath);
        
        // Save metadata
        var metadataPath = Path.Combine(directoryPath, "population_meta.txt");
        var metaSb = new System.Text.StringBuilder();
        metaSb.AppendLine($"InnerPopulationSize={InnerPopulation.Count}");
        metaSb.AppendLine($"NextInnerId={_nextInnerId}");
        
        foreach (var individual in InnerPopulation)
        {
            metaSb.AppendLine($"Individual_{individual.Id}.fitness_error={individual.Fitness?.Error.ToString() ?? "null"}");
            metaSb.AppendLine($"Individual_{individual.Id}.fitness_complexity={individual.Fitness?.Complexity.ToString() ?? "null"}");
        }
        
        File.WriteAllText(metadataPath, metaSb.ToString());
        
        // Save each individual's network
        foreach (var individual in InnerPopulation)
        {
            var networkPath = Path.Combine(directoryPath, $"network_{individual.Id}.txt");
            individual.Network.SaveToFile(networkPath);
        }
    }

    /// <summary>
    /// Loads inner population from file
    /// </summary>
    public void LoadPopulation(string directoryPath)
    {
        if (!Directory.Exists(directoryPath))
            return;
        
        var metadataPath = Path.Combine(directoryPath, "population_meta.txt");
        if (!File.Exists(metadataPath))
            return;
        
        InnerPopulation.Clear();
        
        var metaLines = File.ReadAllLines(metadataPath);
        var fitnessData = new Dictionary<int, Core.FitnessResult>();
        
        foreach (var line in metaLines)
        {
            if (line.StartsWith("NextInnerId="))
            {
                _nextInnerId = int.Parse(line.Split('=')[1]);
            }
            else if (line.Contains(".fitness_error="))
            {
                var parts = line.Split('=');
                var idPart = parts[0].Replace("Individual_", "").Replace(".fitness_error", "");
                if (int.TryParse(idPart, out int id))
                {
                    if (!fitnessData.ContainsKey(id))
                        fitnessData[id] = new Core.FitnessResult();
                    
                    if (parts[1] != "null")
                        fitnessData[id].Error = decimal.Parse(parts[1]);
                }
            }
            else if (line.Contains(".fitness_complexity="))
            {
                var parts = line.Split('=');
                var idPart = parts[0].Replace("Individual_", "").Replace(".fitness_complexity", "");
                if (int.TryParse(idPart, out int id))
                {
                    if (!fitnessData.ContainsKey(id))
                        fitnessData[id] = new Core.FitnessResult();
                    
                    if (parts[1] != "null")
                        fitnessData[id].Complexity = int.Parse(parts[1]);
                }
            }
        }
        
        // Load networks
        var networkFiles = Directory.GetFiles(directoryPath, "network_*.txt");
        foreach (var networkFile in networkFiles)
        {
            var fileName = Path.GetFileNameWithoutExtension(networkFile);
            var idPart = fileName.Replace("network_", "");
            
            if (int.TryParse(idPart, out int id))
            {
                var network = Core.NeuralNetwork.LoadFromFile(networkFile);
                var individual = new Individual(network, id);
                
                if (fitnessData.ContainsKey(id))
                {
                    individual.Fitness = fitnessData[id];
                }
                
                InnerPopulation.Add(individual);
            }
        }
        
        InnerPopulation.Sort((a, b) => a.Id.CompareTo(b.Id));
    }
}
