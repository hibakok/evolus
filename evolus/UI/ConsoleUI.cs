namespace evolus.UI;

/// <summary>
/// Console-based user interface for the evolus application
/// </summary>
public class ConsoleUI
{
    private readonly Config.EvolutionConfig _config;
    private readonly Data.TrainingDataManager _trainingData;
    private Core.FitnessEvaluator? _fitnessEvaluator;
    private Evolution.PopulationManager? _populationManager;
    private bool _isInitialized = false;

    public ConsoleUI(Config.EvolutionConfig config, Data.TrainingDataManager trainingData)
    {
        _config = config;
        _trainingData = trainingData;
    }

    public void Initialize()
    {
        // Load training data
        try
        {
            _trainingData.LoadFromFile(_config.TrainingDataPath);
            Console.WriteLine($"Loaded {_trainingData.DataPairs.Count} training pairs (Input dim: {_trainingData.InputDimension}, Output dim: {_trainingData.OutputDimension})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading training data: {ex.Message}");
            Console.WriteLine("Please ensure the training data file exists and is properly formatted.");
            return;
        }

        // Initialize fitness evaluator
        _fitnessEvaluator = new Core.FitnessEvaluator(_trainingData);

        // Initialize or load population
        _populationManager = new Evolution.PopulationManager(_fitnessEvaluator, _config.RandomSeed);
        
        if (Directory.Exists(_config.PopulationSavePath))
        {
            _populationManager.LoadPopulation(_config.PopulationSavePath);
            if (_populationManager.InnerPopulation.Count > 0)
            {
                Console.WriteLine($"Loaded existing population with {_populationManager.InnerPopulation.Count} individuals");
            }
            else
            {
                InitializeNewPopulation();
            }
        }
        else
        {
            InitializeNewPopulation();
        }

        _isInitialized = true;
    }

    private void InitializeNewPopulation()
    {
        _populationManager!.InitializeInnerPopulation(
            _config.InnerPopulationSize,
            _trainingData.InputDimension,
            _trainingData.OutputDimension
        );
        Console.WriteLine($"Initialized new population with {_config.InnerPopulationSize} individuals");
    }

    public void Run()
    {
        if (!_isInitialized)
        {
            Console.WriteLine("System not initialized. Please check configuration and training data.");
            return;
        }

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("=== EVOLUS - Evolutionary Neural Network ===");
            Console.WriteLine("1. Run Evolution");
            Console.WriteLine("2. Test Current Best Individual");
            Console.WriteLine("3. View Population Status");
            Console.WriteLine("4. Save Population");
            Console.WriteLine("5. Exit");
            Console.Write("Select option: ");

            var input = Console.ReadLine()?.Trim();
            
            switch (input)
            {
                case "1":
                    RunEvolution();
                    break;
                case "2":
                    TestBestIndividual();
                    break;
                case "3":
                    ViewPopulationStatus();
                    break;
                case "4":
                    SavePopulation();
                    break;
                case "5":
                    SavePopulation();
                    Console.WriteLine("Goodbye!");
                    return;
                default:
                    Console.WriteLine("Invalid option. Please try again.");
                    break;
            }
        }
    }

    private void RunEvolution()
    {
        Console.Write("Enter number of generations to run (or press Enter for default from config): ");
        var input = Console.ReadLine()?.Trim();
        
        int generations;
        if (string.IsNullOrEmpty(input) || !int.TryParse(input, out generations))
        {
            generations = _config.GenerationsToRun;
            Console.WriteLine($"Using default: {generations} generations");
        }

        Console.WriteLine();
        Console.WriteLine($"Starting evolution for {generations} generations...");
        Console.WriteLine($"Configuration: Population={_config.InnerPopulationSize}, Offspring/Individual={_config.OffspringPerIndividual}, Mutations={_config.MutationsPerOffspring}");
        Console.WriteLine();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        for (int gen = 1; gen <= generations; gen++)
        {
            // Generate offspring
            _populationManager!.GenerateOffspring(_config.OffspringPerIndividual, _config.MutationsPerOffspring);
            
            // Evaluate offspring
            _populationManager.EvaluateOuterPopulation();
            
            // Update inner population with better offspring
            _populationManager.UpdateInnerPopulation();
            
            // Progress report every 10 generations
            if (gen % 10 == 0 || gen == generations)
            {
                var best = _populationManager.GetBestIndividual();
                Console.WriteLine($"Generation {gen}: Best Error={best?.Fitness?.Error.ToString("F6") ?? "N/A"}, Complexity={best?.Network.GetComplexity() ?? 0}");
            }
        }
        
        stopwatch.Stop();
        
        Console.WriteLine();
        Console.WriteLine("=== Evolution Complete ===");
        Console.WriteLine($"Total time: {stopwatch.ElapsedMilliseconds / 1000.0:F2} seconds");
        
        var finalBest = _populationManager.GetBestIndividual();
        if (finalBest != null)
        {
            Console.WriteLine($"Best individual: Error={finalBest.Fitness?.Error.ToString("F10") ?? "N/A"}, Complexity={finalBest.Network.GetComplexity()}");
            Console.WriteLine($"Network structure: {finalBest.Network.Neurons.Count} neurons, {finalBest.Network.Connections.Count} connections");
        }
        
        Console.WriteLine();
        Console.WriteLine("Press any key to continue...");
        try
        {
            if (Console.KeyAvailable || Console.IsInputRedirected == false)
                Console.ReadKey(true);
        }
        catch
        {
            // Ignore console read errors when running in non-interactive mode
        }
    }

    private void TestBestIndividual()
    {
        var best = _populationManager!.GetBestIndividual();
        if (best == null)
        {
            Console.WriteLine("No individuals available. Run evolution first.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine("=== Testing Best Individual ===");
        Console.WriteLine($"Network: {best.Network.Neurons.Count} neurons, {best.Network.Connections.Count} connections");
        Console.WriteLine("Enter input values separated by spaces (or 'q' to quit):");

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine()?.Trim();
            
            if (input?.ToLower() == "q")
                break;
            
            if (string.IsNullOrEmpty(input))
                continue;
            
            try
            {
                var values = input.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(decimal.Parse).ToArray();
                
                if (values.Length != _trainingData.InputDimension)
                {
                    Console.WriteLine($"Error: Expected {_trainingData.InputDimension} input values, got {values.Length}");
                    continue;
                }
                
                var outputs = best.Network.Run(values);
                
                Console.WriteLine($"Output: {string.Join(" ", outputs.Select(o => o.ToString("F6")))}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }

    private void ViewPopulationStatus()
    {
        Console.WriteLine();
        Console.WriteLine("=== Population Status ===");
        Console.WriteLine($"Inner Population Size: {_populationManager!.InnerPopulation.Count}");
        
        // Evaluate all if needed
        foreach (var individual in _populationManager.InnerPopulation.Where(i => i.Fitness == null))
        {
            individual.Fitness = new Core.FitnessEvaluator(_trainingData).Evaluate(individual.Network);
        }
        
        var sorted = _populationManager.InnerPopulation.OrderBy(i => i.Fitness!.Error).ToList();
        
        Console.WriteLine();
        Console.WriteLine("Top 5 individuals:");
        for (int i = 0; i < Math.Min(5, sorted.Count); i++)
        {
            var ind = sorted[i];
            Console.WriteLine($"  #{ind.Id}: Error={ind.Fitness!.Error.ToString("F6")}, Complexity={ind.Fitness.Complexity}, Neurons={ind.Network.Neurons.Count}, Connections={ind.Network.Connections.Count}");
        }
    }

    private void SavePopulation()
    {
        _populationManager!.SavePopulation(_config.PopulationSavePath);
        Console.WriteLine($"Population saved to {_config.PopulationSavePath}");
    }
}
