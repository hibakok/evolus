namespace evolus.Core;

/// <summary>
/// Represents fitness evaluation result with error and complexity
/// </summary>
public class FitnessResult
{
    public decimal Error { get; set; }
    public int Complexity { get; set; }
    
    /// <summary>
    /// A neural network is more fit if:
    /// 1. It has lower error (primary)
    /// 2. If errors are equal, lower complexity is better
    /// BUT: If a network has higher complexity but even slightly lower error, it's considered more fit
    /// This means error is the ONLY factor for fitness comparison, complexity is only a tiebreaker
    /// </summary>
    public bool IsMoreFitThan(FitnessResult other)
    {
        // Direct comparison: lower error is always better, regardless of complexity
        // Complexity only matters when errors are exactly equal
        if (this.Error < other.Error)
            return true;
        if (this.Error > other.Error)
            return false;
        
        // Errors are equal - lower complexity wins
        return this.Complexity < other.Complexity;
    }
    
    public override string ToString()
    {
        return $"Error={Error}, Complexity={Complexity}";
    }
}

/// <summary>
/// Evaluates the fitness of a neural network
/// </summary>
public class FitnessEvaluator
{
    private readonly Data.TrainingDataManager _trainingData;

    public FitnessEvaluator(Data.TrainingDataManager trainingData)
    {
        _trainingData = trainingData;
    }

    /// <summary>
    /// Calculates fitness by running the network on all training pairs
    /// Returns error (sum of squared differences) and complexity
    /// Lower error = better fitness. 0 error = perfect.
    /// </summary>
    public FitnessResult Evaluate(Core.NeuralNetwork network)
    {
        if (_trainingData.DataPairs.Count == 0)
        {
            return new FitnessResult { Error = 0m, Complexity = network.GetComplexity() };
        }

        decimal totalError = 0m;

        foreach (var dataPair in _trainingData.DataPairs)
        {
            var outputs = network.Run(dataPair.Input);
            
            // Calculate sum of squared errors with high precision
            for (int i = 0; i < dataPair.ExpectedOutput.Length; i++)
            {
                decimal diff = outputs[i] - dataPair.ExpectedOutput[i];
                totalError += diff * diff;
            }
        }

        return new FitnessResult
        {
            Error = totalError,
            Complexity = network.GetComplexity()
        };
    }
}
