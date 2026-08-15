namespace evolus.Core;

/// <summary>
/// Available activation functions for neurons
/// </summary>
public enum ActivationFunctionType
{
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
    Step,
    Gaussian
}

/// <summary>
/// Represents a neuron in the neural network
/// </summary>
public class Neuron
{
    public int Id { get; set; }
    public ActivationFunctionType ActivationFunction { get; set; } = ActivationFunctionType.Sigmoid;
    public decimal Bias { get; set; } = 0m;

    public Neuron Clone()
    {
        return new Neuron
        {
            Id = Id,
            ActivationFunction = ActivationFunction,
            Bias = Bias
        };
    }

    /// <summary>
    /// Applies the activation function to the input value
    /// </summary>
    public decimal Activate(decimal value)
    {
        decimal biasedValue = value + Bias;
        
        return this.ActivationFunction switch
        {
            Core.ActivationFunctionType.Sigmoid => Sigmoid(biasedValue),
            Core.ActivationFunctionType.Tanh => Tanh(biasedValue),
            Core.ActivationFunctionType.ReLU => ReLU(biasedValue),
            Core.ActivationFunctionType.Linear => biasedValue,
            Core.ActivationFunctionType.Step => Step(biasedValue),
            Core.ActivationFunctionType.Gaussian => Gaussian(biasedValue),
            _ => Sigmoid(biasedValue)
        };
    }

    private static decimal Sigmoid(decimal x)
    {
        // Using high precision calculation
        if (x < -500m) return 0m;
        if (x > 500m) return 1m;
        
        double xd = (double)x;
        double result = 1.0 / (1.0 + Math.Exp(-xd));
        return (decimal)result;
    }

    private static decimal Tanh(decimal x)
    {
        if (x < -500m) return -1m;
        if (x > 500m) return 1m;
        
        double xd = (double)x;
        double result = Math.Tanh(xd);
        return (decimal)result;
    }

    private static decimal ReLU(decimal x)
    {
        return x > 0m ? x : 0m;
    }

    private static decimal Step(decimal x)
    {
        return x >= 0m ? 1m : 0m;
    }

    private static decimal Gaussian(decimal x)
    {
        double xd = (double)x;
        double result = Math.Exp(-xd * xd);
        return (decimal)result;
    }
}
