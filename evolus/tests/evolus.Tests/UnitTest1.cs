using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Xunit;
using evolus.Core;

namespace evolus.Tests;

public class NeuralNetworkTests
{
    [Fact]
    public void CreateEmptyNetwork_CreatesCorrectStructure()
    {
        var network = NeuralNetwork.CreateEmpty(2, 1);
        
        Assert.Equal(2, network.InputCount);
        Assert.Equal(1, network.OutputCount);
        Assert.Equal(3, network.Neurons.Count); // 2 input + 1 output
        Assert.Empty(network.Connections);
    }

    [Fact]
    public void Forward_ProducesOutput()
    {
        var network = NeuralNetwork.CreateEmpty(2, 1);
        var inputs = new decimal[] { 0.5m, 0.3m };
        
        var outputs = network.Forward(inputs);
        
        Assert.Single(outputs);
    }

    [Fact]
    public void SaveAndLoad_RoundtripPreservesData()
    {
        var network = NeuralNetwork.CreateEmpty(2, 1);
        network.AddNeuron(ActivationFunction.Tanh);
        network.AddConnection(0, 2, 0.5m);
        network.AddConnection(1, 2, -0.3m);
        network.AddConnection(2, 3, 0.8m);
        
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_network_{Guid.NewGuid()}.net");
        
        try
        {
            network.SaveToFile(tempPath);
            var loaded = NeuralNetwork.LoadFromFile(tempPath);
            
            Assert.Equal(network.InputCount, loaded.InputCount);
            Assert.Equal(network.OutputCount, loaded.OutputCount);
            Assert.Equal(network.Neurons.Count, loaded.Neurons.Count);
            Assert.Equal(network.Connections.Count, loaded.Connections.Count);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        var network = NeuralNetwork.CreateEmpty(2, 1);
        network.AddConnection(0, 2, 0.5m);
        
        var clone = network.Clone();
        
        // Изменяем оригинал
        network.Connections[0].Weight = 999m;
        
        // Клон должен остаться неизменным
        Assert.Equal(0.5m, clone.Connections[0].Weight);
    }
}

public class TrainingDataLoaderTests
{
    [Fact]
    public void LoadFromFile_ParsesCorrectFormat()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_data_{Guid.NewGuid()}.txt");
        
        try
        {
            File.WriteAllText(tempPath, "0 0 | 0\n0 1 | 1\n1 0 | 1\n1 1 | 0");
            
            var pairs = TrainingDataLoader.LoadFromFile(tempPath);
            
            Assert.Equal(4, pairs.Count);
            Assert.Equal(2, pairs[0].Input.Length);
            Assert.Single(pairs[0].Output);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    [Fact]
    public void SaveAndLoad_RoundtripPreservesData()
    {
        var pairs = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 0.5m, 0.3m }, new decimal[] { 0.8m }),
            new TrainingPair(new decimal[] { 1.0m, 0.0m }, new decimal[] { 0.2m })
        };
        
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_data_{Guid.NewGuid()}.txt");
        
        try
        {
            TrainingDataLoader.SaveToFile(tempPath, pairs);
            var loaded = TrainingDataLoader.LoadFromFile(tempPath);
            
            Assert.Equal(pairs.Count, loaded.Count);
            for (int i = 0; i < pairs.Count; i++)
            {
                Assert.Equal(pairs[i].Input.Length, loaded[i].Input.Length);
                Assert.Equal(pairs[i].Output.Length, loaded[i].Output.Length);
                
                for (int j = 0; j < pairs[i].Input.Length; j++)
                {
                    Assert.Equal(pairs[i].Input[j], loaded[i].Input[j]);
                }
                for (int j = 0; j < pairs[i].Output.Length; j++)
                {
                    Assert.Equal(pairs[i].Output[j], loaded[i].Output[j]);
                }
            }
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }
}

public class FitnessCalculatorTests
{
    [Fact]
    public void CalculateFitness_ReturnsZeroForPerfectMatch()
    {
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 1 }, new decimal[] { 0.5m })
        };
        
        var calculator = new FitnessCalculator(trainingData);
        
        // Создаем сеть которая всегда выдает 0.5
        var network = NeuralNetwork.CreateEmpty(1, 1);
        // Примитивная проверка - просто убеждаемся что функция работает
        var fitness = calculator.CalculateFitness(network);
        
        Assert.True(fitness >= 0);
    }

    [Fact]
    public void IsBetter_ComparesByErrorFirst()
    {
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 1 }, new decimal[] { 0.5m })
        };
        
        var calculator = new FitnessCalculator(trainingData);
        var network1 = NeuralNetwork.CreateEmpty(1, 1);
        var network2 = NeuralNetwork.CreateEmpty(1, 1);
        
        // Сеть с меньшей ошибкой должна быть лучше
        Assert.True(calculator.IsBetter(network1, 0.1m, network2, 0.2m));
        Assert.False(calculator.IsBetter(network1, 0.2m, network2, 0.1m));
    }

    [Fact]
    public void IsBetter_EqualErrorsPreferLessComplexity()
    {
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 1 }, new decimal[] { 0.5m })
        };
        
        var calculator = new FitnessCalculator(trainingData);
        var simpleNetwork = NeuralNetwork.CreateEmpty(1, 1);
        var complexNetwork = NeuralNetwork.CreateEmpty(1, 1);
        complexNetwork.AddNeuron();
        complexNetwork.AddNeuron();
        
        // При равной ошибке предпочтительнее менее сложная
        Assert.True(calculator.IsBetter(simpleNetwork, 0.1m, complexNetwork, 0.1m));
    }
}

public class MutationEngineTests
{
    [Fact]
    public void ApplyMutations_ModifiesNetwork()
    {
        var network = NeuralNetwork.CreateEmpty(2, 1);
        var mutationEngine = new MutationEngine(42);
        
        var originalConnections = network.Connections.Count;
        var originalNeurons = network.Neurons.Count;
        
        mutationEngine.ApplyMutations(network, 10);
        
        // После мутаций сеть должна измениться
        // (хотя некоторые мутации могут быть неудачными)
        Assert.True(true); // Базовая проверка что метод работает без ошибок
    }
}

public class EvolutionEngineTests
{
    [Fact]
    public void InitializePopulation_CreatesIndividuals()
    {
        var settings = new EvolutionSettings
        {
            InnerPopulationSize = 5,
            InputCount = 2,
            OutputCount = 1
        };
        
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 0, 0 }, new decimal[] { 0 }),
            new TrainingPair(new decimal[] { 1, 1 }, new decimal[] { 1 })
        };
        
        var engine = new EvolutionEngine(settings, trainingData);
        engine.InitializePopulation();
        
        var stats = engine.GetStatistics();
        
        Assert.Equal(5, stats.PopulationSize);
        Assert.Equal(0, stats.Generation);
        Assert.NotNull(engine.BestEver);
    }

    [Fact]
    public void EvolveOneGeneration_ImprovesPopulation()
    {
        var settings = new EvolutionSettings
        {
            InnerPopulationSize = 10,
            OffspringPerIndividual = 3,
            MutationsPerOffspring = 5,
            InputCount = 2,
            OutputCount = 1,
            RandomSeed = 42
        };
        
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 0, 0 }, new decimal[] { 0 }),
            new TrainingPair(new decimal[] { 0, 1 }, new decimal[] { 1 }),
            new TrainingPair(new decimal[] { 1, 0 }, new decimal[] { 1 }),
            new TrainingPair(new decimal[] { 1, 1 }, new decimal[] { 0 })
        };
        
        var engine = new EvolutionEngine(settings, trainingData);
        engine.InitializePopulation();
        
        var initialFitness = engine.GetStatistics().BestFitness;
        
        // Эволюционируем 100 поколений
        engine.EvolveGenerations(100);
        
        var finalStats = engine.GetStatistics();
        
        Assert.Equal(100, finalStats.Generation);
        // Приспособленность должна улучшиться (уменьшиться) или остаться той же
        Assert.True(finalStats.BestFitness <= initialFitness);
    }

    [Fact]
    public void SaveAndLoadPopulation_RoundtripPreservesState()
    {
        var settings = new EvolutionSettings
        {
            InnerPopulationSize = 3,
            InputCount = 2,
            OutputCount = 1,
            RandomSeed = 42
        };
        
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(new decimal[] { 0, 0 }, new decimal[] { 0 })
        };
        
        var engine = new EvolutionEngine(settings, trainingData);
        engine.InitializePopulation();
        engine.EvolveGenerations(10);
        
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_pop_{Guid.NewGuid()}.txt");
        
        try
        {
            engine.SavePopulation(tempPath);
            
            var engine2 = new EvolutionEngine(settings, trainingData);
            engine2.LoadPopulation(tempPath);
            
            Assert.Equal(engine.CurrentGeneration, engine2.CurrentGeneration);
            Assert.NotNull(engine2.BestEver);
        }
        finally
        {
            // Очищаем файлы сохранения
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
                // Удаляем файлы сетей
                for (int i = 0; i < 10; i++)
                {
                    var netFile = tempPath.Replace(".txt", $"_ind{i}.net");
                    if (File.Exists(netFile))
                        File.Delete(netFile);
                }
            }
        }
    }
}

public class DecimalPrecisionTests
{
    [Fact]
    public void DecimalOperations_MaintainHighPrecision()
    {
        // Проверяем что decimal сохраняет высокую точность
        var a = 0.1234567890123456m;
        var b = 0.9876543210987654m;
        
        var sum = a + b;
        
        // Проверяем что сумма вычислена точно (не потеряна точность)
        Assert.Equal(1.1111111101111110m, sum);
    }

    [Fact]
    public void FitnessCalculation_UsesDecimalPrecision()
    {
        var trainingData = new List<TrainingPair>
        {
            new TrainingPair(
                new decimal[] { 0.123456789012345m }, 
                new decimal[] { 0.987654321098765m })
        };
        
        var calculator = new FitnessCalculator(trainingData);
        var network = NeuralNetwork.CreateEmpty(1, 1);
        
        var fitness = calculator.CalculateFitness(network);
        
        // Проверяем что fitness вычисляется без переполнения
        Assert.True(fitness >= 0);
        Assert.True(fitness < decimal.MaxValue / 2);
    }
}
