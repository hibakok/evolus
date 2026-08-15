using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace evolus.Core
{
    /// <summary>
    /// Настройки эволюционного алгоритма
    /// </summary>
    public class EvolutionSettings
{
    public int InnerPopulationSize { get; set; } = 10; // Размер внутренней популяции
    public int OffspringPerIndividual { get; set; } = 3; // Потомков от каждой особи
    public int MutationsPerOffspring { get; set; } = 5; // Мутаций на потомка
    public int InputCount { get; set; } = 2; // Количество входов по умолчанию
    public int OutputCount { get; set; } = 2; // Количество выходов по умолчанию
    public int RandomSeed { get; set; } = 42; // Seed для воспроизводимости
}

/// <summary>
/// Загрузчик настроек из файла
/// </summary>
public static class SettingsLoader
{
    public static EvolutionSettings LoadFromFile(string path)
    {
        var settings = new EvolutionSettings();

        if (!File.Exists(path))
        {
            SaveToFile(path, settings);
            return settings;
        }

        var lines = File.ReadAllLines(path);
        
        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (string.IsNullOrEmpty(trimmedLine) || trimmedLine.StartsWith("#"))
                continue;

            var parts = trimmedLine.Split('=');
            if (parts.Length != 2) continue;

            var key = parts[0].Trim().ToLower();
            var value = parts[1].Trim();

            switch (key)
            {
                case "innerpopulationsize":
                    settings.InnerPopulationSize = int.Parse(value);
                    break;
                case "offspringperindividual":
                    settings.OffspringPerIndividual = int.Parse(value);
                    break;
                case "mutationsperoffspring":
                    settings.MutationsPerOffspring = int.Parse(value);
                    break;
                case "inputcount":
                    settings.InputCount = int.Parse(value);
                    break;
                case "outputcount":
                    settings.OutputCount = int.Parse(value);
                    break;
                case "randomseed":
                    settings.RandomSeed = int.Parse(value);
                    break;
            }
        }

        return settings;
    }

    public static void SaveToFile(string path, EvolutionSettings settings)
    {
        var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
        try
        {
            writer.WriteLine("# Настройки эволюционного алгоритма evolus");
            writer.WriteLine("# Формат: ключ=значение");
            writer.WriteLine();
            writer.WriteLine($"InnerPopulationSize={settings.InnerPopulationSize}");
            writer.WriteLine($"OffspringPerIndividual={settings.OffspringPerIndividual}");
            writer.WriteLine($"MutationsPerOffspring={settings.MutationsPerOffspring}");
            writer.WriteLine($"InputCount={settings.InputCount}");
            writer.WriteLine($"OutputCount={settings.OutputCount}");
            writer.WriteLine($"RandomSeed={settings.RandomSeed}");
        }
        finally
        {
            if (writer != null) writer.Dispose();
        }
    }
}

/// <summary>
/// Эволюционный движок с внутренней и внешней популяциями
/// </summary>
public class EvolutionEngine
{
    private readonly EvolutionSettings _settings;
    private readonly MutationEngine _mutationEngine;
    private readonly FitnessCalculator _fitnessCalculator;
    
    // Внутренняя популяция (неприкосновенная)
    private List<Individual> _innerPopulation = new List<Individual>();
    
    // Лучшая особь за все время
    public Individual BestEver { get; private set; }
    
    public int CurrentGeneration { get; private set; } = 0;

    public EvolutionEngine(
        EvolutionSettings settings,
        List<TrainingPair> trainingData)
    {
        _settings = settings;
        _mutationEngine = new MutationEngine(settings.RandomSeed);
        _fitnessCalculator = new FitnessCalculator(trainingData);
    }

    /// <summary>
    /// Инициализировать внутреннюю популяцию с нуля
    /// </summary>
    public void InitializePopulation()
    {
        _innerPopulation.Clear();
        CurrentGeneration = 0;
        BestEver = null;

        for (int i = 0; i < _settings.InnerPopulationSize; i++)
        {
            var network = NeuralNetwork.CreateEmpty(_settings.InputCount, _settings.OutputCount);
            _innerPopulation.Add(new Individual(network));
        }

        // Первичная оценка приспособленности
        EvaluatePopulation();
    }

    /// <summary>
    /// Загрузить состояние внутренней популяции из файла сохранения
    /// </summary>
    public void LoadPopulation(string savePath)
    {
        if (!File.Exists(savePath))
        {
            InitializePopulation();
            return;
        }

        _innerPopulation.Clear();
        CurrentGeneration = 0;
        BestEver = null;

        var lines = File.ReadAllLines(savePath);
        var generationLine = lines.FirstOrDefault(l => l.StartsWith("GENERATION="));
        if (generationLine != null)
        {
            CurrentGeneration = int.Parse(generationLine.Split('=')[1]);
        }

        var bestLine = lines.FirstOrDefault(l => l.StartsWith("BEST_FITNESS="));
        if (bestLine != null)
        {
            var fitness = decimal.Parse(bestLine.Split('=')[1], CultureInfo.InvariantCulture);
            // BestEver будет установлен после загрузки первой особи с такой fitness
        }

        Individual currentBest = null;
        decimal bestFitness = decimal.MaxValue;

        foreach (var line in lines)
        {
            if (line.StartsWith("INDIVIDUAL="))
            {
                var parts = line.Split('=');
                if (parts.Length >= 3)
                {
                    var parentIdStr = parts[1];
                    var fitnessStr = parts[2];
                    
                    int? parentId = parentIdStr == "null" ? null : int.Parse(parentIdStr);
                    var fitness = decimal.Parse(fitnessStr, CultureInfo.InvariantCulture);

                    // Ищем файл сети
                    var individualIndex = _innerPopulation.Count;
                    var networkPath = savePath.Replace(".txt", $"_ind{individualIndex}.net");
                    
                    if (File.Exists(networkPath))
                    {
                        var network = NeuralNetwork.LoadFromFile(networkPath);
                        var individual = new Individual(network, parentId) { Fitness = fitness };
                        _innerPopulation.Add(individual);

                        if (fitness < bestFitness)
                        {
                            bestFitness = fitness;
                            currentBest = individual;
                        }
                    }
                }
            }
        }

        BestEver = currentBest;
    }

    /// <summary>
    /// Сохранить состояние внутренней популяции в файл
    /// </summary>
    public void SavePopulation(string savePath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(savePath) ?? ".");

        var writer = new StreamWriter(savePath, false, System.Text.Encoding.UTF8);
        try
        {
            writer.WriteLine($"GENERATION={CurrentGeneration}");
            
            if (BestEver != null)
            {
                writer.WriteLine($"BEST_FITNESS={BestEver.Fitness.ToString(CultureInfo.InvariantCulture)}");
            }

            for (int i = 0; i < _innerPopulation.Count; i++)
            {
                var ind = _innerPopulation[i];
                var networkPath = savePath.Replace(".txt", $"_ind{i}.net");
                ind.Network.SaveToFile(networkPath);
                
                writer.WriteLine($"INDIVIDUAL={ind.ParentId?.ToString() ?? "null"}={ind.Fitness.ToString(CultureInfo.InvariantCulture)}");
            }
        }
        finally
        {
            if (writer != null) writer.Dispose();
        }
    }

    /// <summary>
    /// Выполнить одно поколение эволюции
    /// </summary>
    public void EvolveOneGeneration()
    {
        // Создаем внешнюю популяцию - потомков от внутренней
        var offspringPopulation = new List<Individual>();

        foreach (var parent in _innerPopulation)
        {
            for (int i = 0; i < _settings.OffspringPerIndividual; i++)
            {
                var childNetwork = parent.Network.Clone();
                _mutationEngine.ApplyMutations(childNetwork, _settings.MutationsPerOffspring);
                
                var child = new Individual(childNetwork, parent.Id);
                offspringPopulation.Add(child);
            }
        }

        // Оцениваем приспособленность потомков
        foreach (var offspring in offspringPopulation)
        {
            offspring.Fitness = _fitnessCalculator.CalculateFitness(offspring.Network);
        }

        // Обновляем лучшую особь за все время
        foreach (var offspring in offspringPopulation)
        {
            if (BestEver == null || offspring.Fitness < BestEver.Fitness)
            {
                BestEver = offspring.Clone();
            }
        }

        // Замена родителей потомками если потомки лучше
        foreach (var offspring in offspringPopulation)
        {
            if (offspring.ParentId.HasValue)
            {
                var parent = _innerPopulation.FirstOrDefault(p => p.Id == offspring.ParentId.Value);
                if (parent != null)
                {
                    if (_fitnessCalculator.IsBetter(offspring.Network, offspring.Fitness, 
                                                    parent.Network, parent.Fitness))
                    {
                        // Заменяем родителя потомком
                        var index = _innerPopulation.IndexOf(parent);
                        _innerPopulation[index] = new Individual(offspring.Network.Clone(), offspring.ParentId)
                        {
                            Fitness = offspring.Fitness
                        };
                    }
                }
            }
        }

        // Переоцениваем внутреннюю популяцию (на случай изменений)
        EvaluatePopulation();

        CurrentGeneration++;
    }

    /// <summary>
    /// Выполнить N поколений эволюции
    /// </summary>
    public void EvolveGenerations(int generationsCount)
    {
        for (int i = 0; i < generationsCount; i++)
        {
            EvolveOneGeneration();
        }
    }

    /// <summary>
    /// Оценить всю внутреннюю популяцию
    /// </summary>
    private void EvaluatePopulation()
    {
        foreach (var individual in _innerPopulation)
        {
            individual.Fitness = _fitnessCalculator.CalculateFitness(individual.Network);
            
            if (BestEver == null || individual.Fitness < BestEver.Fitness)
            {
                BestEver = individual.Clone();
            }
        }
    }

    /// <summary>
    /// Получить лучшую особь из внутренней популяции
    /// </summary>
    public Individual GetBestIndividual()
    {
        return _innerPopulation.OrderBy(i => i.Fitness).First();
    }

    /// <summary>
    /// Получить статистику популяции
    /// </summary>
    public PopulationStatistics GetStatistics()
    {
        var fitnesses = _innerPopulation.Select(i => i.Fitness).ToList();
        
        return new PopulationStatistics
        {
            Generation = CurrentGeneration,
            PopulationSize = _innerPopulation.Count,
            BestFitness = fitnesses.Min(),
            WorstFitness = fitnesses.Max(),
            AverageFitness = (double)fitnesses.Average(),
            BestComplexity = _innerPopulation.OrderBy(i => i.Fitness).First().Network.Complexity
        };
    }
}

/// <summary>
/// Статистика популяции
/// </summary>
public class PopulationStatistics
{
    public int Generation { get; set; }
    public int PopulationSize { get; set; }
    public decimal BestFitness { get; set; }
    public decimal WorstFitness { get; set; }
    public double AverageFitness { get; set; }
    public int BestComplexity { get; set; }
}
}
