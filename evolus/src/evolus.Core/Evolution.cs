using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace evolus.Core
{
    /// <summary>
    /// Пара входных-выходных данных для обучения
    /// </summary>
    public class TrainingPair
{
    public decimal[] Input { get; }
    public decimal[] Output { get; }

    public TrainingPair(decimal[] input, decimal[] output)
    {
        Input = input;
        Output = output;
    }
}

/// <summary>
/// Загрузчик обучающих данных из файла
/// </summary>
public static class TrainingDataLoader
{
    /// <summary>
    /// Загрузить пары данных из файла
    /// Формат: "0 1 | 1 0" (входные | выходные)
    /// </summary>
    public static List<TrainingPair> LoadFromFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Файл обучающих данных не найден: {path}");

        var pairs = new List<TrainingPair>();
        var lines = File.ReadAllLines(path);

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (string.IsNullOrEmpty(trimmedLine) || trimmedLine.StartsWith("#"))
                continue; // Пропускаем пустые строки и комментарии

            var parts = trimmedLine.Split('|');
            if (parts.Length != 2)
                throw new Exception($"Неверный формат строки: {line}. Ожидается 'входные | выходные'");

            var inputValues = parts[0].Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(x => decimal.Parse(x, CultureInfo.InvariantCulture)).ToArray();
            
            var outputValues = parts[1].Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(x => decimal.Parse(x, CultureInfo.InvariantCulture)).ToArray();

            pairs.Add(new TrainingPair(inputValues, outputValues));
        }

        return pairs;
    }

    /// <summary>
    /// Сохранить пары данных в файл
    /// </summary>
    public static void SaveToFile(string path, List<TrainingPair> pairs)
    {
        var writer = new StreamWriter(path);
        try
        {
            writer.WriteLine("# Формат: входные_значения | выходные_значения");
            writer.WriteLine("# Пример: 0 1 | 1 0");
            writer.WriteLine();

            foreach (var pair in pairs)
            {
                var inputStr = string.Join(" ", pair.Input.Select(x => x.ToString(CultureInfo.InvariantCulture)));
                var outputStr = string.Join(" ", pair.Output.Select(x => x.ToString(CultureInfo.InvariantCulture)));
                writer.WriteLine($"{inputStr} | {outputStr}");
            }
        }
        finally
        {
            if (writer != null) writer.Dispose();
        }
    }
}

/// <summary>
/// Особь в эволюционном алгоритме
/// </summary>
public class Individual
{
    public NeuralNetwork Network { get; }
    public decimal Fitness { get; set; } = decimal.MaxValue; // Меньше = лучше (ошибка)
    public int? ParentId { get; set; } // ID родителя во внутренней популяции
    public int Id { get; }

    private static int _nextId = 0;

    public Individual(NeuralNetwork network, int? parentId = null)
    {
        Network = network;
        ParentId = parentId;
        Id = _nextId++;
    }

    public Individual Clone()
    {
        return new Individual(Network.Clone(), ParentId)
        {
            Fitness = Fitness
        };
    }
}

/// <summary>
/// Менеджер мутаций
/// </summary>
public class MutationEngine
{
    private readonly Random _random;
    
    public MutationEngine(int seed)
    {
        _random = new Random(seed);
    }

    /// <summary>
    /// Применить N минимальных мутаций к сети
    /// </summary>
    public void ApplyMutations(NeuralNetwork network, int mutationCount)
    {
        for (int i = 0; i < mutationCount; i++)
        {
            var mutationType = _random.Next(6); // 6 типов мутаций
            
            switch (mutationType)
            {
                case 0: MutateWeight(network); break;
                case 1: AddConnection(network); break;
                case 2: RemoveConnection(network); break;
                case 3: AddNeuron(network); break;
                case 4: RemoveNeuron(network); break;
                case 5: ChangeActivationFunction(network); break;
            }
        }
    }

    private void MutateWeight(NeuralNetwork network)
    {
        if (network.Connections.Count == 0) return;

        var conn = network.Connections[_random.Next(network.Connections.Count)];
        var delta = (decimal)(_random.NextDouble() * 2 - 1) * 0.5m; // Изменение от -0.5 до 0.5
        conn.Weight += delta;
    }

    private void AddConnection(NeuralNetwork network)
    {
        if (network.Neurons.Count < 2) return;

        var from = network.Neurons[_random.Next(network.Neurons.Count)].Id;
        var to = network.Neurons[_random.Next(network.Neurons.Count)].Id;
        
        if (from == to) return;
        
        var weight = (decimal)(_random.NextDouble() * 2 - 1); // Вес от -1 до 1
        network.AddConnection(from, to, weight);
    }

    private void RemoveConnection(NeuralNetwork network)
    {
        if (network.Connections.Count == 0) return;

        var conn = network.Connections[_random.Next(network.Connections.Count)];
        network.RemoveConnection(conn.FromNeuronId, conn.ToNeuronId);
    }

    private void AddNeuron(NeuralNetwork network)
    {
        var functions = Enum.GetValues(typeof(ActivationFunction)).Cast<ActivationFunction>().ToList();
        var func = functions[_random.Next(functions.Count)];
        network.AddNeuron(func);
    }

    private void RemoveNeuron(NeuralNetwork network)
    {
        // Не удаляем входные и выходные нейроны
        var removableNeurons = network.Neurons
            .Where(n => n.Id >= network.InputCount + network.OutputCount)
            .ToList();

        if (removableNeurons.Count == 0) return;

        var neuron = removableNeurons[_random.Next(removableNeurons.Count)];
        network.RemoveNeuron(neuron.Id);
    }

    private void ChangeActivationFunction(NeuralNetwork network)
    {
        // Меняем функцию только у скрытых нейронов
        var hiddenNeurons = network.Neurons
            .Where(n => n.Id >= network.InputCount)
            .ToList();

        if (hiddenNeurons.Count == 0) return;

        var neuron = hiddenNeurons[_random.Next(hiddenNeurons.Count)];
        var functions = Enum.GetValues(typeof(ActivationFunction)).Cast<ActivationFunction>().ToList();
        var newFunc = functions[_random.Next(functions.Count)];
        
        network.ChangeActivationFunction(neuron.Id, newFunc);
    }
}

/// <summary>
/// Вычисление приспособленности
/// </summary>
public class FitnessCalculator
{
    private readonly List<TrainingPair> _trainingData;

    public FitnessCalculator(List<TrainingPair> trainingData)
    {
        _trainingData = trainingData;
    }

    /// <summary>
    /// Вычислить приспособленность особи
    /// Возвращает ошибку (меньше = лучше)
    /// </summary>
    public decimal CalculateFitness(NeuralNetwork network)
    {
        decimal totalError = 0;

        foreach (var pair in _trainingData)
        {
            try
            {
                var outputs = network.Forward(pair.Input);
                
                // Считаем среднеквадратичную ошибку с максимальной точностью
                for (int i = 0; i < Math.Min(outputs.Length, pair.Output.Length); i++)
                {
                    var diff = outputs[i] - pair.Output[i];
                    totalError += diff * diff;
                }

                // Если сеть выдала меньше выходов чем ожидалось
                if (outputs.Length < pair.Output.Length)
                {
                    for (int i = outputs.Length; i < pair.Output.Length; i++)
                    {
                        totalError += pair.Output[i] * pair.Output[i];
                    }
                }
            }
            catch
            {
                // При ошибке вычисления считаем ошибку максимальной
                return decimal.MaxValue / 2;
            }
        }

        return totalError / _trainingData.Count;
    }

    /// <summary>
    /// Сравнить две особи по приспособленности
    /// Возвращает true если first лучше second
    /// Правила:
    /// 1. Если сложность одинакова, лучше та у которой ошибка меньше
    /// 2. Если сложность разная, но ошибка хотя бы чуть-чуть меньше у более сложной - она лучше
    /// 3. Сложность не добавляет штрафа, а анализируется напрямую
    /// </summary>
    public bool IsBetter(NeuralNetwork first, decimal firstError, NeuralNetwork second, decimal secondError)
    {
        // Если ошибки равны с максимальной точностью
        if (firstError == secondError)
        {
            // При равной ошибке предпочтительнее менее сложная
            return first.Complexity < second.Complexity;
        }

        // Если ошибка первой хоть немного меньше - она лучше независимо от сложности
        if (firstError < secondError)
        {
            return true;
        }

        return false;
    }
}
}