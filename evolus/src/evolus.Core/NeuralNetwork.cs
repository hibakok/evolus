using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace evolus.Core
{
    /// <summary>
    /// Представляет связь между нейронами
    /// </summary>
    public class Connection
{
    public int FromNeuronId { get; set; }
    public int ToNeuronId { get; set; }
    public decimal Weight { get; set; }

    public Connection(int from, int to, decimal weight)
    {
        FromNeuronId = from;
        ToNeuronId = to;
        Weight = weight;
    }
}

/// <summary>
/// Функции активации нейронов
/// </summary>
public enum ActivationFunction
{
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
    Step
}

/// <summary>
/// Нейрон в сети
/// </summary>
public class Neuron
{
    public int Id { get; set; }
    public ActivationFunction ActivationFunction { get; set; } = ActivationFunction.Sigmoid;
    public decimal Output { get; set; }

    public Neuron(int id)
    {
        Id = id;
    }
}

/// <summary>
/// Нейросеть без слоев с произвольными связями
/// </summary>
public class NeuralNetwork
{
    public List<Neuron> Neurons { get; } = new();
    public List<Connection> Connections { get; } = new();
    
    private int _nextNeuronId = 0;
    private int _inputCount = 0;
    private int _outputCount = 0;

    public int InputCount 
    { 
        get => _inputCount; 
        set => _inputCount = value;
    }
    
    public int OutputCount 
    { 
        get => _outputCount; 
        set => _outputCount = value;
    }

    /// <summary>
    /// Вычислительная сложность (количество связей + количество нейронов)
    /// </summary>
    public int Complexity => Connections.Count + Neurons.Count;

    /// <summary>
    /// Создать новую пустую сеть
    /// </summary>
    public static NeuralNetwork CreateEmpty(int inputCount, int outputCount)
    {
        var network = new NeuralNetwork
        {
            _inputCount = inputCount,
            _outputCount = outputCount,
            _nextNeuronId = 0
        };

        // Создаем входные нейроны
        for (int i = 0; i < inputCount; i++)
        {
            network.Neurons.Add(new Neuron(network._nextNeuronId++)
            {
                ActivationFunction = ActivationFunction.Linear
            });
        }

        // Создаем выходные нейроны - их ID должны идти после входных
        int outputStartId = inputCount;
        for (int i = 0; i < outputCount; i++)
        {
            network.Neurons.Add(new Neuron(outputStartId + i)
            {
                ActivationFunction = ActivationFunction.Sigmoid
            });
        }
        
        network._nextNeuronId = outputStartId + outputCount;

        return network;
    }

    /// <summary>
    /// Прямое распространение сигнала
    /// </summary>
    public decimal[] Forward(decimal[] inputs)
    {
        if (inputs.Length != _inputCount)
            throw new ArgumentException($"Ожидается {_inputCount} входных значений, получено {inputs.Length}");

        // Устанавливаем значения входных нейронов
        for (int i = 0; i < _inputCount; i++)
        {
            Neurons[i].Output = inputs[i];
        }

        // Сортируем нейроны для правильного порядка вычислений (топологический порядок упрощенный)
        // Для полносвязной сети без слоев выполняем несколько итераций распространения
        int iterations = Math.Max(5, Neurons.Count);
        
        for (int iter = 0; iter < iterations; iter++)
        {
            foreach (var neuron in Neurons.Skip(_inputCount)) // Пропускаем входные
            {
                decimal sum = 0;
                
                foreach (var conn in Connections.Where(c => c.ToNeuronId == neuron.Id))
                {
                    var fromNeuron = Neurons.FirstOrDefault(n => n.Id == conn.FromNeuronId);
                    if (fromNeuron != null)
                    {
                        sum += fromNeuron.Output * conn.Weight;
                    }
                }

                neuron.Output = ApplyActivation(sum, neuron.ActivationFunction);
            }
        }

        // Возвращаем значения выходных нейронов
        var outputs = new decimal[_outputCount];
        for (int i = 0; i < _outputCount; i++)
        {
            var outputNeuron = Neurons.FirstOrDefault(n => n.Id == _inputCount + i);
            if (outputNeuron != null)
            {
                outputs[i] = outputNeuron.Output;
            }
        }

        return outputs;
    }

    private decimal ApplyActivation(decimal x, ActivationFunction func)
    {
        return func switch
        {
            ActivationFunction.Sigmoid => 1m / (1m + Exp(-x)),
            ActivationFunction.Tanh => Tanh(x),
            ActivationFunction.ReLU => x > 0 ? x : 0m,
            ActivationFunction.Linear => x,
            ActivationFunction.Step => x >= 0 ? 1m : 0m,
            _ => x
        };
    }

    // Приближенное вычисление exp для decimal
    private decimal Exp(decimal x)
    {
        // Используем разложение в ряд Тейлора для точности
        if (x < -700m) return 0m;
        if (x > 700m) return decimal.MaxValue;
        
        decimal result = 1m;
        decimal term = 1m;
        
        for (int i = 1; i <= 50; i++)
        {
            term *= x / i;
            result += term;
            
            if (Math.Abs((double)term) < 1e-28) break;
        }
        
        return result;
    }

    // Приближенное вычисление tanh для decimal
    private decimal Tanh(decimal x)
    {
        if (x < -20m) return -1m;
        if (x > 20m) return 1m;
        
        var expX = Exp(x);
        var expNegX = Exp(-x);
        
        return (expX - expNegX) / (expX + expNegX);
    }

    /// <summary>
    /// Добавить нейрон
    /// </summary>
    public void AddNeuron(ActivationFunction func = ActivationFunction.Sigmoid)
    {
        Neurons.Add(new Neuron(_nextNeuronId++) { ActivationFunction = func });
    }

    /// <summary>
    /// Удалить нейрон по ID
    /// </summary>
    public bool RemoveNeuron(int id)
    {
        var neuron = Neurons.FirstOrDefault(n => n.Id == id);
        if (neuron == null) return false;

        // Удаляем все связи с этим нейроном
        Connections.RemoveAll(c => c.FromNeuronId == id || c.ToNeuronId == id);
        Neurons.Remove(neuron);
        
        return true;
    }

    /// <summary>
    /// Добавить связь
    /// </summary>
    public void AddConnection(int from, int to, decimal weight)
    {
        if (!Neurons.Any(n => n.Id == from) || !Neurons.Any(n => n.Id == to))
            return;
        
        if (from == to) return; // Без петель
        
        // Проверяем, нет ли уже такой связи
        if (Connections.Any(c => c.FromNeuronId == from && c.ToNeuronId == to))
            return;

        Connections.Add(new Connection(from, to, weight));
    }

    /// <summary>
    /// Удалить связь
    /// </summary>
    public bool RemoveConnection(int from, int to)
    {
        var conn = Connections.FirstOrDefault(c => c.FromNeuronId == from && c.ToNeuronId == to);
        if (conn == null) return false;
        
        Connections.Remove(conn);
        return true;
    }

    /// <summary>
    /// Изменить вес связи
    /// </summary>
    public bool ChangeConnectionWeight(int from, int to, decimal newWeight)
    {
        var conn = Connections.FirstOrDefault(c => c.FromNeuronId == from && c.ToNeuronId == to);
        if (conn == null) return false;
        
        conn.Weight = newWeight;
        return true;
    }

    /// <summary>
    /// Изменить функцию активации нейрона
    /// </summary>
    public bool ChangeActivationFunction(int neuronId, ActivationFunction newFunc)
    {
        var neuron = Neurons.FirstOrDefault(n => n.Id == neuronId);
        if (neuron == null) return false;
        
        // Входные нейроны всегда линейные
        if (neuron.Id < _inputCount) return false;
        
        neuron.ActivationFunction = newFunc;
        return true;
    }

    /// <summary>
    /// Сохранить сеть в файл
    /// </summary>
    public void SaveToFile(string path)
    {
        var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
        try
        {
            // Заголовок: входы|выходы
            writer.WriteLine($"{_inputCount}|{_outputCount}");
            
            // Нейроны: ID|Функция
            foreach (var neuron in Neurons.OrderBy(n => n.Id))
            {
                writer.WriteLine($"N|{neuron.Id}|{(int)neuron.ActivationFunction}");
            }
            
            // Связи: От|К|Вес
            foreach (var conn in Connections)
            {
                writer.WriteLine($"C|{conn.FromNeuronId}|{conn.ToNeuronId}|{conn.Weight.ToString(CultureInfo.InvariantCulture)}");
            }
        }
        finally
        {
            if (writer != null) writer.Dispose();
        }
    }

    /// <summary>
    /// Загрузить сеть из файла
    /// </summary>
    public static NeuralNetwork LoadFromFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Файл не найден: {path}");

        var lines = File.ReadAllLines(path);
        if (lines.Length == 0)
            throw new Exception("Пустой файл сети");

        var header = lines[0].Split('|');
        var inputCount = int.Parse(header[0]);
        var outputCount = int.Parse(header[1]);

        var network = new NeuralNetwork
        {
            _inputCount = inputCount,
            _outputCount = outputCount,
            _nextNeuronId = 0
        };

        foreach (var line in lines.Skip(1))
        {
            var parts = line.Split('|');
            if (parts[0] == "N")
            {
                var id = int.Parse(parts[1]);
                var func = (ActivationFunction)int.Parse(parts[2]);
                network.Neurons.Add(new Neuron(id) { ActivationFunction = func });
                if (id >= network._nextNeuronId)
                    network._nextNeuronId = id + 1;
            }
            else if (parts[0] == "C")
            {
                var from = int.Parse(parts[1]);
                var to = int.Parse(parts[2]);
                var weight = decimal.Parse(parts[3], CultureInfo.InvariantCulture);
                network.Connections.Add(new Connection(from, to, weight));
            }
        }

        return network;
    }

    /// <summary>
    /// Клонировать сеть
    /// </summary>
    public NeuralNetwork Clone()
    {
        var clone = new NeuralNetwork
        {
            _inputCount = _inputCount,
            _outputCount = _outputCount,
            _nextNeuronId = _nextNeuronId
        };

        foreach (var neuron in Neurons.OrderBy(n => n.Id))
        {
            clone.Neurons.Add(new Neuron(neuron.Id) { ActivationFunction = neuron.ActivationFunction });
        }

        foreach (var conn in Connections)
        {
            clone.Connections.Add(new Connection(conn.FromNeuronId, conn.ToNeuronId, conn.Weight));
        }

        return clone;
    }
}
}
