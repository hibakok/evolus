using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace evolus.Core
{
    // Используем decimal для максимальной точности дробных чисел
    public struct DecimalVector
    {
        public decimal[] Values;
        public DecimalVector(int size) => Values = new decimal[size];
        public static DecimalVector Parse(string line)
        {
            var parts = line.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            var vec = new DecimalVector(parts.Length);
            for (int i = 0; i < parts.Length; i++)
                vec.Values[i] = decimal.Parse(parts[i], CultureInfo.InvariantCulture);
            return vec;
        }
        public override string ToString() => string.Join(" ", Values.Select(v => v.ToString(CultureInfo.InvariantCulture)));
    }

    public class DataPair
    {
        public DecimalVector Input { get; set; }
        public DecimalVector Output { get; set; }
        public DataPair(DecimalVector input, DecimalVector output)
        {
            Input = input;
            Output = output;
        }
    }

    public enum ActivationFunctionType
    {
        Sigmoid,
        Tanh,
        ReLU,
        Linear,
        Step
    }

    public class Neuron
    {
        public int Id { get; set; }
        public ActivationFunctionType ActivationType { get; set; }
        public decimal Bias { get; set; }

        public decimal Activate(decimal sum)
        {
            return ActivationType switch
            {
                ActivationFunctionType.Sigmoid => 1m / (1m + Exp(-sum)),
                ActivationFunctionType.Tanh => (Exp(sum) - Exp(-sum)) / (Exp(sum) + Exp(-sum)),
                ActivationFunctionType.ReLU => sum > 0 ? sum : 0m,
                ActivationFunctionType.Linear => sum,
                ActivationFunctionType.Step => sum >= 0 ? 1m : 0m,
                _ => sum
            };
        }

        private decimal Exp(decimal x)
        {
            // Приближение экспоненты для decimal
            if (x < -50m) return 0m;
            if (x > 50m) return 999999999999m;
            double dx = (double)x;
            return (decimal)Math.Exp(dx);
        }
    }

    public class Connection
    {
        public int FromNeuronId { get; set; }
        public int ToNeuronId { get; set; }
        public decimal Weight { get; set; }
    }

    public class NeuralNetwork
    {
        public List<Neuron> Neurons { get; set; } = new();
        public List<Connection> Connections { get; set; } = new();
        public int InputCount { get; set; }
        public int OutputCount { get; set; }

        public NeuralNetwork Clone()
        {
            var clone = new NeuralNetwork
            {
                InputCount = InputCount,
                OutputCount = OutputCount
            };
            foreach (var n in Neurons)
                clone.Neurons.Add(new Neuron { Id = n.Id, ActivationType = n.ActivationType, Bias = n.Bias });
            foreach (var c in Connections)
                clone.Connections.Add(new Connection { FromNeuronId = c.FromNeuronId, ToNeuronId = c.ToNeuronId, Weight = c.Weight });
            return clone;
        }

        public DecimalVector Forward(DecimalVector input)
        {
            if (input.Values.Length != InputCount)
                throw new ArgumentException($"Input size mismatch: expected {InputCount}, got {input.Values.Length}");

            var neuronOutputs = new Dictionary<int, decimal>();
            
            // Инициализируем входы
            for (int i = 0; i < InputCount; i++)
                neuronOutputs[i] = input.Values[i];

            // Топологическая сортировка или итеративный проход
            int maxIterations = Neurons.Count * 10;
            for (int iter = 0; iter < maxIterations; iter++)
            {
                bool changed = false;
                foreach (var neuron in Neurons.Where(n => n.Id >= InputCount))
                {
                    if (neuronOutputs.ContainsKey(neuron.Id)) continue;

                    decimal sum = neuron.Bias;
                    var incoming = Connections.Where(c => c.ToNeuronId == neuron.Id).ToList();
                    bool allInputsReady = true;
                    foreach (var conn in incoming)
                    {
                        if (!neuronOutputs.ContainsKey(conn.FromNeuronId))
                        {
                            allInputsReady = false;
                            break;
                        }
                        sum += neuronOutputs[conn.FromNeuronId] * conn.Weight;
                    }

                    if (allInputsReady)
                    {
                        neuronOutputs[neuron.Id] = neuron.Activate(sum);
                        changed = true;
                    }
                }

                if (Neurons.Where(n => n.Id >= InputCount).All(n => neuronOutputs.ContainsKey(n.Id)))
                    break;
                
                if (!changed && Neurons.Any(n => n.Id >= InputCount && !neuronOutputs.ContainsKey(n.Id)))
                {
                    // Цикл или недостижимые нейроны, присваиваем 0
                    foreach (var neuron in Neurons.Where(n => n.Id >= InputCount && !neuronOutputs.ContainsKey(n.Id)))
                        neuronOutputs[neuron.Id] = 0m;
                    break;
                }
            }

            var output = new DecimalVector(OutputCount);
            for (int i = 0; i < OutputCount; i++)
            {
                int outputNeuronId = InputCount + i;
                if (neuronOutputs.ContainsKey(outputNeuronId))
                    output.Values[i] = neuronOutputs[outputNeuronId];
                else
                    output.Values[i] = 0m;
            }

            return output;
        }

        public int GetComplexity()
        {
            return Neurons.Count + Connections.Count;
        }

        public void SaveToFile(string path)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine($"INPUTS:{InputCount}");
            writer.WriteLine($"OUTPUTS:{OutputCount}");
            writer.WriteLine($"NEURONS_COUNT:{Neurons.Count}");
            foreach (var n in Neurons)
                writer.WriteLine($"NEURON:{n.Id}:{n.ActivationType}:{n.Bias.ToString(CultureInfo.InvariantCulture)}");
            writer.WriteLine($"CONNECTIONS_COUNT:{Connections.Count}");
            foreach (var c in Connections)
                writer.WriteLine($"CONNECTION:{c.FromNeuronId}:{c.ToNeuronId}:{c.Weight.ToString(CultureInfo.InvariantCulture)}");
        }

        public static NeuralNetwork LoadFromFile(string path)
        {
            var nn = new NeuralNetwork();
            using var reader = new StreamReader(path);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Split(':');
                if (parts[0] == "INPUTS")
                    nn.InputCount = int.Parse(parts[1]);
                else if (parts[0] == "OUTPUTS")
                    nn.OutputCount = int.Parse(parts[1]);
                else if (parts[0] == "NEURON" && parts.Length >= 4)
                {
                    nn.Neurons.Add(new Neuron
                    {
                        Id = int.Parse(parts[1]),
                        ActivationType = (ActivationFunctionType)Enum.Parse(typeof(ActivationFunctionType), parts[2]),
                        Bias = decimal.Parse(parts[3], CultureInfo.InvariantCulture)
                    });
                }
                else if (parts[0] == "CONNECTION" && parts.Length >= 4)
                {
                    nn.Connections.Add(new Connection
                    {
                        FromNeuronId = int.Parse(parts[1]),
                        ToNeuronId = int.Parse(parts[2]),
                        Weight = decimal.Parse(parts[3], CultureInfo.InvariantCulture)
                    });
                }
            }
            return nn;
        }

        public static NeuralNetwork CreateEmpty(int inputCount, int outputCount)
        {
            var nn = new NeuralNetwork
            {
                InputCount = inputCount,
                OutputCount = outputCount
            };
            // Создаем только входные и выходные нейроны изначально
            for (int i = 0; i < inputCount + outputCount; i++)
            {
                nn.Neurons.Add(new Neuron
                {
                    Id = i,
                    ActivationType = i < inputCount ? ActivationFunctionType.Linear : ActivationFunctionType.Sigmoid,
                    Bias = 0m
                });
            }
            return nn;
        }
    }

    public class FitnessResult
    {
        public decimal Error { get; set; }
        public int Complexity { get; set; }

        public bool IsBetterThan(FitnessResult other)
        {
            if (Error < other.Error) return true;
            if (Error > other.Error) return false;
            return Complexity < other.Complexity;
        }
    }

    public class TrainingData
    {
        public List<DataPair> Pairs { get; set; } = new();

        public void LoadFromFile(string path)
        {
            Pairs.Clear();
            foreach (var line in File.ReadAllLines(path))
            {
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                var parts = line.Split('|');
                if (parts.Length != 2) continue;
                var input = DecimalVector.Parse(parts[0]);
                var output = DecimalVector.Parse(parts[1]);
                Pairs.Add(new DataPair(input, output));
            }
        }

        public void SaveToFile(string path)
        {
            using var writer = new StreamWriter(path);
            foreach (var p in Pairs)
                writer.WriteLine($"{p.Input} | {p.Output}");
        }
    }

    public class Config
    {
        public int PopulationSize { get; set; } = 20;
        public int OffspringPerIndividual { get; set; } = 3;
        public int MutationsPerOffspring { get; set; } = 5;
        public string DataFilePath { get; set; } = "data/training.txt";
        public string SaveFilePath { get; set; } = "save/population.txt";

        public void LoadFromFile(string path)
        {
            if (!File.Exists(path)) return;
            foreach (var line in File.ReadAllLines(path))
            {
                var parts = line.Split('=');
                if (parts.Length != 2) continue;
                var key = parts[0].Trim();
                var value = parts[1].Trim();
                switch (key)
                {
                    case "PopulationSize": PopulationSize = int.Parse(value); break;
                    case "OffspringPerIndividual": OffspringPerIndividual = int.Parse(value); break;
                    case "MutationsPerOffspring": MutationsPerOffspring = int.Parse(value); break;
                    case "DataFilePath": DataFilePath = value; break;
                    case "SaveFilePath": SaveFilePath = value; break;
                }
            }
        }

        public void SaveToFile(string path)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine($"PopulationSize={PopulationSize}");
            writer.WriteLine($"OffspringPerIndividual={OffspringPerIndividual}");
            writer.WriteLine($"MutationsPerOffspring={MutationsPerOffspring}");
            writer.WriteLine($"DataFilePath={DataFilePath}");
            writer.WriteLine($"SaveFilePath={SaveFilePath}");
        }
    }
}
