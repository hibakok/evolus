using System;
using System.Collections.Generic;
using System.Linq;

namespace evolus.Core
{
    /// <summary>
    /// Представляет функцию активации нейрона
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
        public double Bias { get; set; } = 0.0;

        public Neuron() { }

        public Neuron(int id, ActivationFunction activationFunction = ActivationFunction.Sigmoid)
        {
            Id = id;
            ActivationFunction = activationFunction;
        }

        public double Activate(double input)
        {
            double value = input + Bias;
            return ActivationFunction switch
            {
                ActivationFunction.Sigmoid => 1.0 / (1.0 + Math.Exp(-value)),
                ActivationFunction.Tanh => Math.Tanh(value),
                ActivationFunction.ReLU => Math.Max(0, value),
                ActivationFunction.Linear => value,
                ActivationFunction.Step => value >= 0 ? 1.0 : 0.0,
                _ => value
            };
        }
    }

    /// <summary>
    /// Связь между нейронами
    /// </summary>
    public class Connection
    {
        public int FromNeuronId { get; set; }
        public int ToNeuronId { get; set; }
        public decimal Weight { get; set; } = 0m;

        public Connection() { }

        public Connection(int from, int to, decimal weight = 0m)
        {
            FromNeuronId = from;
            ToNeuronId = to;
            Weight = weight;
        }
    }

    /// <summary>
    /// Нейросеть без разделения на слои - любой нейрон может быть связан с любым
    /// </summary>
    public class NeuralNetwork
    {
        public List<Neuron> Neurons { get; set; } = new List<Neuron>();
        public List<Connection> Connections { get; set; } = new List<Connection>();
        
        // Входы и выходы определяются как первые N и последние M нейронов
        public int InputCount { get; set; } = 0;
        public int OutputCount { get; set; } = 0;

        private int _nextNeuronId = 0;
        private int _complexity = 0;

        public NeuralNetwork() { }

        public NeuralNetwork(int inputCount, int outputCount)
        {
            InputCount = inputCount;
            OutputCount = outputCount;

            // Создаем входные нейроны
            for (int i = 0; i < inputCount; i++)
            {
                Neurons.Add(new Neuron(_nextNeuronId++, ActivationFunction.Linear));
            }

            // Создаем выходные нейроны
            for (int i = 0; i < outputCount; i++)
            {
                Neurons.Add(new Neuron(_nextNeuronId++));
            }
        }

        /// <summary>
        /// Вычислительная сложность = количество связей + количество нейронов
        /// </summary>
        public int Complexity => Connections.Count + Neurons.Count;

        /// <summary>
        /// Прямое распространение сигнала
        /// </summary>
        public decimal[] Forward(decimal[] inputs)
        {
            if (inputs.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} inputs, got {inputs.Length}");

            // Инициализируем значения нейронов
            var neuronValues = new Dictionary<int, decimal>();
            for (int i = 0; i < InputCount; i++)
            {
                neuronValues[Neurons[i].Id] = inputs[i];
            }

            // Топологическая сортировка для правильного порядка вычислений
            var sortedNeurons = TopologicalSort();
            
            // Вычисляем значения для остальных нейронов
            foreach (var neuronId in sortedNeurons)
            {
                if (neuronId < InputCount) continue; // Пропускаем входные нейроны

                var neuron = Neurons.First(n => n.Id == neuronId);
                decimal sum = 0m;

                foreach (var conn in Connections.Where(c => c.ToNeuronId == neuronId))
                {
                    if (neuronValues.ContainsKey(conn.FromNeuronId))
                    {
                        sum += neuronValues[conn.FromNeuronId] * (decimal)conn.Weight;
                    }
                }

                // Преобразуем в double для функции активации, затем обратно в decimal
                double inputValue = (double)sum;
                double activated = neuron.Activate(inputValue);
                neuronValues[neuronId] = (decimal)activated;
            }

            // Собираем выходные значения
            var outputs = new decimal[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                int outputNeuronId = Neurons[InputCount + i].Id;
                outputs[i] = neuronValues.ContainsKey(outputNeuronId) ? neuronValues[outputNeuronId] : 0m;
            }

            return outputs;
        }

        /// <summary>
        /// Топологическая сортировка нейронов для корректного порядка вычислений
        /// </summary>
        private List<int> TopologicalSort()
        {
            var visited = new HashSet<int>();
            var result = new List<int>();
            var tempMark = new HashSet<int>();

            void Visit(int nodeId)
            {
                if (tempMark.Contains(nodeId))
                    return; // Цикл, пропускаем
                
                if (visited.Contains(nodeId))
                    return;

                tempMark.Add(nodeId);

                foreach (var conn in Connections.Where(c => c.FromNeuronId == nodeId))
                {
                    Visit(conn.ToNeuronId);
                }

                tempMark.Remove(nodeId);
                visited.Add(nodeId);
                result.Add(nodeId);
            }

            foreach (var neuron in Neurons)
            {
                if (!visited.Contains(neuron.Id))
                {
                    Visit(neuron.Id);
                }
            }

            return result;
        }

        /// <summary>
        /// Добавляет случайную связь
        /// </summary>
        public void AddRandomConnection(Random random)
        {
            if (Neurons.Count < 2) return;

            int from = Neurons[random.Next(Neurons.Count)].Id;
            int to = Neurons[random.Next(Neurons.Count)].Id;

            if (from == to) return;
            if (Connections.Any(c => c.FromNeuronId == from && c.ToNeuronId == to)) return;

            Connections.Add(new Connection(from, to, (decimal)(random.NextDouble() * 2 - 1)));
        }

        /// <summary>
        /// Удаляет случайную связь
        /// </summary>
        public void RemoveRandomConnection(Random random)
        {
            if (Connections.Count == 0) return;

            int index = random.Next(Connections.Count);
            Connections.RemoveAt(index);
        }

        /// <summary>
        /// Изменяет вес случайной связи
        /// </summary>
        public void MutateWeight(Random random, decimal maxChange = 0.5m)
        {
            if (Connections.Count == 0) return;

            var conn = Connections[random.Next(Connections.Count)];
            decimal change = (decimal)(random.NextDouble() * 2 - 1) * maxChange;
            conn.Weight += change;
        }

        /// <summary>
        /// Добавляет новый нейрон
        /// </summary>
        public void AddNeuron(Random random)
        {
            var newNeuron = new Neuron(_nextNeuronId++, 
                (ActivationFunction)random.Next(Enum.GetNames(typeof(ActivationFunction)).Length));
            Neurons.Add(newNeuron);
        }

        /// <summary>
        /// Удаляет случайный нейрон (не входной и не выходной)
        /// </summary>
        public void RemoveNeuron(Random random)
        {
            var removableNeurons = Neurons
                .Where((n, i) => i >= InputCount && i < Neurons.Count - OutputCount)
                .ToList();

            if (removableNeurons.Count == 0) return;

            var neuronToRemove = removableNeurons[random.Next(removableNeurons.Count)];
            Neurons.Remove(neuronToRemove);
            Connections.RemoveAll(c => c.FromNeuronId == neuronToRemove.Id || c.ToNeuronId == neuronToRemove.Id);
        }

        /// <summary>
        /// Изменяет функцию активации случайного нейрона
        /// </summary>
        public void MutateActivationFunction(Random random)
        {
            var nonInputNeurons = Neurons.Skip(InputCount).ToList();
            if (nonInputNeurons.Count == 0) return;

            var neuron = nonInputNeurons[random.Next(nonInputNeurons.Count)];
            neuron.ActivationFunction = (ActivationFunction)random.Next(Enum.GetNames(typeof(ActivationFunction)).Length);
        }

        /// <summary>
        /// Применяет указанное количество минимальных мутаций
        /// </summary>
        public void ApplyMutations(Random random, int mutationCount)
        {
            var mutationActions = new List<Action>
            {
                () => MutateWeight(random),
                () => AddRandomConnection(random),
                () => RemoveRandomConnection(random),
                () => AddNeuron(random),
                () => RemoveNeuron(random),
                () => MutateActivationFunction(random)
            };

            for (int i = 0; i < mutationCount; i++)
            {
                var action = mutationActions[random.Next(mutationActions.Count)];
                action();
            }
        }

        /// <summary>
        /// Клонирует нейросеть
        /// </summary>
        public NeuralNetwork Clone()
        {
            var clone = new NeuralNetwork
            {
                InputCount = InputCount,
                OutputCount = OutputCount,
                _nextNeuronId = _nextNeuronId
            };

            foreach (var neuron in Neurons)
            {
                clone.Neurons.Add(new Neuron
                {
                    Id = neuron.Id,
                    ActivationFunction = neuron.ActivationFunction,
                    Bias = neuron.Bias
                });
            }

            foreach (var conn in Connections)
            {
                clone.Connections.Add(new Connection
                {
                    FromNeuronId = conn.FromNeuronId,
                    ToNeuronId = conn.ToNeuronId,
                    Weight = conn.Weight
                });
            }

            return clone;
        }

        /// <summary>
        /// Сохраняет нейросеть в текстовый файл
        /// </summary>
        public void SaveToFile(string filePath)
        {
            using (var writer = new System.IO.StreamWriter(filePath))
            {
                writer.WriteLine($"# Evolus Neural Network");
                writer.WriteLine($"# Inputs: {InputCount}, Outputs: {OutputCount}");
                writer.WriteLine($"# NextNeuronId: {_nextNeuronId}");
                writer.WriteLine();

                writer.WriteLine("# Neurons (Id, ActivationFunction, Bias)");
                foreach (var neuron in Neurons)
                {
                    writer.WriteLine($"N {neuron.Id} {neuron.ActivationFunction} {neuron.Bias}");
                }
                writer.WriteLine();

                writer.WriteLine("# Connections (From, To, Weight)");
                foreach (var conn in Connections)
                {
                    writer.WriteLine($"C {conn.FromNeuronId} {conn.ToNeuronId} {conn.Weight}");
                }
            }
        }

        /// <summary>
        /// Загружает нейросеть из текстового файла
        /// </summary>
        public static NeuralNetwork LoadFromFile(string filePath)
        {
            var network = new NeuralNetwork();
            
            using (var reader = new System.IO.StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;

                    var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2) continue;

                    if (parts[0] == "N" && parts.Length >= 3)
                    {
                        int id = int.Parse(parts[1]);
                        var activation = (ActivationFunction)Enum.Parse(typeof(ActivationFunction), parts[2]);
                        double bias = parts.Length > 3 ? double.Parse(parts[3]) : 0.0;

                        network.Neurons.Add(new Neuron(id, activation) { Bias = bias });
                        
                        if (id >= network._nextNeuronId)
                            network._nextNeuronId = id + 1;
                    }
                    else if (parts[0] == "C" && parts.Length >= 4)
                    {
                        int from = int.Parse(parts[1]);
                        int to = int.Parse(parts[2]);
                        decimal weight = decimal.Parse(parts[3]);

                        network.Connections.Add(new Connection(from, to, weight));
                    }
                    else if (line.StartsWith("# Inputs:"))
                    {
                        var match = System.Text.RegularExpressions.Regex.Match(line, @"Inputs:\s*(\d+),\s*Outputs:\s*(\d+)");
                        if (match.Success)
                        {
                            network.InputCount = int.Parse(match.Groups[1].Value);
                            network.OutputCount = int.Parse(match.Groups[2].Value);
                        }
                    }
                    else if (line.StartsWith("# NextNeuronId:"))
                    {
                        network._nextNeuronId = int.Parse(line.Split(':')[1].Trim());
                    }
                }
            }

            return network;
        }
    }
}
