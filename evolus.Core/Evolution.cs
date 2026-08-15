using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace evolus.Core
{
    public class Individual
    {
        public NeuralNetwork Network { get; set; }
        public FitnessResult? Fitness { get; set; }
        public int ParentId { get; set; } = -1;
        public int Id { get; set; }

        public Individual(NeuralNetwork network, int id)
        {
            Network = network;
            Id = id;
        }
    }

    public class EvolutionEngine
    {
        private readonly Config _config;
        private readonly TrainingData _data;
        private List<Individual> _innerPopulation = new();
        private int _nextId = 0;
        private readonly Random _random = new();

        public EvolutionEngine(Config config, TrainingData data)
        {
            _config = config;
            _data = data;
        }

        public void Initialize()
        {
            _innerPopulation.Clear();
            _nextId = 0;

            // Определяем размерность входа и выхода из данных
            if (_data.Pairs.Count == 0)
                throw new InvalidOperationException("No training data loaded");

            int inputCount = _data.Pairs[0].Input.Values.Length;
            int outputCount = _data.Pairs[0].Output.Values.Length;

            for (int i = 0; i < _config.PopulationSize; i++)
            {
                var nn = NeuralNetwork.CreateEmpty(inputCount, outputCount);
                _innerPopulation.Add(new Individual(nn, _nextId++));
            }
        }

        public void LoadPopulation(string path)
        {
            if (!File.Exists(path)) return;

            _innerPopulation.Clear();
            var lines = File.ReadAllLines(path);
            int idx = 0;
            
            while (idx < lines.Length)
            {
                var line = lines[idx];
                if (line.StartsWith("INDIVIDUAL:"))
                {
                    var parts = line.Split(':');
                    int id = int.Parse(parts[1]);
                    int parentId = int.Parse(parts[2]);
                    
                    // Считываем сеть до следующего INDIVIDUAL или конца
                    var networkLines = new List<string>();
                    idx++;
                    while (idx < lines.Length && !lines[idx].StartsWith("INDIVIDUAL:"))
                    {
                        networkLines.Add(lines[idx]);
                        idx++;
                    }

                    // Временный файл для сети
                    var tempPath = Path.GetTempFileName();
                    File.WriteAllLines(tempPath, networkLines);
                    var nn = NeuralNetwork.LoadFromFile(tempPath);
                    File.Delete(tempPath);

                    var ind = new Individual(nn, id);
                    ind.ParentId = parentId;
                    ind.Fitness = CalculateFitness(nn); // Вычисляем fitness при загрузке
                    _innerPopulation.Add(ind);
                    _nextId = Math.Max(_nextId, id + 1);
                }
                else
                {
                    idx++;
                }
            }
        }

        public void SavePopulation(string path)
        {
            using var writer = new StreamWriter(path);
            foreach (var ind in _innerPopulation)
            {
                writer.WriteLine($"INDIVIDUAL:{ind.Id}:{ind.ParentId}");
                
                // Сохраняем сеть во временный формат внутри файла
                var tempPath = Path.GetTempFileName();
                ind.Network.SaveToFile(tempPath);
                var networkLines = File.ReadAllLines(tempPath);
                File.Delete(tempPath);
                
                foreach (var nl in networkLines)
                    writer.WriteLine(nl);
            }
        }

        public FitnessResult CalculateFitness(NeuralNetwork nn)
        {
            decimal totalError = 0m;
            foreach (var pair in _data.Pairs)
            {
                var output = nn.Forward(pair.Input);
                for (int i = 0; i < pair.Output.Values.Length; i++)
                {
                    decimal diff = output.Values[i] - pair.Output.Values[i];
                    totalError += diff * diff;
                }
            }
            
            return new FitnessResult
            {
                Error = totalError,
                Complexity = nn.GetComplexity()
            };
        }

        private void ApplyMutation(NeuralNetwork nn)
        {
            int mutationType = _random.Next(6);
            
            switch (mutationType)
            {
                case 0: // Изменение веса связи
                    if (nn.Connections.Count > 0)
                    {
                        var conn = nn.Connections[_random.Next(nn.Connections.Count)];
                        // Меняем вес на случайную величину
                        conn.Weight = (decimal)(_random.NextDouble() * 10 - 5);
                    }
                    else if (nn.Neurons.Count > 1)
                    {
                        // Добавляем связь если нет
                        int from = _random.Next(nn.Neurons.Count);
                        int to = _random.Next(nn.Neurons.Count);
                        if (from != to)
                            nn.Connections.Add(new Connection { FromNeuronId = from, ToNeuronId = to, Weight = (decimal)(_random.NextDouble() * 2 - 1) });
                    }
                    break;

                case 1: // Удаление связи
                    if (nn.Connections.Count > 0)
                    {
                        int idx = _random.Next(nn.Connections.Count);
                        nn.Connections.RemoveAt(idx);
                    }
                    break;

                case 2: // Добавление связи
                    if (nn.Neurons.Count > 1)
                    {
                        int from = _random.Next(nn.Neurons.Count);
                        int to = _random.Next(nn.Neurons.Count);
                        if (from != to && !nn.Connections.Any(c => c.FromNeuronId == from && c.ToNeuronId == to))
                            nn.Connections.Add(new Connection { FromNeuronId = from, ToNeuronId = to, Weight = (decimal)(_random.NextDouble() * 2 - 1) });
                    }
                    break;

                case 3: // Добавление нейрона
                    {
                        int newId = nn.Neurons.Count > 0 ? nn.Neurons.Max(n => n.Id) + 1 : 0;
                        var types = Enum.GetValues(typeof(ActivationFunctionType));
                        nn.Neurons.Add(new Neuron
                        {
                            Id = newId,
                            ActivationType = (ActivationFunctionType)types.GetValue(_random.Next(types.Length))!,
                            Bias = (decimal)(_random.NextDouble() * 2 - 1)
                        });
                        
                        // Добавляем случайные связи к новому нейрону
                        foreach (var neuron in nn.Neurons.Where(n => n.Id != newId).Take(3))
                        {
                            if (_random.Next(2) == 0)
                                nn.Connections.Add(new Connection { FromNeuronId = neuron.Id, ToNeuronId = newId, Weight = (decimal)(_random.NextDouble() * 2 - 1) });
                            else
                                nn.Connections.Add(new Connection { FromNeuronId = newId, ToNeuronId = neuron.Id, Weight = (decimal)(_random.NextDouble() * 2 - 1) });
                        }
                    }
                    break;

                case 4: // Удаление нейрона (если не входной/выходной)
                    {
                        var removable = nn.Neurons.Where(n => n.Id >= nn.InputCount + nn.OutputCount).ToList();
                        if (removable.Count > 0)
                        {
                            var toRemove = removable[_random.Next(removable.Count)];
                            nn.Neurons.Remove(toRemove);
                            nn.Connections.RemoveAll(c => c.FromNeuronId == toRemove.Id || c.ToNeuronId == toRemove.Id);
                        }
                    }
                    break;

                case 5: // Изменение функции активации
                    {
                        if (nn.Neurons.Count > 0)
                        {
                            var neuron = nn.Neurons.First(n => n.Id >= nn.InputCount);
                            var types = Enum.GetValues(typeof(ActivationFunctionType));
                            neuron.ActivationType = (ActivationFunctionType)types.GetValue(_random.Next(types.Length))!;
                        }
                    }
                    break;
            }
        }

        public void EvolveGeneration()
        {
            var offspringList = new List<Individual>();

            // Каждая особь внутренней популяции создает потомков
            foreach (var parent in _innerPopulation)
            {
                for (int o = 0; o < _config.OffspringPerIndividual; o++)
                {
                    var childNetwork = parent.Network.Clone();
                    
                    // Применяем мутации
                    for (int m = 0; m < _config.MutationsPerOffspring; m++)
                        ApplyMutation(childNetwork);

                    var child = new Individual(childNetwork, _nextId++);
                    child.ParentId = parent.Id;
                    offspringList.Add(child);
                }
            }

            // Тестируем потомков и заменяем родителей если лучше
            foreach (var child in offspringList)
            {
                child.Fitness = CalculateFitness(child.Network);
                
                var parent = _innerPopulation.FirstOrDefault(p => p.Id == child.ParentId);
                if (parent != null)
                {
                    if (parent.Fitness == null)
                        parent.Fitness = CalculateFitness(parent.Network);

                    if (child.Fitness.IsBetterThan(parent.Fitness))
                    {
                        // Заменяем родителя
                        var idx = _innerPopulation.IndexOf(parent);
                        _innerPopulation[idx] = child;
                    }
                }
            }
        }

        public void RunEvolution(int generations, Action<int, decimal>? onGenerationEnd = null)
        {
            for (int gen = 0; gen < generations; gen++)
            {
                EvolveGeneration();
                
                var bestFitness = _innerPopulation
                    .Where(i => i.Fitness != null)
                    .OrderBy(i => i!.Fitness!.Error)
                    .ThenBy(i => i!.Fitness!.Complexity)
                    .FirstOrDefault()?.Fitness;

                onGenerationEnd?.Invoke(gen, bestFitness?.Error ?? -1m);
            }
        }

        public Individual GetBestIndividual()
        {
            return _innerPopulation
                .Where(i => i.Fitness != null)
                .OrderBy(i => i!.Fitness!.Error)
                .ThenBy(i => i!.Fitness!.Complexity)
                .FirstOrDefault() ?? _innerPopulation[0];
        }

        public List<Individual> GetInnerPopulation() => _innerPopulation;
    }
}
