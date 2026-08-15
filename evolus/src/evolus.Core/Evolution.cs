using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace evolus.Core
{
    /// <summary>
    /// Пара входных-выходных данных для обучения/тестирования
    /// </summary>
    public class DataPair
    {
        public decimal[] Input { get; set; }
        public decimal[] Output { get; set; }

        public DataPair(decimal[] input, decimal[] output)
        {
            Input = input;
            Output = output;
        }
    }

    /// <summary>
    /// Менеджер для загрузки и управления данными обучения
    /// </summary>
    public class DataManager
    {
        public List<DataPair> DataPairs { get; set; } = new List<DataPair>();
        public int InputDimension { get; private set; } = 0;
        public int OutputDimension { get; private set; } = 0;

        /// <summary>
        /// Загружает данные из текстового файла
        /// Формат: "input1 input2 | output1 output2"
        /// </summary>
        public void LoadFromFile(string filePath)
        {
            DataPairs.Clear();
            InputDimension = 0;
            OutputDimension = 0;

            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Data file not found: {filePath}");

            using (var reader = new StreamReader(filePath))
            {
                string line;
                int lineNumber = 0;

                while ((line = reader.ReadLine()) != null)
                {
                    lineNumber++;
                    line = line.Trim();

                    // Пропускаем пустые строки и комментарии
                    if (string.IsNullOrEmpty(line) || line.StartsWith("#"))
                        continue;

                    var parts = line.Split('|');
                    if (parts.Length != 2)
                        throw new FormatException($"Line {lineNumber}: Expected format 'inputs | outputs'");

                    var inputValues = parts[0].Trim()
                        .Split(' ')
                        .Where(s => !string.IsNullOrEmpty(s))
                        .Select(decimal.Parse)
                        .ToArray();

                    var outputValues = parts[1].Trim()
                        .Split(' ')
                        .Where(s => !string.IsNullOrEmpty(s))
                        .Select(decimal.Parse)
                        .ToArray();

                    if (InputDimension == 0)
                    {
                        InputDimension = inputValues.Length;
                        OutputDimension = outputValues.Length;
                    }
                    else
                    {
                        if (inputValues.Length != InputDimension)
                            throw new FormatException($"Line {lineNumber}: Input dimension mismatch");
                        if (outputValues.Length != OutputDimension)
                            throw new FormatException($"Line {lineNumber}: Output dimension mismatch");
                    }

                    DataPairs.Add(new DataPair(inputValues, outputValues));
                }
            }

            if (DataPairs.Count == 0)
                throw new InvalidOperationException("No data pairs loaded from file");
        }

        /// <summary>
        /// Сохраняет данные в текстовый файл
        /// </summary>
        public void SaveToFile(string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            {
                writer.WriteLine("# Evolus Training Data");
                writer.WriteLine($"# Input dimension: {InputDimension}, Output dimension: {OutputDimension}");
                writer.WriteLine("# Format: input1 input2 ... | output1 output2 ...");
                writer.WriteLine();

                foreach (var pair in DataPairs)
                {
                    var inputStr = string.Join(" ", pair.Input);
                    var outputStr = string.Join(" ", pair.Output);
                    writer.WriteLine($"{inputStr} | {outputStr}");
                }
            }
        }
    }

    /// <summary>
    /// Особь в эволюционном алгоритме
    /// </summary>
    public class Individual
    {
        public NeuralNetwork Network { get; set; }
        public decimal Fitness { get; set; } = decimal.MaxValue;
        public int ParentId { get; set; } = -1;
        public int Id { get; set; }
        public int Generation { get; set; }

        public Individual(NeuralNetwork network, int id, int generation, int parentId = -1)
        {
            Network = network;
            Id = id;
            Generation = generation;
            ParentId = parentId;
        }

        /// <summary>
        /// Вычисляет приспособленность особи на основе всех пар данных
        /// Возвращает ошибку (0 = идеальное выполнение)
        /// </summary>
        public decimal CalculateFitness(List<DataPair> dataPairs)
        {
            decimal totalError = 0m;

            foreach (var pair in dataPairs)
            {
                try
                {
                    var outputs = Network.Forward(pair.Input);

                    for (int i = 0; i < Math.Min(outputs.Length, pair.Output.Length); i++)
                    {
                        totalError += Math.Abs(outputs[i] - pair.Output[i]);
                    }
                }
                catch
                {
                    // Если сеть не может обработать вход, считаем ошибку максимальной
                    return decimal.MaxValue;
                }
            }

            Fitness = totalError;
            return totalError;
        }
    }

    /// <summary>
    /// Конфигурация эволюционного алгоритма
    /// </summary>
    public class EvolutionConfig
    {
        public int PopulationSize { get; set; } = 50;
        public int OffspringPerIndividual { get; set; } = 3;
        public int MutationsPerOffspring { get; set; } = 5;
        public int MaxGenerations { get; set; } = 1000;
        public int RandomSeed { get; set; } = 42;

        public static EvolutionConfig LoadFromFile(string filePath)
        {
            var config = new EvolutionConfig();

            if (!File.Exists(filePath))
            {
                config.SaveToFile(filePath);
                return config;
            }

            using (var reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (string.IsNullOrEmpty(line) || line.StartsWith("#"))
                        continue;

                    var parts = line.Split('=');
                    if (parts.Length != 2)
                        continue;

                    var key = parts[0].Trim();
                    var value = parts[1].Trim();

                    switch (key.ToLower())
                    {
                        case "populationsize":
                            config.PopulationSize = int.Parse(value);
                            break;
                        case "offspringperindividual":
                            config.OffspringPerIndividual = int.Parse(value);
                            break;
                        case "mutationsperoffspring":
                            config.MutationsPerOffspring = int.Parse(value);
                            break;
                        case "maxgenerations":
                            config.MaxGenerations = int.Parse(value);
                            break;
                        case "randomseed":
                            config.RandomSeed = int.Parse(value);
                            break;
                    }
                }
            }

            return config;
        }

        public void SaveToFile(string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            {
                writer.WriteLine("# Evolus Evolution Configuration");
                writer.WriteLine($"PopulationSize={PopulationSize}");
                writer.WriteLine($"OffspringPerIndividual={OffspringPerIndividual}");
                writer.WriteLine($"MutationsPerOffspring={MutationsPerOffspring}");
                writer.WriteLine($"MaxGenerations={MaxGenerations}");
                writer.WriteLine($"RandomSeed={RandomSeed}");
            }
        }
    }

    /// <summary>
    /// Эволюционный движок
    /// </summary>
    public class EvolutionEngine
    {
        private readonly EvolutionConfig _config;
        private readonly List<DataPair> _dataPairs;
        private readonly Random _random;
        
        // Внутренняя популяция (неприкосновенные особи)
        private List<Individual> _innerPopulation;
        
        // Внешняя популяция (потомки для тестирования)
        private List<Individual> _outerPopulation;

        private int _currentGeneration;
        private int _nextIndividualId;

        public EvolutionEngine(EvolutionConfig config, List<DataPair> dataPairs)
        {
            _config = config;
            _dataPairs = dataPairs;
            _random = new Random(config.RandomSeed);
            _innerPopulation = new List<Individual>();
            _outerPopulation = new List<Individual>();
            _currentGeneration = 0;
            _nextIndividualId = 0;
        }

        /// <summary>
        /// Инициализирует эволюцию с нулевой нейросети
        /// </summary>
        public void Initialize(int inputCount, int outputCount)
        {
            _innerPopulation.Clear();
            _outerPopulation.Clear();
            _currentGeneration = 0;
            _nextIndividualId = 0;

            // Создаем начальную популяцию из совершенно нулевых сетей
            for (int i = 0; i < _config.PopulationSize; i++)
            {
                var network = new NeuralNetwork(inputCount, outputCount);
                var individual = new Individual(network, _nextIndividualId++, _currentGeneration);
                individual.CalculateFitness(_dataPairs);
                _innerPopulation.Add(individual);
            }
        }

        /// <summary>
        /// Выполняет одно поколение эволюции
        /// </summary>
        public EvolutionResult EvolveOneGeneration()
        {
            _outerPopulation.Clear();

            // Каждая особь внутренней популяции создает потомков
            foreach (var parent in _innerPopulation)
            {
                for (int i = 0; i < _config.OffspringPerIndividual; i++)
                {
                    var childNetwork = parent.Network.Clone();
                    childNetwork.ApplyMutations(_random, _config.MutationsPerOffspring);

                    var child = new Individual(childNetwork, _nextIndividualId++, _currentGeneration + 1, parent.Id);
                    child.CalculateFitness(_dataPairs);
                    _outerPopulation.Add(child);
                }
            }

            // Сортируем потомков по приспособленности (лучшие первые)
            // Сначала по ошибке (меньше = лучше), затем по сложности (меньше = лучше при равной ошибке)
            var sortedOffspring = _outerPopulation
                .OrderBy(x => x.Fitness)
                .ThenBy(x => x.Network.Complexity)
                .ToList();

            // Заменяем родителей только если потомок лучше
            var newInnerPopulation = new List<Individual>();
            var replacedCount = 0;

            foreach (var parent in _innerPopulation)
            {
                // Находим лучшего потомка этого родителя
                var parentOffspring = sortedOffspring
                    .Where(o => o.ParentId == parent.Id)
                    .OrderBy(o => o.Fitness)
                    .ThenBy(o => o.Network.Complexity)
                    .FirstOrDefault();

                if (parentOffspring != null)
                {
                    // Прямое сравнение: потомок заменяет родителя только если ошибка меньше
                    // Или если ошибка равна, но сложность меньше
                    bool isBetter = parentOffspring.Fitness < parent.Fitness ||
                                   (parentOffspring.Fitness == parent.Fitness && 
                                    parentOffspring.Network.Complexity < parent.Network.Complexity);

                    if (isBetter)
                    {
                        newInnerPopulation.Add(parentOffspring);
                        replacedCount++;
                    }
                    else
                    {
                        newInnerPopulation.Add(parent);
                    }
                }
                else
                {
                    newInnerPopulation.Add(parent);
                }
            }

            _innerPopulation = newInnerPopulation;
            _currentGeneration++;

            // Находим лучшую особь
            var bestIndividual = _innerPopulation
                .OrderBy(x => x.Fitness)
                .ThenBy(x => x.Network.Complexity)
                .First();

            return new EvolutionResult
            {
                Generation = _currentGeneration,
                BestFitness = bestIndividual.Fitness,
                BestComplexity = bestIndividual.Network.Complexity,
                ReplacedCount = replacedCount,
                AverageFitness = _innerPopulation.Average(x => x.Fitness)
            };
        }

        /// <summary>
        /// Выполняет указанное количество поколений
        /// </summary>
        public EvolutionResult EvolveGenerations(int generations)
        {
            EvolutionResult lastResult = null;

            for (int i = 0; i < generations; i++)
            {
                lastResult = EvolveOneGeneration();
            }

            return lastResult;
        }

        /// <summary>
        /// Получает лучшую особь из внутренней популяции
        /// </summary>
        public Individual GetBestIndividual()
        {
            return _innerPopulation
                .OrderBy(x => x.Fitness)
                .ThenBy(x => x.Network.Complexity)
                .First();
        }

        /// <summary>
        /// Сохраняет состояние эволюции (внутреннюю популяцию)
        /// </summary>
        public void SaveProgress(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            // Сохраняем каждую особь внутренней популяции
            for (int i = 0; i < _innerPopulation.Count; i++)
            {
                var individual = _innerPopulation[i];
                var filePath = Path.Combine(directoryPath, $"individual_{i}.net");
                individual.Network.SaveToFile(filePath);
            }

            // Сохраняем метаданные
            var metadataPath = Path.Combine(directoryPath, "metadata.txt");
            using (var writer = new StreamWriter(metadataPath))
            {
                writer.WriteLine($"CurrentGeneration={_currentGeneration}");
                writer.WriteLine($"NextIndividualId={_nextIndividualId}");
                writer.WriteLine($"PopulationSize={_innerPopulation.Count}");
                
                for (int i = 0; i < _innerPopulation.Count; i++)
                {
                    var ind = _innerPopulation[i];
                    writer.WriteLine($"Individual_{i}_Fitness={ind.Fitness}");
                    writer.WriteLine($"Individual_{i}_ParentId={ind.ParentId}");
                    writer.WriteLine($"Individual_{i}_Generation={ind.Generation}");
                }
            }
        }

        /// <summary>
        /// Загружает состояние эволюции
        /// </summary>
        public void LoadProgress(string directoryPath)
        {
            var metadataPath = Path.Combine(directoryPath, "metadata.txt");
            if (!File.Exists(metadataPath))
                throw new FileNotFoundException("Metadata file not found");

            _innerPopulation.Clear();
            
            using (var reader = new StreamReader(metadataPath))
            {
                string line;
                var individualData = new Dictionary<int, (decimal Fitness, int ParentId, int Generation)>();

                while ((line = reader.ReadLine()) != null)
                {
                    var parts = line.Split('=');
                    if (parts.Length != 2)
                        continue;

                    var key = parts[0].Trim();
                    var value = parts[1].Trim();

                    switch (key)
                    {
                        case "CurrentGeneration":
                            _currentGeneration = int.Parse(value);
                            break;
                        case "NextIndividualId":
                            _nextIndividualId = int.Parse(value);
                            break;
                        case "PopulationSize":
                            // Просто читаем, размер будет определен по файлам
                            break;
                    }

                    if (key.StartsWith("Individual_") && key.EndsWith("_Fitness"))
                    {
                        var index = int.Parse(key.Split('_')[1]);
                        var fitness = decimal.Parse(value);
                        
                        // Читаем остальные данные для этой особи
                        // (в реальном коде нужно читать все сразу, здесь упрощено)
                    }
                }
            }

            // Загружаем сети особей
            for (int i = 0; ; i++)
            {
                var filePath = Path.Combine(directoryPath, $"individual_{i}.net");
                if (!File.Exists(filePath))
                    break;

                var network = NeuralNetwork.LoadFromFile(filePath);
                var individual = new Individual(network, i, _currentGeneration);
                individual.CalculateFitness(_dataPairs);
                _innerPopulation.Add(individual);
            }

            // Перечитываем метаданные с правильной привязкой к особям
            using (var reader = new StreamReader(metadataPath))
            {
                string line;
                int currentIndex = -1;

                while ((line = reader.ReadLine()) != null)
                {
                    var parts = line.Split('=');
                    if (parts.Length != 2)
                        continue;

                    var key = parts[0].Trim();
                    var value = parts[1].Trim();

                    if (key.StartsWith("Individual_") && key.Contains("_Fitness"))
                    {
                        currentIndex = int.Parse(key.Split('_')[1]);
                        if (currentIndex < _innerPopulation.Count)
                        {
                            _innerPopulation[currentIndex].Fitness = decimal.Parse(value);
                        }
                    }
                    else if (key.StartsWith("Individual_") && key.Contains("_ParentId"))
                    {
                        if (currentIndex >= 0 && currentIndex < _innerPopulation.Count)
                        {
                            _innerPopulation[currentIndex].ParentId = int.Parse(value);
                        }
                    }
                    else if (key.StartsWith("Individual_") && key.Contains("_Generation"))
                    {
                        if (currentIndex >= 0 && currentIndex < _innerPopulation.Count)
                        {
                            _innerPopulation[currentIndex].Generation = int.Parse(value);
                        }
                    }
                }
            }
        }

        public int CurrentGeneration => _currentGeneration;
        public List<Individual> InnerPopulation => _innerPopulation;
    }

    /// <summary>
    /// Результат эволюции
    /// </summary>
    public class EvolutionResult
    {
        public int Generation { get; set; }
        public decimal BestFitness { get; set; }
        public int BestComplexity { get; set; }
        public int ReplacedCount { get; set; }
        public decimal AverageFitness { get; set; }

        public override string ToString()
        {
            return $"Generation: {Generation}\n" +
                   $"Best Fitness: {BestFitness}\n" +
                   $"Best Complexity: {BestComplexity}\n" +
                   $"Replaced: {ReplacedCount}\n" +
                   $"Average Fitness: {AverageFitness}";
        }
    }
}
