using System;
using System.Globalization;
using evolus.Core;

namespace evolus.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Переходим в корневую директорию приложения
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            // Пытаемся найти config/settings.txt относительно разных путей
            string[] possibleBaseDirs = new[] {
                Directory.GetCurrentDirectory(),
                Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? "",
                Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."))
            };

            string? foundBaseDir = null;
            foreach (var dir in possibleBaseDirs)
            {
                if (File.Exists(Path.Combine(dir, "config", "settings.txt")))
                {
                    foundBaseDir = dir;
                    break;
                }
            }

            if (foundBaseDir != null)
                Directory.SetCurrentDirectory(foundBaseDir);
            
            var config = new Config();
            config.LoadFromFile("config/settings.txt");

            var data = new TrainingData();
            if (System.IO.File.Exists(config.DataFilePath))
                data.LoadFromFile(config.DataFilePath);
            else
            {
                Console.WriteLine($"Training data file not found: {config.DataFilePath}");
                Console.WriteLine("Creating sample training data...");
                data.Pairs.Add(new DataPair(new DecimalVector(2) { Values = new[] { 0m, 0m } }, new DecimalVector(1) { Values = new[] { 0m } }));
                data.Pairs.Add(new DataPair(new DecimalVector(2) { Values = new[] { 0m, 1m } }, new DecimalVector(1) { Values = new[] { 1m } }));
                data.Pairs.Add(new DataPair(new DecimalVector(2) { Values = new[] { 1m, 0m } }, new DecimalVector(1) { Values = new[] { 1m } }));
                data.Pairs.Add(new DataPair(new DecimalVector(2) { Values = new[] { 1m, 1m } }, new DecimalVector(1) { Values = new[] { 0m } }));
                data.SaveToFile(config.DataFilePath);
                Console.WriteLine($"Sample XOR data saved to {config.DataFilePath}");
            }

            var engine = new EvolutionEngine(config, data);
            
            // Загружаем сохраненную популяцию если есть
            if (System.IO.File.Exists(config.SaveFilePath))
            {
                Console.WriteLine("Loading saved population...");
                engine.LoadPopulation(config.SaveFilePath);
            }
            else
            {
                Console.WriteLine("Initializing new population...");
                engine.Initialize();
            }

            bool running = true;
            while (running)
            {
                Console.WriteLine("\n=== evolus - Главное меню ===");
                Console.WriteLine("1. Эволюция");
                Console.WriteLine("2. Протестировать текущую особь");
                Console.WriteLine("3. Сохранить прогресс");
                Console.WriteLine("4. Загрузить прогресс");
                Console.WriteLine("5. Показать лучшую особь");
                Console.WriteLine("6. Выход");
                Console.Write("Выберите пункт: ");

                string? input = Console.ReadLine();
                if (string.IsNullOrEmpty(input)) continue;
                char key = input[0];

                switch (key)
                {
                    case '1':
                        RunEvolution(engine, config);
                        break;
                    case '2':
                        TestIndividual(engine);
                        break;
                    case '3':
                        engine.SavePopulation(config.SaveFilePath);
                        Console.WriteLine($"Прогресс сохранен в {config.SaveFilePath}");
                        break;
                    case '4':
                        if (System.IO.File.Exists(config.SaveFilePath))
                        {
                            engine.LoadPopulation(config.SaveFilePath);
                            Console.WriteLine("Прогресс загружен.");
                        }
                        else
                            Console.WriteLine("Файл сохранения не найден.");
                        break;
                    case '5':
                        ShowBestIndividual(engine);
                        break;
                    case '6':
                        running = false;
                        break;
                }
            }
        }

        static void RunEvolution(EvolutionEngine engine, Config config)
        {
            Console.Write("Введите количество поколений: ");
            string? input = Console.ReadLine();
            if (!int.TryParse(input, out int generations) || generations <= 0)
            {
                Console.WriteLine("Некорректное количество поколений.");
                return;
            }

            Console.WriteLine($"\nЗапуск эволюции на {generations} поколений...");
            
            engine.RunEvolution(generations, (gen, error) =>
            {
                if ((gen + 1) % 10 == 0 || gen == 0)
                    Console.WriteLine($"Поколение {gen + 1}: Ошибка = {error.ToString(CultureInfo.InvariantCulture)}");
            });

            var best = engine.GetBestIndividual();
            Console.WriteLine("\n=== Отчет об эволюции ===");
            Console.WriteLine($"Лучшая ошибка: {best.Fitness?.Error.ToString(CultureInfo.InvariantCulture)}");
            Console.WriteLine($"Сложность: {best.Fitness?.Complexity}");
            Console.WriteLine($"Нейронов: {best.Network.Neurons.Count}");
            Console.WriteLine($"Связей: {best.Network.Connections.Count}");
            Console.WriteLine("\nНажмите Enter для возврата в меню...");
            Console.ReadLine();
        }

        static void TestIndividual(EvolutionEngine engine)
        {
            var best = engine.GetBestIndividual();
            Console.WriteLine($"\nТестирование лучшей особи (Ошибка: {best.Fitness?.Error}, Сложность: {best.Fitness?.Complexity})");
            Console.WriteLine($"Входных нейронов: {best.Network.InputCount}, Выходных: {best.Network.OutputCount}");
            Console.WriteLine("Введите входные данные через пробел (или 'q' для выхода):");

            while (true)
            {
                Console.Write("> ");
                string? input = Console.ReadLine();
                if (input == null || input.ToLower() == "q") break;

                try
                {
                    var inputVec = DecimalVector.Parse(input);
                    if (inputVec.Values.Length != best.Network.InputCount)
                    {
                        Console.WriteLine($"Ошибка: ожидается {best.Network.InputCount} значений, введено {inputVec.Values.Length}");
                        continue;
                    }

                    var output = best.Network.Forward(inputVec);
                    Console.WriteLine($"Результат: {output}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка: {ex.Message}");
                }
            }
        }

        static void ShowBestIndividual(EvolutionEngine engine)
        {
            var best = engine.GetBestIndividual();
            Console.WriteLine("\n=== Лучшая особь ===");
            Console.WriteLine($"ID: {best.Id}, Родитель: {best.ParentId}");
            Console.WriteLine($"Ошибка: {best.Fitness?.Error.ToString(CultureInfo.InvariantCulture)}");
            Console.WriteLine($"Сложность: {best.Fitness?.Complexity}");
            Console.WriteLine($"Нейронов: {best.Network.Neurons.Count}");
            Console.WriteLine($"Связей: {best.Network.Connections.Count}");
            Console.WriteLine("\nСтруктура сети:");
            foreach (var n in best.Network.Neurons)
                Console.WriteLine($"  Нейрон {n.Id}: Функция={n.ActivationType}, Bias={n.Bias}");
            Console.WriteLine("\nСвязи:");
            foreach (var c in best.Network.Connections.Take(20))
                Console.WriteLine($"  {c.FromNeuronId} -> {c.ToNeuronId} : {c.Weight}");
            if (best.Network.Connections.Count > 20)
                Console.WriteLine($"  ... и еще {best.Network.Connections.Count - 20} связей");
            
            Console.WriteLine("\nНажмите Enter для продолжения...");
            Console.ReadLine();
        }
    }
}
