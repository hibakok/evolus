using System;
using System.Globalization;
using System.IO;
using evolus.Core;

namespace evolus.ConsoleApp;

class Program
{
    private static string BasePath => AppContext.BaseDirectory;
    private static string ConfigPath => Path.Combine(BasePath, "..", "..", "..", "..", "config", "settings.txt");
    private static string DataPath => Path.Combine(BasePath, "..", "..", "..", "..", "data", "training.txt");
    private static string SavePath => Path.Combine(BasePath, "..", "..", "..", "..", "save", "population.txt");
    
    // Для отладки используем относительные пути от рабочей директории
    private static string GetConfigPath() => Path.Combine(Directory.GetCurrentDirectory(), "config", "settings.txt");
    private static string GetDataPath() => Path.Combine(Directory.GetCurrentDirectory(), "data", "training.txt");
    private static string GetSavePath() => Path.Combine(Directory.GetCurrentDirectory(), "save", "population.txt");

    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        
        Console.WriteLine("╔════════════════════════════════════════╗");
        Console.WriteLine("║         evolus - Neural Network        ║");
        Console.WriteLine("║      Evolutionary Strategy Engine      ║");
        Console.WriteLine("╚════════════════════════════════════════╝");
        Console.WriteLine();

        // Загружаем настройки
        var configPath = GetConfigPath();
        var settings = SettingsLoader.LoadFromFile(configPath);
        Console.WriteLine($"Настройки загружены из: {configPath}");
        Console.WriteLine($"  Внутренняя популяция: {settings.InnerPopulationSize}");
        Console.WriteLine($"  Потомков на особь: {settings.OffspringPerIndividual}");
        Console.WriteLine($"  Мутаций на потомка: {settings.MutationsPerOffspring}");
        Console.WriteLine($"  Входов: {settings.InputCount}, Выходов: {settings.OutputCount}");
        Console.WriteLine();

        // Загружаем обучающие данные
        var dataPath = GetDataPath();
        List<TrainingPair> trainingData;
        try
        {
            trainingData = TrainingDataLoader.LoadFromFile(dataPath);
            Console.WriteLine($"Обучающих пар загружено: {trainingData.Count}");
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine($"Файл обучающих данных не найден: {dataPath}");
            Console.WriteLine("Создан файл с примером данных XOR.");
            
            // Создаем пример данных XOR
            var xorData = new List<TrainingPair>
            {
                new TrainingPair(new decimal[] { 0, 0 }, new decimal[] { 0 }),
                new TrainingPair(new decimal[] { 0, 1 }, new decimal[] { 1 }),
                new TrainingPair(new decimal[] { 1, 0 }, new decimal[] { 1 }),
                new TrainingPair(new decimal[] { 1, 1 }, new decimal[] { 0 })
            };
            
            Directory.CreateDirectory(Path.GetDirectoryName(dataPath)!);
            TrainingDataLoader.SaveToFile(dataPath, xorData);
            trainingData = xorData;
            
            // Обновляем настройки под XOR
            settings.InputCount = 2;
            settings.OutputCount = 1;
            SettingsLoader.SaveToFile(configPath, settings);
        }
        Console.WriteLine();

        // Инициализируем эволюционный движок
        var engine = new EvolutionEngine(settings, trainingData);
        var savePath = GetSavePath();
        
        // Пытаемся загрузить сохранение
        if (File.Exists(savePath))
        {
            Console.WriteLine($"Загрузка сохранения из: {savePath}");
            engine.LoadPopulation(savePath);
            Console.WriteLine($"Поколение: {engine.CurrentGeneration}");
            if (engine.BestEver != null)
            {
                Console.WriteLine($"Лучшая приспособленность: {engine.BestEver.Fitness.ToString("G29", CultureInfo.InvariantCulture)}");
            }
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("Сохранение не найдено. Инициализация новой популяции...");
            engine.InitializePopulation();
            Console.WriteLine();
        }

        // Главное меню
        bool running = true;
        while (running)
        {
            Console.WriteLine("═══════════════════════════════════════");
            Console.WriteLine("ГЛАВНОЕ МЕНЮ:");
            Console.WriteLine("  1. Эволюция (указать количество поколений)");
            Console.WriteLine("  2. Протестировать текущую лучшую особь");
            Console.WriteLine("  3. Показать статистику популяции");
            Console.WriteLine("  4. Сохранить прогресс");
            Console.WriteLine("  5. Загрузить прогресс");
            Console.WriteLine("  6. Выход");
            Console.WriteLine("═══════════════════════════════════════");
            Console.Write("Выберите пункт: ");

            var input = Console.ReadLine();
            Console.WriteLine();

            if (string.IsNullOrEmpty(input)) continue;
            
            var keyChar = input.Trim().FirstOrDefault();
            
            switch (keyChar)
            {
                case '1':
                    RunEvolution(engine);
                    break;
                case '2':
                    TestBestIndividual(engine);
                    break;
                case '3':
                    ShowStatistics(engine);
                    break;
                case '4':
                    SaveProgress(engine, savePath);
                    break;
                case '5':
                    LoadProgress(engine, savePath);
                    break;
                case '6':
                    running = false;
                    break;
                default:
                    Console.WriteLine("Неверный выбор. Попробуйте снова.");
                    break;
            }
        }

        Console.WriteLine("До свидания!");
    }

    private static void RunEvolution(EvolutionEngine engine)
    {
        Console.Write("Введите количество поколений для эволюции: ");
        var input = Console.ReadLine();
        
        if (!int.TryParse(input, out int generations) || generations <= 0)
        {
            Console.WriteLine("Неверное количество поколений.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine($"Запуск эволюции на {generations} поколений...");
        Console.WriteLine();

        var startTime = DateTime.Now;
        
        for (int i = 0; i < generations; i++)
        {
            engine.EvolveOneGeneration();
            
            if ((i + 1) % 100 == 0 || i == generations - 1)
            {
                var stats = engine.GetStatistics();
                Console.WriteLine($"Поколение {stats.Generation}: " +
                    $"Лучшая={stats.BestFitness.ToString("G6", CultureInfo.InvariantCulture)}, " +
                    $"Средняя={stats.AverageFitness.ToString("G6", CultureInfo.InvariantCulture)}, " +
                    $"Сложность={stats.BestComplexity}");
            }
        }

        var endTime = DateTime.Now;
        var duration = endTime - startTime;

        Console.WriteLine();
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine("ОТЧЕТ ОБ ЭВОЛЮЦИИ:");
        Console.WriteLine($"  Поколений выполнено: {generations}");
        Console.WriteLine($"  Время выполнения: {duration.TotalSeconds:F2} сек");
        
        var finalStats = engine.GetStatistics();
        Console.WriteLine($"  Лучшая приспособленность: {finalStats.BestFitness.ToString("G29", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"  Сложность лучшей особи: {finalStats.BestComplexity}");
        
        if (engine.BestEver != null)
        {
            Console.WriteLine($"  Нейронов в лучшей: {engine.BestEver.Network.Neurons.Count}");
            Console.WriteLine($"  Связей в лучшей: {engine.BestEver.Network.Connections.Count}");
        }
        
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine();
        Console.WriteLine("Нажмите Enter для возврата в меню...");
        Console.ReadLine();
        Console.WriteLine();
    }

    private static void TestBestIndividual(EvolutionEngine engine)
    {
        var best = engine.BestEver ?? engine.GetBestIndividual();
        
        Console.WriteLine();
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine("ТЕСТИРОВАНИЕ ЛУЧШЕЙ ОСОБИ");
        Console.WriteLine($"Приспособленность: {best.Fitness.ToString("G29", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"Сложность: {best.Network.Complexity}");
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine();
        Console.WriteLine("Вводите входные данные через пробел (или 'q' для выхода):");

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine();
            
            if (input?.Trim().ToLower() == "q")
                break;

            try
            {
                var values = input?.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(x => decimal.Parse(x, CultureInfo.InvariantCulture))
                    .ToArray() ?? Array.Empty<decimal>();

                if (values.Length != best.Network.InputCount)
                {
                    Console.WriteLine($"Ошибка: ожидалось {best.Network.InputCount} значений, введено {values.Length}");
                    continue;
                }

                var outputs = best.Network.Forward(values);
                
                Console.Write("Выход: ");
                Console.WriteLine(string.Join(" ", outputs.Select(x => x.ToString("G17", CultureInfo.InvariantCulture))));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка: {ex.Message}");
            }
        }

        Console.WriteLine();
    }

    private static void ShowStatistics(EvolutionEngine engine)
    {
        var stats = engine.GetStatistics();
        
        Console.WriteLine();
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine("СТАТИСТИКА ПОПУЛЯЦИИ:");
        Console.WriteLine($"  Текущее поколение: {stats.Generation}");
        Console.WriteLine($"  Размер популяции: {stats.PopulationSize}");
        Console.WriteLine($"  Лучшая приспособленность: {stats.BestFitness.ToString("G29", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"  Худшая приспособленность: {stats.WorstFitness.ToString("G29", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"  Средняя приспособленность: {stats.AverageFitness.ToString("G6", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"  Сложность лучшей особи: {stats.BestComplexity}");
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine();
        Console.WriteLine("Нажмите любую клавишу...");
        Console.ReadKey();
        Console.WriteLine();
    }

    private static void SaveProgress(EvolutionEngine engine, string savePath)
    {
        try
        {
            engine.SavePopulation(savePath);
            Console.WriteLine();
            Console.WriteLine($"Прогресс успешно сохранен в: {savePath}");
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine();
            Console.WriteLine($"Ошибка сохранения: {ex.Message}");
            Console.WriteLine();
        }
    }

    private static void LoadProgress(EvolutionEngine engine, string savePath)
    {
        try
        {
            if (!File.Exists(savePath))
            {
                Console.WriteLine();
                Console.WriteLine("Сохранение не найдено.");
                Console.WriteLine();
                return;
            }
            
            engine.LoadPopulation(savePath);
            Console.WriteLine();
            Console.WriteLine($"Прогресс успешно загружен из: {savePath}");
            Console.WriteLine($"Текущее поколение: {engine.CurrentGeneration}");
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine();
            Console.WriteLine($"Ошибка загрузки: {ex.Message}");
            Console.WriteLine();
        }
    }
}
