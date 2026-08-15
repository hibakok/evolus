using evolus.Config;
using evolus.Data;
using evolus.UI;

// Main entry point for the evolus application
Console.WriteLine("=== EVOLUS - Evolutionary Neural Network System ===");
Console.WriteLine();

// Load configuration
var config = new EvolutionConfig();
config.LoadFromFile("config.txt");
Console.WriteLine($"Configuration loaded from config.txt");

// Initialize training data manager
var trainingData = new TrainingDataManager();

// Create and run UI
var ui = new ConsoleUI(config, trainingData);
ui.Initialize();
ui.Run();
