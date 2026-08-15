namespace evolus.Config;

/// <summary>
/// Configuration settings loaded from/saved to text files
/// </summary>
public class EvolutionConfig
{
    public int InnerPopulationSize { get; set; } = 10;
    public int OffspringPerIndividual { get; set; } = 5;
    public int MutationsPerOffspring { get; set; } = 3;
    public int GenerationsToRun { get; set; } = 100;
    public string TrainingDataPath { get; set; } = "data/training.txt";
    public string PopulationSavePath { get; set; } = "data/population";
    public int RandomSeed { get; set; } = 42;

    public void LoadFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            SaveToFile(filePath);
            return;
        }

        var lines = File.ReadAllLines(filePath);
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#"))
                continue;

            var parts = trimmed.Split('=');
            if (parts.Length != 2)
                continue;

            var key = parts[0].Trim();
            var value = parts[1].Trim();

            switch (key)
            {
                case "InnerPopulationSize":
                    InnerPopulationSize = int.Parse(value);
                    break;
                case "OffspringPerIndividual":
                    OffspringPerIndividual = int.Parse(value);
                    break;
                case "MutationsPerOffspring":
                    MutationsPerOffspring = int.Parse(value);
                    break;
                case "GenerationsToRun":
                    GenerationsToRun = int.Parse(value);
                    break;
                case "TrainingDataPath":
                    TrainingDataPath = value;
                    break;
                case "PopulationSavePath":
                    PopulationSavePath = value;
                    break;
                case "RandomSeed":
                    RandomSeed = int.Parse(value);
                    break;
            }
        }
    }

    public void SaveToFile(string filePath)
    {
        var dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("# Evolution Configuration File");
        sb.AppendLine("# Each setting is on its own line: Key=Value");
        sb.AppendLine();
        sb.AppendLine($"InnerPopulationSize={InnerPopulationSize}");
        sb.AppendLine($"OffspringPerIndividual={OffspringPerIndividual}");
        sb.AppendLine($"MutationsPerOffspring={MutationsPerOffspring}");
        sb.AppendLine($"GenerationsToRun={GenerationsToRun}");
        sb.AppendLine($"TrainingDataPath={TrainingDataPath}");
        sb.AppendLine($"PopulationSavePath={PopulationSavePath}");
        sb.AppendLine($"RandomSeed={RandomSeed}");

        File.WriteAllText(filePath, sb.ToString());
    }
}
