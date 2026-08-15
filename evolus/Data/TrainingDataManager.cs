namespace evolus.Data;

/// <summary>
/// Represents a single training data pair (input -> expected output)
/// </summary>
public class TrainingDataPair
{
    public decimal[] Input { get; set; } = Array.Empty<decimal>();
    public decimal[] ExpectedOutput { get; set; } = Array.Empty<decimal>();

    public override string ToString()
    {
        return $"{string.Join(" ", Input)} | {string.Join(" ", ExpectedOutput)}";
    }
}

/// <summary>
/// Manages loading and saving training data from/to text files
/// Format: each line is "input1 input2 ... | output1 output2 ..."
/// </summary>
public class TrainingDataManager
{
    public List<TrainingDataPair> DataPairs { get; set; } = new();
    public int InputDimension { get; private set; }
    public int OutputDimension { get; private set; }

    /// <summary>
    /// Loads training data from a text file
    /// </summary>
    public void LoadFromFile(string filePath)
    {
        DataPairs.Clear();
        
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Training data file not found: {filePath}");
        
        var lines = File.ReadAllLines(filePath);
        bool dimensionsSet = false;
        
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            
            // Skip empty lines and comments
            if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#"))
                continue;
            
            // Parse the line: "input1 input2 ... | output1 output2 ..."
            var parts = trimmed.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2)
                throw new FormatException($"Invalid format in line: {line}. Expected 'inputs | outputs'");
            
            var inputParts = parts[0].Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            var outputParts = parts[1].Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            
            var input = inputParts.Select(decimal.Parse).ToArray();
            var output = outputParts.Select(decimal.Parse).ToArray();
            
            // Verify dimensions consistency
            if (!dimensionsSet)
            {
                InputDimension = input.Length;
                OutputDimension = output.Length;
                dimensionsSet = true;
            }
            else
            {
                if (input.Length != InputDimension)
                    throw new FormatException($"Input dimension mismatch in line: {line}. Expected {InputDimension}, got {input.Length}");
                if (output.Length != OutputDimension)
                    throw new FormatException($"Output dimension mismatch in line: {line}. Expected {OutputDimension}, got {output.Length}");
            }
            
            DataPairs.Add(new TrainingDataPair
            {
                Input = input,
                ExpectedOutput = output
            });
        }
        
        if (DataPairs.Count == 0)
            throw new InvalidOperationException("No training data pairs loaded from file");
    }

    /// <summary>
    /// Saves training data to a text file
    /// </summary>
    public void SaveToFile(string filePath)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("# Training Data File");
        sb.AppendLine("# Format: input1 input2 ... | output1 output2 ...");
        sb.AppendLine("# Each line represents one input-output pair");
        sb.AppendLine();
        
        foreach (var pair in DataPairs)
        {
            sb.AppendLine(pair.ToString());
        }
        
        File.WriteAllText(filePath, sb.ToString());
    }

    /// <summary>
    /// Adds a new data pair
    /// </summary>
    public void AddDataPair(decimal[] input, decimal[] output)
    {
        if (DataPairs.Count == 0)
        {
            InputDimension = input.Length;
            OutputDimension = output.Length;
        }
        else
        {
            if (input.Length != InputDimension)
                throw new ArgumentException($"Input dimension must be {InputDimension}");
            if (output.Length != OutputDimension)
                throw new ArgumentException($"Output dimension must be {OutputDimension}");
        }
        
        DataPairs.Add(new TrainingDataPair
        {
            Input = input,
            ExpectedOutput = output
        });
    }

    /// <summary>
    /// Clears all data
    /// </summary>
    public void Clear()
    {
        DataPairs.Clear();
        InputDimension = 0;
        OutputDimension = 0;
    }
}
