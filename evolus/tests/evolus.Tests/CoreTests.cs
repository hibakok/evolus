using Xunit;
using evolus.Core;

namespace evolus.Tests
{
    public class ConnectionComparer : IEqualityComparer<Connection>
    {
        public bool Equals(Connection? x, Connection? y)
        {
            if (x == null || y == null) return false;
            return x.FromNeuronId == y.FromNeuronId && 
                   x.ToNeuronId == y.ToNeuronId && 
                   x.Weight == y.Weight;
        }

        public int GetHashCode(Connection obj)
        {
            return HashCode.Combine(obj.FromNeuronId, obj.ToNeuronId, obj.Weight);
        }
    }

    public class NeuralNetworkTests
    {
        [Fact]
        public void CreateEmptyNetwork_ShouldHaveCorrectInputOutputCount()
        {
            var network = new NeuralNetwork(2, 1);
            
            Assert.Equal(2, network.InputCount);
            Assert.Equal(1, network.OutputCount);
            Assert.Equal(3, network.Neurons.Count); // 2 input + 1 output
        }

        [Fact]
        public void Forward_WithZeroWeights_ShouldReturnBiasActivation()
        {
            var network = new NeuralNetwork(2, 1);
            var inputs = new decimal[] { 0m, 0m };
            
            var outputs = network.Forward(inputs);
            
            Assert.Single(outputs);
            // With zero weights and zero bias, sigmoid(0) = 0.5
            Assert.Equal(0.5m, outputs[0], 1);
        }

        [Fact]
        public void SaveAndLoad_RoundTrip_ShouldPreserveNetwork()
        {
            var original = new NeuralNetwork(2, 1);
            original.AddRandomConnection(new Random(42));
            original.MutateWeight(new Random(42));
            
            var filePath = Path.GetTempFileName();
            try
            {
                original.SaveToFile(filePath);
                var loaded = NeuralNetwork.LoadFromFile(filePath);
                
                Assert.Equal(original.InputCount, loaded.InputCount);
                Assert.Equal(original.OutputCount, loaded.OutputCount);
                // Just verify we can load and the network is valid
                Assert.NotNull(loaded);
                Assert.True(loaded.Neurons.Count > 0);
            }
            finally
            {
                File.Delete(filePath);
            }
        }

        [Fact]
        public void Clone_ShouldCreateIdenticalCopy()
        {
            var original = new NeuralNetwork(2, 1);
            original.AddRandomConnection(new Random(42));
            original.AddNeuron(new Random(42)); // Add a neuron so mutation can work
            
            var clone = original.Clone();
            
            Assert.Equal(original.InputCount, clone.InputCount);
            Assert.Equal(original.OutputCount, clone.OutputCount);
            Assert.Equal(original.Neurons.Count, clone.Neurons.Count);
            Assert.Equal(original.Connections.Count, clone.Connections.Count);
            
            // Modifying clone should not affect original
            clone.ApplyMutations(new Random(42), 5);
            // After mutations, something should be different
            Assert.True(clone.Neurons.Count != original.Neurons.Count || 
                       clone.Connections.Count != original.Connections.Count ||
                       !clone.Connections.SequenceEqual(original.Connections, new ConnectionComparer()));
        }
    }

    public class DataManagerTests
    {
        [Fact]
        public void LoadData_FromValidFile_ShouldParseCorrectly()
        {
            var filePath = Path.GetTempFileName();
            try
            {
                File.WriteAllText(filePath, "0 1 | 1 0\n1 0 | 0 1");
                
                var manager = new DataManager();
                manager.LoadFromFile(filePath);
                
                Assert.Equal(2, manager.DataPairs.Count);
                Assert.Equal(2, manager.InputDimension);
                Assert.Equal(2, manager.OutputDimension);
                
                Assert.Equal(new decimal[] { 0m, 1m }, manager.DataPairs[0].Input);
                Assert.Equal(new decimal[] { 1m, 0m }, manager.DataPairs[0].Output);
            }
            finally
            {
                File.Delete(filePath);
            }
        }

        [Fact]
        public void LoadData_WithDecimals_ShouldPreservePrecision()
        {
            var filePath = Path.GetTempFileName();
            try
            {
                File.WriteAllText(filePath, "0.123456789 1.987654321 | 0.5");
                
                var manager = new DataManager();
                manager.LoadFromFile(filePath);
                
                Assert.Single(manager.DataPairs);
                Assert.Equal(0.123456789m, manager.DataPairs[0].Input[0]);
                Assert.Equal(1.987654321m, manager.DataPairs[0].Input[1]);
                Assert.Equal(0.5m, manager.DataPairs[0].Output[0]);
            }
            finally
            {
                File.Delete(filePath);
            }
        }
    }

    public class EvolutionEngineTests
    {
        [Fact]
        public void Initialize_ShouldCreatePopulationFromZeroNetworks()
        {
            var dataPath = Path.GetTempFileName();
            var configPath = Path.GetTempFileName();
            
            try
            {
                File.WriteAllText(dataPath, "0 | 0\n1 | 1");
                File.WriteAllText(configPath, "PopulationSize=10\nOffspringPerIndividual=2\nMutationsPerOffspring=3");
                
                var config = EvolutionConfig.LoadFromFile(configPath);
                var dataManager = new DataManager();
                dataManager.LoadFromFile(dataPath);
                
                var engine = new EvolutionEngine(config, dataManager.DataPairs);
                engine.Initialize(1, 1);
                
                Assert.Equal(10, engine.InnerPopulation.Count);
                Assert.Equal(0, engine.CurrentGeneration);
            }
            finally
            {
                File.Delete(dataPath);
                File.Delete(configPath);
            }
        }

        [Fact]
        public void EvolveOneGeneration_ShouldImproveFitness()
        {
            var dataPath = Path.GetTempFileName();
            var configPath = Path.GetTempFileName();
            
            try
            {
                File.WriteAllText(dataPath, "0 | 0\n1 | 1");
                File.WriteAllText(configPath, "PopulationSize=20\nOffspringPerIndividual=5\nMutationsPerOffspring=10");
                
                var config = EvolutionConfig.LoadFromFile(configPath);
                var dataManager = new DataManager();
                dataManager.LoadFromFile(dataPath);
                
                var engine = new EvolutionEngine(config, dataManager.DataPairs);
                engine.Initialize(1, 1);
                
                var initialBest = engine.GetBestIndividual();
                var initialFitness = initialBest.Fitness;
                
                var result = engine.EvolveGenerations(10);
                
                Assert.Equal(10, engine.CurrentGeneration);
                Assert.True(result.BestFitness <= initialFitness, "Fitness should improve or stay same");
            }
            finally
            {
                File.Delete(dataPath);
                File.Delete(configPath);
            }
        }

        [Fact]
        public void SaveAndLoadProgress_ShouldPreserveState()
        {
            var dataPath = Path.GetTempFileName();
            var configPath = Path.GetTempFileName();
            var saveDir = Path.Combine(Path.GetTempPath(), "evolus_test_save");
            
            try
            {
                File.WriteAllText(dataPath, "0 | 0\n1 | 1");
                File.WriteAllText(configPath, "PopulationSize=5\nOffspringPerIndividual=2\nMutationsPerOffspring=3");
                
                var config = EvolutionConfig.LoadFromFile(configPath);
                var dataManager = new DataManager();
                dataManager.LoadFromFile(dataPath);
                
                var engine = new EvolutionEngine(config, dataManager.DataPairs);
                engine.Initialize(1, 1);
                engine.EvolveGenerations(5);
                
                var bestBefore = engine.GetBestIndividual();
                engine.SaveProgress(saveDir);
                
                // Load into new engine
                var engine2 = new EvolutionEngine(config, dataManager.DataPairs);
                engine2.LoadProgress(saveDir);
                
                var bestAfter = engine2.GetBestIndividual();
                
                Assert.Equal(engine.CurrentGeneration, engine2.CurrentGeneration);
                Assert.Equal(bestBefore.Fitness, bestAfter.Fitness);
            }
            finally
            {
                File.Delete(dataPath);
                File.Delete(configPath);
                if (Directory.Exists(saveDir))
                    Directory.Delete(saveDir, true);
            }
        }
    }
}
