using System;
using System.Linq;
using System.Windows;
using evolus.Core;

namespace evolus.UI
{
    public partial class TestNetworkWindow : Window
    {
        private readonly NeuralNetwork _network;

        public TestNetworkWindow(NeuralNetwork network)
        {
            InitializeComponent();
            _network = network;
            TxtInput.Text = string.Join(" ", Enumerable.Repeat("0", network.InputCount));
        }

        private void BtnExecute_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var inputValues = TxtInput.Text
                    .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(decimal.Parse)
                    .ToArray();

                if (inputValues.Length != _network.InputCount)
                {
                    MessageBox.Show($"Ожидается {_network.InputCount} входных значений, введено {inputValues.Length}", 
                        "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                var outputs = _network.Forward(inputValues);
                TxtOutput.Text = string.Join(" ", outputs);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void BtnClose_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
