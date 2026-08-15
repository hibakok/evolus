using System.Windows;
using evolus.Core;

namespace evolus.UI
{
    public partial class SettingsWindow : Window
    {
        public EvolutionConfig Config { get; private set; }

        public SettingsWindow(EvolutionConfig config)
        {
            InitializeComponent();
            Config = config;

            TxtPopulationSize.Text = config.PopulationSize.ToString();
            TxtOffspringPerIndividual.Text = config.OffspringPerIndividual.ToString();
            TxtMutationsPerOffspring.Text = config.MutationsPerOffspring.ToString();
            TxtMaxGenerations.Text = config.MaxGenerations.ToString();
            TxtRandomSeed.Text = config.RandomSeed.ToString();
        }

        private void BtnOk_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Config.PopulationSize = int.Parse(TxtPopulationSize.Text);
                Config.OffspringPerIndividual = int.Parse(TxtOffspringPerIndividual.Text);
                Config.MutationsPerOffspring = int.Parse(TxtMutationsPerOffspring.Text);
                Config.MaxGenerations = int.Parse(TxtMaxGenerations.Text);
                Config.RandomSeed = int.Parse(TxtRandomSeed.Text);

                DialogResult = true;
                Close();
            }
            catch
            {
                MessageBox.Show("Введите корректные числовые значения", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
