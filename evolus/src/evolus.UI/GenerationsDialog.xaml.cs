using System.Windows;

namespace evolus.UI
{
    public partial class GenerationsDialog : Window
    {
        public int Generations { get; private set; } = 100;

        public GenerationsDialog()
        {
            InitializeComponent();
            TxtGenerations.Text = "100";
        }

        private void BtnOk_Click(object sender, RoutedEventArgs e)
        {
            if (int.TryParse(TxtGenerations.Text, out int generations) && generations > 0)
            {
                Generations = generations;
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Введите корректное положительное число", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
