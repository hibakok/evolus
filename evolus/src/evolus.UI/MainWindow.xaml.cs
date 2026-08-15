using System;
using System.Windows;
using Microsoft.Win32;
using evolus.Core;

namespace evolus.UI
{
    public partial class MainWindow : Window
    {
        private DataManager _dataManager;
        private EvolutionEngine _evolutionEngine;
        private EvolutionConfig _config;
        private string _saveDirectory = "evolus_save";

        public MainWindow()
        {
            InitializeComponent();
            InitializeApplication();
        }

        private void InitializeApplication()
        {
            try
            {
                // Загружаем конфигурацию
                var configPath = "evolus_config.txt";
                _config = EvolutionConfig.LoadFromFile(configPath);

                // Пытаемся загрузить сохраненный прогресс
                if (System.IO.Directory.Exists(_saveDirectory))
                {
                    _dataManager = new DataManager();
                    var dataPath = "training_data.txt";
                    
                    if (System.IO.File.Exists(dataPath))
                    {
                        _dataManager.LoadFromFile(dataPath);
                        
                        _evolutionEngine = new EvolutionEngine(_config, _dataManager.DataPairs);
                        _evolutionEngine.LoadProgress(_saveDirectory);
                        
                        StatusText.Text = $"Прогресс загружен. Поколение: {_evolutionEngine.CurrentGeneration}";
                    }
                }

                StatusText.Text = "Готов. Загрузите данные для начала работы.";
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Ошибка инициализации: {ex.Message}";
            }
        }

        private void BtnEvolve_Click(object sender, RoutedEventArgs e)
        {
            if (_dataManager == null || _dataManager.DataPairs.Count == 0)
            {
                MessageBox.Show("Сначала загрузите данные обучения!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var dialog = new GenerationsDialog();
            if (dialog.ShowDialog() == true)
            {
                int generations = dialog.Generations;
                
                try
                {
                    StatusText.Text = "Эволюция началась...";
                    
                    // Если эволюция еще не инициализирована, создаем новую
                    if (_evolutionEngine == null)
                    {
                        _evolutionEngine = new EvolutionEngine(_config, _dataManager.DataPairs);
                        _evolutionEngine.Initialize(_dataManager.InputDimension, _dataManager.OutputDimension);
                    }

                    var result = _evolutionEngine.EvolveGenerations(generations);
                    
                    // Сохраняем прогресс
                    _evolutionEngine.SaveProgress(_saveDirectory);

                    var report = $"Эволюция завершена!\n\n{result}\n\nНажмите OK для возврата в меню.";
                    MessageBox.Show(report, "Отчет об эволюции", MessageBoxButton.OK, MessageBoxImage.Information);
                    
                    StatusText.Text = $"Эволюция завершена. Поколение: {_evolutionEngine.CurrentGeneration}";
                }
                catch (Exception ex)
                {
                    StatusText.Text = $"Ошибка эволюции: {ex.Message}";
                    MessageBox.Show($"Ошибка: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void BtnTest_Click(object sender, RoutedEventArgs e)
        {
            if (_evolutionEngine == null)
            {
                MessageBox.Show("Сначала проведите эволюцию или загрузите сохранение!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var bestIndividual = _evolutionEngine.GetBestIndividual();
            var testWindow = new TestNetworkWindow(bestIndividual.Network);
            testWindow.ShowDialog();
        }

        private void BtnLoadData_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*",
                Title = "Выберите файл с данными обучения"
            };

            if (dialog.ShowDialog() == true)
            {
                try
                {
                    _dataManager = new DataManager();
                    _dataManager.LoadFromFile(dialog.FileName);
                    
                    // Сбрасываем эволюцию при загрузке новых данных
                    _evolutionEngine = null;
                    
                    StatusText.Text = $"Данные загружены. Пар: {_dataManager.DataPairs.Count}, Входов: {_dataManager.InputDimension}, Выходов: {_dataManager.OutputDimension}";
                    MessageBox.Show($"Данные успешно загружены!\nПар вход-выход: {_dataManager.DataPairs.Count}\nРазмерность входа: {_dataManager.InputDimension}\nРазмерность выхода: {_dataManager.OutputDimension}", "Успех", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    StatusText.Text = $"Ошибка загрузки данных: {ex.Message}";
                    MessageBox.Show($"Ошибка загрузки данных: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void BtnSaveNetwork_Click(object sender, RoutedEventArgs e)
        {
            if (_evolutionEngine == null)
            {
                MessageBox.Show("Нет сети для сохранения!", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var dialog = new SaveFileDialog
            {
                Filter = "Neural Network files (*.net)|*.net|All files (*.*)|*.*",
                Title = "Сохранить лучшую нейросеть"
            };

            if (dialog.ShowDialog() == true)
            {
                try
                {
                    var bestIndividual = _evolutionEngine.GetBestIndividual();
                    bestIndividual.Network.SaveToFile(dialog.FileName);
                    StatusText.Text = $"Сеть сохранена в {dialog.FileName}";
                    MessageBox.Show("Сеть успешно сохранена!", "Успех", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    StatusText.Text = $"Ошибка сохранения: {ex.Message}";
                    MessageBox.Show($"Ошибка сохранения: {ex.Message}", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void BtnSettings_Click(object sender, RoutedEventArgs e)
        {
            var settingsWindow = new SettingsWindow(_config);
            if (settingsWindow.ShowDialog() == true)
            {
                _config = settingsWindow.Config;
                _config.SaveToFile("evolus_config.txt");
                StatusText.Text = "Настройки сохранены";
            }
        }
    }
}
