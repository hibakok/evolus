#!/usr/bin/env python3
"""
Evotus - Универсальная нейронная сеть с эволюционными стратегиями
Оптимизированная версия для максимальной скорости эволюции
"""

import random
import math
import json
import os
import sys
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import copy

# Условный импорт для определения нажатия клавиш в Windows
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

# Пути к файлам конфигурации
CONFIG_FILE = "config.txt"
DATA_FILE = "data.txt"
POPULATION_FILE = "population.json"

class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"
    STEP = "step"

# Предварительно вычисленные таблицы для активаций (ускорение)
_SIGMOID_TABLE_SIZE = 20000
_SIGMOID_MIN = -10.0
_SIGMOID_MAX = 10.0
_SIGMOID_STEP = (_SIGMOID_MAX - _SIGMOID_MIN) / _SIGMOID_TABLE_SIZE
_SIGMOID_TABLE = [1.0 / (1.0 + math.exp(-(_SIGMOID_MIN + i * _SIGMOID_STEP))) 
                  for i in range(_SIGMOID_TABLE_SIZE)]

def sigmoid(x: float) -> float:
    if x < _SIGMOID_MIN:
        return 0.0
    if x > _SIGMOID_MAX:
        return 1.0
    idx = int((x - _SIGMOID_MIN) / _SIGMOID_STEP)
    return _SIGMOID_TABLE[idx]

def tanh_act(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0.0, x)

def linear(x: float) -> float:
    return x

def step_act(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

ACTIVATION_FUNCTIONS = {
    ActivationFunction.SIGMOID: sigmoid,
    ActivationFunction.TANH: tanh_act,
    ActivationFunction.RELU: relu,
    ActivationFunction.LINEAR: linear,
    ActivationFunction.STEP: step_act,
}

@dataclass
class Neuron:
    id: int
    activation: ActivationFunction = ActivationFunction.SIGMOID
    bias: float = 0.0

@dataclass(slots=True)
class Connection:
    from_neuron: int
    to_neuron: int
    weight: float

@dataclass
class Individual:
    neurons: Dict[int, Neuron] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    fitness: float = float('inf')
    complexity: int = 0
    input_size: int = 0
    output_size: int = 0
    # Кэшированные структуры для ускорения forward pass
    _conn_by_target: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict, repr=False, init=False)
    _needs_rebuild: bool = field(default=True, repr=False, init=False)
    
    def __post_init__(self):
        self.complexity = len(self.connections)
        self._needs_rebuild = True
    
    def _rebuild_cache(self):
        """Построить кэш связей по целевым нейронам для быстрого доступа"""
        if not self._needs_rebuild:
            return
        self._conn_by_target = {}
        for conn in self.connections:
            if conn.to_neuron not in self._conn_by_target:
                self._conn_by_target[conn.to_neuron] = []
            self._conn_by_target[conn.to_neuron].append((conn.from_neuron, conn.weight))
        self._needs_rebuild = False
    
    def clone(self) -> 'Individual':
        new_neurons = {k: Neuron(v.id, v.activation, v.bias) for k, v in self.neurons.items()}
        new_connections = [Connection(c.from_neuron, c.to_neuron, c.weight) for c in self.connections]
        clone = Individual(
            neurons=new_neurons,
            connections=new_connections,
            fitness=self.fitness,
            complexity=self.complexity,
            input_size=self.input_size,
            output_size=self.output_size
        )
        clone._needs_rebuild = True
        return clone
    
    def get_input_neurons(self) -> Set[int]:
        return set(range(self.input_size))
    
    def get_output_neurons(self) -> Set[int]:
        return set(range(self.input_size, self.input_size + self.output_size))
    
    def get_hidden_neurons(self) -> Set[int]:
        all_neurons = set(self.neurons.keys())
        return all_neurons - self.get_input_neurons() - self.get_output_neurons()
    
    def forward(self, inputs: List[float]) -> List[float]:
        # Быстрый forward pass с использованием кэша связей
        return self.forward_fast(inputs)
    
    def forward_fast(self, inputs: List[float]) -> List[float]:
        """Оптимизированный forward pass с использованием кэша связей"""
        self._rebuild_cache()
        
        # Инициализировать значения нейронов
        neuron_values: Dict[int, float] = {}
        
        # Установить входные нейроны
        for i, val in enumerate(inputs):
            neuron_values[i] = val
        
        # Установить нейрон смещения в 1.0 если он существует
        bias_id = self.input_size + self.output_size
        if bias_id in self.neurons:
            neuron_values[bias_id] = 1.0
        
        # Получить списки нейронов
        hidden = list(self.get_hidden_neurons())
        output = list(self.get_output_neurons())
        
        # Удалить смещение из скрытых (оно уже установлено)
        hidden = [h for h in hidden if h != bias_id]
        
        # Оптимизированный forward pass с кэшем
        max_iterations = 10
        for _ in range(max_iterations):
            changed = False
            
            # Обработать все не-входные нейроны
            for neuron_id in hidden + output:
                if neuron_id not in self.neurons:
                    continue
                
                neuron = self.neurons[neuron_id]
                act_func = ACTIVATION_FUNCTIONS[neuron.activation]
                
                # Использовать кэш для быстрого доступа к связям
                total = neuron.bias  # Добавляем смещение нейрона
                if neuron_id in self._conn_by_target:
                    for from_neuron, weight in self._conn_by_target[neuron_id]:
                        if from_neuron in neuron_values:
                            total += weight * neuron_values[from_neuron]
                
                new_value = act_func(total)
                
                if neuron_id not in neuron_values or abs(neuron_values[neuron_id] - new_value) > 1e-15:
                    neuron_values[neuron_id] = new_value
                    changed = True
            
            if not changed:
                break
        
        # Извлечь выходы
        outputs = []
        for i in range(self.output_size):
            out_neuron_id = self.input_size + i
            if out_neuron_id in neuron_values:
                outputs.append(neuron_values[out_neuron_id])
            else:
                outputs.append(0.0)
        
        return outputs
    
    def save_to_dict(self) -> dict:
        return {
            'neurons': {str(k): {'id': v.id, 'activation': v.activation.value, 'bias': v.bias} 
                       for k, v in self.neurons.items()},
            'connections': [{'from': c.from_neuron, 'to': c.to_neuron, 'weight': c.weight} 
                           for c in self.connections],
            'fitness': self.fitness,
            'complexity': self.complexity,
            'input_size': self.input_size,
            'output_size': self.output_size
        }
    
    @classmethod
    def load_from_dict(cls, data: dict) -> 'Individual':
        individual = cls()
        individual.neurons = {int(k): Neuron(v['id'], ActivationFunction(v['activation']), v.get('bias', 0.0)) 
                             for k, v in data['neurons'].items()}
        individual.connections = [Connection(c['from'], c['to'], c['weight']) 
                                 for c in data['connections']]
        individual.fitness = data.get('fitness', float('inf'))
        individual.complexity = data.get('complexity', len(individual.connections))
        individual.input_size = data.get('input_size', 0)
        individual.output_size = data.get('output_size', 0)
        return individual


class DataManager:
    def __init__(self, filename: str = DATA_FILE):
        self.filename = filename
        self.data: List[Tuple[List[float], List[float]]] = []
        self.input_size = 0
        self.output_size = 0
    
    def load(self) -> bool:
        if not os.path.exists(self.filename):
            return False
        
        self.data = []
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) != 2:
                    continue
                
                input_part = parts[0].strip().split()
                output_part = parts[1].strip().split()
                
                inputs = [float(x) for x in input_part]
                outputs = [float(x) for x in output_part]
                
                if self.data:
                    if len(inputs) != self.input_size or len(outputs) != self.output_size:
                        print(f"Предупреждение: Несоответствие размерностей данных в строке: {line}")
                        continue
                else:
                    self.input_size = len(inputs)
                    self.output_size = len(outputs)
                
                self.data.append((inputs, outputs))
        
        return len(self.data) > 0
    
    def save_example(self):
        """Создаёт пример файла данных, если он не существует"""
        with open(self.filename, 'w') as f:
            f.write("# Пример задачи XOR\n")
            f.write("# Формат: вход1 вход2 ... | выход1 выход2 ...\n")
            f.write("0 0 | 0\n")
            f.write("0 1 | 1\n")
            f.write("1 0 | 1\n")
            f.write("1 1 | 0\n")


class ConfigManager:
    def __init__(self, filename: str = CONFIG_FILE):
        self.filename = filename
        self.config = {
            'population_size': 50,
            'offspring_per_individual': 4,
            'mutations_per_offspring': 5,
            'mutation_rate_weight': 0.3,
            'mutation_rate_connection': 0.2,
            'mutation_rate_neuron': 0.2,
            'mutation_rate_activation': 0.3,
            'weight_mutation_std': 0.5,
            'min_mutation_std': 0.01,
            'max_mutation_std': 2.0,
            'initial_hidden_neurons': 4,
        }
        self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Пропустить пустые строки и комментарии
                    if not line or line.startswith('#'):
                        continue
                    
                    # Разделить строку на ключ и значение по '='
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Преобразовать значение в соответствующий тип
                        if key in self.config:
                            current_value = self.config[key]
                            if isinstance(current_value, int):
                                self.config[key] = int(value)
                            elif isinstance(current_value, float):
                                self.config[key] = float(value)
                            elif isinstance(current_value, bool):
                                self.config[key] = value.lower() in ('true', '1', 'yes')
                            else:
                                self.config[key] = value
    
    def save(self):
        # Сохранение в текстовый файл с комментариями не требуется,
        # так как пользователь редактирует config.txt напрямую
        pass
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        # Не сохраняем автоматически, пользователь редактирует config.txt вручную


class Mutator:
    def __init__(self, config: ConfigManager):
        self.config = config
        # Кэшируем списки для ускорения
        self._activation_list = list(ActivationFunction)
    
    def mutate(self, individual: Individual, num_mutations: int) -> Individual:
        """
        Применяет мутации к индивиду.
        Мутации применяются как последовательность минимальных изменений:
        - добавление/удаление нейрона
        - добавление/удаление связи между нейронами
        - изменение веса связи нейронов
        - изменение биаса нейрона
        
        Масштаб мутаций определяется количеством одновременных случайных мелких мутаций (num_mutations).
        """
        mutant = individual.clone()
        
        for _ in range(num_mutations):
            # Выбираем случайный тип минимальной мутации с равной вероятностью
            mutation_choice = random.randint(0, 5)
            
            if mutation_choice == 0:
                # Изменение веса связи
                self._mutate_weight_fast(mutant)
            elif mutation_choice == 1:
                # Добавление связи
                self._add_connection_fast(mutant)
            elif mutation_choice == 2:
                # Удаление связи
                self._remove_connection_fast(mutant)
            elif mutation_choice == 3:
                # Добавление нейрона
                self._add_neuron_fast(mutant)
            elif mutation_choice == 4:
                # Удаление нейрона
                self._remove_neuron_fast(mutant)
            elif mutation_choice == 5:
                # Изменение биаса нейрона
                self._mutate_bias_fast(mutant)
        
        mutant.complexity = len(mutant.connections)
        mutant._needs_rebuild = True  # Пометить для перестройки кэша
        return mutant
    
    def _mutate_weight_fast(self, individual: Individual):
        """Изменяет вес случайной связи (минимальная мутация)"""
        if not individual.connections:
            return
        conn = random.choice(individual.connections)
        # Небольшое изменение веса
        conn.weight += random.gauss(0, self.config.get('weight_mutation_std'))
    
    def _mutate_bias_fast(self, individual: Individual):
        """Изменяет биас случайного нейрона (минимальная мутация)"""
        if not individual.neurons:
            return
        
        neuron_id = random.choice(list(individual.neurons.keys()))
        neuron = individual.neurons[neuron_id]
        # Небольшое изменение биаса
        neuron.bias += random.gauss(0, self.config.get('weight_mutation_std', 0.5))
    
    def _add_connection_fast(self, individual: Individual):
        if len(individual.neurons) < 2:
            return
        
        neurons = list(individual.neurons.keys())
        input_neurons = list(range(individual.input_size))
        output_neurons = list(range(individual.input_size, individual.input_size + individual.output_size))
        hidden_neurons = [n for n in neurons if n not in input_neurons and n not in output_neurons]
        
        # Быстрое создание множества для проверки существующих связей
        existing = {(c.from_neuron, c.to_neuron) for c in individual.connections}
        
        max_attempts = 50
        for _ in range(max_attempts):
            r = random.random()
            
            if r < 0.4 and hidden_neurons:
                from_neuron = random.choice(input_neurons)
                to_neuron = random.choice(hidden_neurons)
            elif r < 0.8 and hidden_neurons:
                from_neuron = random.choice(hidden_neurons)
                to_neuron = random.choice(output_neurons)
            elif r < 0.9 and hidden_neurons and len(hidden_neurons) > 1:
                h1, h2 = random.sample(hidden_neurons, 2)
                from_neuron, to_neuron = h1, h2
            else:
                from_neuron = random.choice(neurons)
                to_neuron = random.choice(neurons)
                if from_neuron == to_neuron:
                    continue
            
            if (from_neuron, to_neuron) not in existing:
                individual.connections.append(Connection(from_neuron, to_neuron, random.gauss(0, 1.0)))
                return
    
    def _remove_connection_fast(self, individual: Individual):
        if len(individual.connections) <= 5:
            return
        
        input_set = set(range(individual.input_size))
        output_set = set(range(individual.input_size, individual.input_size + individual.output_size))
        hidden_set = set(n for n in individual.neurons if n not in input_set and n not in output_set)
        
        removable = [c for c in individual.connections 
                     if (c.from_neuron in hidden_set and c.to_neuron in hidden_set) or
                        (c.from_neuron in input_set and c.to_neuron in output_set)]
        
        if removable:
            individual.connections.remove(random.choice(removable))
        else:
            individual.connections.pop(random.randrange(len(individual.connections)))
    
    def _add_neuron_fast(self, individual: Individual):
        """Добавляет новый нейрон (минимальная мутация)"""
        existing_ids = set(individual.neurons.keys())
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        input_range = set(range(individual.input_size))
        output_range = set(range(individual.input_size, individual.input_size + individual.output_size))
        
        while new_id in input_range or new_id in output_range:
            new_id += 1
        
        # Новый нейрон с нулевым биасом
        individual.neurons[new_id] = Neuron(new_id, random.choice(self._activation_list), bias=0.0)
    
    def _remove_neuron_fast(self, individual: Individual):
        """Удаляет случайный скрытый нейрон (минимальная мутация)"""
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        neuron_id = random.choice(list(hidden))
        individual.connections = [c for c in individual.connections 
                                   if c.from_neuron != neuron_id and c.to_neuron != neuron_id]
        del individual.neurons[neuron_id]
    
    def _mutate_activation_fast(self, individual: Individual):
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        neuron_id = random.choice(list(hidden))
        current = individual.neurons[neuron_id].activation
        new_activations = [a for a in self._activation_list if a != current]
        individual.neurons[neuron_id].activation = random.choice(new_activations)


class FitnessCalculator:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        # Кэшируем данные для быстрого доступа
        self._cached_data = None
    
    def _prepare_data(self):
        """Подготовить кэшированные данные для быстрого вычисления fitness"""
        if self._cached_data is not None:
            return self._cached_data
        
        self._cached_data = []
        for inputs, expected_outputs in self.data_manager.data:
            self._cached_data.append((inputs, expected_outputs))
        return self._cached_data
    
    def calculate(self, individual: Individual) -> Tuple[float, int]:
        """
        Вычисляет приспособленность как сумму квадратов ошибок.
        Возвращает кортеж (ошибка, сложность).
        Меньшая ошибка лучше.
        Если ошибки равны, меньшая сложность лучше.
        НО: если одна имеет большую сложность, но даже немного меньшую ошибку, она лучше.
        """
        if not self.data_manager.data:
            return float('inf'), individual.complexity
        
        # Использовать кэшированные данные
        data = self._cached_data if self._cached_data else self._prepare_data()
        
        total_error = 0.0
        
        # Быстрый forward pass с кэшем
        individual._rebuild_cache()
        
        for inputs, expected_outputs in data:
            actual_outputs = individual.forward_fast(inputs)
            
            for actual, expected in zip(actual_outputs, expected_outputs):
                diff = actual - expected
                total_error += diff * diff
        
        return total_error, individual.complexity
    
    def calculate_batch(self, individuals: List[Individual]) -> List[Tuple[float, int]]:
        """Пакетное вычисление fitness для нескольких особей"""
        results = []
        data = self._cached_data if self._cached_data else self._prepare_data()
        
        for ind in individuals:
            ind._rebuild_cache()
            total_error = 0.0
            
            for inputs, expected_outputs in data:
                actual_outputs = ind.forward_fast(inputs)
                for actual, expected in zip(actual_outputs, expected_outputs):
                    diff = actual - expected
                    total_error += diff * diff
            
            results.append((total_error, ind.complexity))
        
        return results
    
    def compare_fitness(self, ind1: Individual, ind2: Individual) -> int:
        """
        Сравнивает две особи.
        Возвращает:
          -1 если ind1 лучше
           1 если ind2 лучше
           0 если равны
        
        Правила:
        - Меньшая ошибка всегда лучше
        - Если ошибки равны, меньшая сложность лучше
        - Большая сложность с даже немного меньшей ошибкой лучше
        """
        err1, comp1 = ind1.fitness, ind1.complexity
        err2, comp2 = ind2.fitness, ind2.complexity
        
        # Use high precision comparison
        if err1 < err2 - 1e-15:
            return -1
        elif err1 > err2 + 1e-15:
            return 1
        else:
            # Errors are essentially equal
            if comp1 < comp2:
                return -1
            elif comp1 > comp2:
                return 1
            else:
                return 0


class PopulationManager:
    def __init__(self, config: ConfigManager, data_manager: DataManager, 
                 mutator: Mutator, fitness_calc: FitnessCalculator):
        self.config = config
        self.data_manager = data_manager
        self.mutator = mutator
        self.fitness_calc = fitness_calc
        self.internal_population: List[Individual] = []  # Внутренняя популяция прародителей (неизменна в рамках поколения)
        self.generation = 0
    
    def initialize(self):
        """Создаёт начальную популяцию нейронных сетей с некоторыми случайными связями для быстрого старта"""
        self.internal_population = []
        
        for _ in range(self.config.get('population_size')):
            individual = Individual()
            individual.input_size = self.data_manager.input_size
            individual.output_size = self.data_manager.output_size
            
            # Create input neurons
            for i in range(individual.input_size):
                individual.neurons[i] = Neuron(i, ActivationFunction.LINEAR, bias=0.0)
            
            # Create output neurons
            for i in range(individual.output_size):
                neuron_id = individual.input_size + i
                individual.neurons[neuron_id] = Neuron(neuron_id, ActivationFunction.SIGMOID, bias=0.0)
            
            # Add bias neuron (constant value of 1)
            bias_id = individual.input_size + individual.output_size
            individual.neurons[bias_id] = Neuron(bias_id, ActivationFunction.LINEAR, bias=0.0)
            
            # Add initial hidden neurons for better starting point
            num_hidden = self.config.get('initial_hidden_neurons', 4)
            hidden_ids = []
            for i in range(num_hidden):
                hid_id = bias_id + 1 + i
                individual.neurons[hid_id] = Neuron(hid_id, ActivationFunction.SIGMOID, bias=0.0)
                hidden_ids.append(hid_id)
            
            # Connect inputs to hidden neurons
            for inp_id in range(individual.input_size):
                for hid_id in hidden_ids:
                    weight = random.gauss(0, 0.5)
                    individual.connections.append(Connection(inp_id, hid_id, weight))
            
            # Connect hidden neurons to outputs
            for hid_id in hidden_ids:
                for out_id in range(individual.input_size, individual.input_size + individual.output_size):
                    weight = random.gauss(0, 0.5)
                    individual.connections.append(Connection(hid_id, out_id, weight))
            
            # Connect bias to hidden and outputs
            for hid_id in hidden_ids:
                weight = random.gauss(0, 0.5)
                individual.connections.append(Connection(bias_id, hid_id, weight))
            for out_id in range(individual.input_size, individual.input_size + individual.output_size):
                weight = random.gauss(0, 0.5)
                individual.connections.append(Connection(bias_id, out_id, weight))
            
            # Add some random recurrent connections between hidden neurons
            for i, h1 in enumerate(hidden_ids):
                for h2 in hidden_ids[i+1:]:
                    if random.random() < 0.3:
                        weight = random.gauss(0, 0.5)
                        if random.random() < 0.5:
                            individual.connections.append(Connection(h1, h2, weight))
                        else:
                            individual.connections.append(Connection(h2, h1, weight))
            
            individual.fitness = float('inf')
            individual.complexity = len(individual.connections)
            
            self.internal_population.append(individual)
        
        self.generation = 0
    
    def evolve_generation(self, num_generations: int, print_interval: int = 1) -> List[str]:
        """
        Запускает эволюцию на указанное количество поколений.
        
        Принцип работы:
        1. Внутренняя популяция прародителей неизменна в начале каждого поколения
        2. Каждый прародитель порождает N потомков (мутантов) во внешнюю популяцию
        3. Потомки оцениваются по всем парам вход-выход
        4. Лучший потомок сравнивается со своим родителем
        5. Потомок заменяет родителя ТОЛЬКО если он строго лучше:
           - Меньшая ошибка всегда лучше
           - При РАВНОЙ ошибке (полностью равной) выбирается особь с меньшей сложностью
           - Если более сложная особь имеет ХОТЯ БЫ ЧУТЬ меньшую ошибку - она лучше
        6. Сравнение производится напрямую без штрафов за сложность
        """
        progress_log = []
        
        # Адаптивные параметры мутации
        current_mutation_std = self.config.get('weight_mutation_std', 0.5)
        stagnation_counter = 0
        best_ever_fitness = float('inf')
        
        for gen in range(num_generations):
            self.generation += 1
            
            # Выводим прогресс только в определенные интервалы
            should_log = (gen % print_interval == 0) or (gen == num_generations - 1)
            
            # Создаем копию внутренней популяции для замены (чтобы не менять во время оценки)
            new_internal_population = []
            
            # Для каждого прародителя создаем потомков и оцениваем их
            for parent_idx, parent in enumerate(self.internal_population):
                num_offspring = self.config.get('offspring_per_individual')
                
                # Список всех потомков этого родителя для оценки
                offspring_list = []
                
                for _ in range(num_offspring):
                    mutations = self.config.get('mutations_per_offspring')
                    
                    # Адаптировать силу мутаций на лету
                    original_std = self.config.get('weight_mutation_std')
                    self.config.set('weight_mutation_std', current_mutation_std)
                    
                    # Создаем мутантного потомка
                    offspring = self.mutator.mutate(parent, mutations)
                    
                    # Восстановить оригинальное значение (оно может быть изменено в config.txt)
                    self.config.set('weight_mutation_std', original_std)
                    
                    # Гарантируем, что у потомка есть хотя бы некоторые связи
                    if len(parent.connections) == 0 and len(offspring.connections) == 0:
                        self._force_add_connection(offspring)
                    
                    offspring_list.append(offspring)
                
                # Оцениваем всех потомков одновременно по всем парам вход-выход
                for offspring in offspring_list:
                    error, complexity = self.fitness_calc.calculate(offspring)
                    offspring.fitness = error
                    offspring.complexity = complexity
                
                # Находим лучшего потомка (с минимальной ошибкой, при равной ошибке - минимальной сложностью)
                best_offspring = min(offspring_list, key=lambda x: (x.fitness, x.complexity))
                
                # Получаем ошибку родителя если еще не вычислена
                parent_error = parent.fitness
                if parent_error == float('inf'):
                    parent_error, parent_complexity = self.fitness_calc.calculate(parent)
                    parent.fitness = parent_error
                    parent.complexity = parent_complexity
                
                # ПРЯМОЕ СРАВНЕНИЕ без штрафов за сложность:
                # 1. Если ошибка потомка < ошибки родителя - потомок лучше
                # 2. Если ошибки РАВНЫ (полностью), то особь с меньшей сложностью лучше
                # 3. Если ошибка потомка > ошибки родителя - родитель лучше
                # 4. Если более сложная особь имеет ХОТЯ БЫ ЧУТЬ меньшую ошибку - она лучше
                
                offspring_error = best_offspring.fitness
                offspring_complexity = best_offspring.complexity
                parent_complexity = parent.complexity
                
                replace_parent = False
                
                # Проверяем: ошибка потомка меньше ошибки родителя?
                if offspring_error < parent_error - 1e-15:
                    # Потомок имеет меньшую ошибку - он лучше независимо от сложности
                    replace_parent = True
                elif abs(offspring_error - parent_error) <= 1e-15:
                    # Ошибки РАВНЫ (полностью равны)
                    # Выбираем особь с меньшей сложностью
                    if offspring_complexity < parent_complexity:
                        replace_parent = True
                    else:
                        replace_parent = False
                else:
                    # Ошибка потомка больше - родитель лучше
                    replace_parent = False
                
                if replace_parent:
                    # Потомок заменяет родителя
                    new_internal_population.append(best_offspring.clone())
                else:
                    # Родитель остается
                    new_internal_population.append(parent.clone())
            
            # Обновляем внутреннюю популяцию
            self.internal_population = new_internal_population
            
            # Записываем прогресс только если нужно
            if should_log:
                best = min(self.internal_population, key=lambda x: x.fitness)
                avg_fitness = sum(ind.fitness for ind in self.internal_population) / len(self.internal_population)
                
                # Проверка на улучшение для адаптации мутаций
                if best.fitness < best_ever_fitness - 1e-12:
                    best_ever_fitness = best.fitness
                    stagnation_counter = 0
                    # Уменьшить силу мутаций для точной настройки
                    current_mutation_std = max(
                        self.config.get('min_mutation_std', 0.01),
                        current_mutation_std * 0.95
                    )
                else:
                    stagnation_counter += 1
                    # Если застой, увеличить разнообразие мутаций
                    if stagnation_counter > 20:
                        current_mutation_std = min(
                            self.config.get('max_mutation_std', 2.0),
                            current_mutation_std * 1.1
                        )
                        if stagnation_counter > 50:
                            # Сильный застой - резкое увеличение мутаций
                            current_mutation_std = min(
                                self.config.get('max_mutation_std', 2.0),
                                current_mutation_std * 1.5
                            )
                
                progress_log.append(
                    f"Поколение {self.generation}: Лучшая={best.fitness:.15f}, Средняя={avg_fitness:.15f}, "
                    f"Мутация={current_mutation_std:.4f}"
                )
        
        return progress_log
    
    def _force_add_connection(self, individual: Individual):
        """Принудительно добавляет связь между случайными нейронами, обеспечивая передачу сигнала от входа к выходу"""
        if len(individual.neurons) < 2:
            return
        
        neurons = list(individual.neurons.keys())
        input_neurons = set(range(individual.input_size))
        output_neurons = set(range(individual.input_size, individual.input_size + individual.output_size))
        
        # Prefer adding connections that help signal flow from input to output
        # Try multiple times to find a good connection
        max_attempts = 50
        for _ in range(max_attempts):
            # Bias towards connecting input/hidden -> output/hidden
            if random.random() < 0.7:
                # From input or hidden
                from_candidates = [n for n in neurons if n not in output_neurons]
                if not from_candidates:
                    from_candidates = neurons
                from_neuron = random.choice(from_candidates)
                
                # To hidden or output
                to_candidates = [n for n in neurons if n not in input_neurons and n != from_neuron]
                if not to_candidates:
                    to_candidates = [n for n in neurons if n != from_neuron]
                if not to_candidates:
                    continue
                to_neuron = random.choice(to_candidates)
            else:
                from_neuron = random.choice(neurons)
                to_neuron = random.choice(neurons)
                if from_neuron == to_neuron:
                    continue
            
            # Check if connection already exists
            exists = False
            for conn in individual.connections:
                if conn.from_neuron == from_neuron and conn.to_neuron == to_neuron:
                    exists = True
                    break
            
            if not exists:
                weight = random.gauss(0, 1.0)
                individual.connections.append(Connection(from_neuron, to_neuron, weight))
                return
        
        # If all attempts failed, just add any connection
        for from_n in neurons:
            for to_n in neurons:
                if from_n != to_n:
                    exists = any(c.from_neuron == from_n and c.to_neuron == to_n 
                                for c in individual.connections)
                    if not exists:
                        weight = random.gauss(0, 1.0)
                        individual.connections.append(Connection(from_n, to_n, weight))
                        return
    
    def get_best(self) -> Individual:
        if not self.internal_population:
            return None
        return min(self.internal_population, key=lambda x: x.fitness)
    
    def save(self, filename: str = POPULATION_FILE):
        data = {
            'generation': self.generation,
            'population': [ind.save_to_dict() for ind in self.internal_population]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str = POPULATION_FILE) -> bool:
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.generation = data.get('generation', 0)
        self.internal_population = [
            Individual.load_from_dict(ind_data) for ind_data in data['population']
        ]
        
        return True


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_key():
    """Wait for any key press"""
    if os.name == 'nt':
        try:
            msvcrt.getch()
        except Exception:
            pass
    else:
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except (termios.error, io.UnsupportedOperation):
            # Неинтерактивный режим (пайп, редирект) - просто читаем строку или пропускаем
            try:
                sys.stdin.read(1)
            except Exception:
                pass
            return
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def show_menu() -> int:
    clear_screen()
    print("=" * 60)
    print("EVOTUS - Универсальная нейронная сеть с эволюционными стратегиями")
    print("=" * 60)
    print("\nГлавное меню:")
    print("1. Начать эволюцию")
    print("2. Тестировать текущую лучшую особь")
    print("3. Просмотреть статус популяции")
    print("4. Сохранить популяцию")
    print("5. Загрузить популяцию")
    print("6. Настройки")
    print("7. Просмотреть/редактировать обучающие данные")
    print("8. Экспортировать лучшую особь в файл")
    print("9. Импортировать особь из файла")
    print("0. Выход")
    print("\n" + "=" * 60)
    
    try:
        choice = int(input("Введите выбор: "))
        return choice
    except (ValueError, EOFError):
        return -1


def run_evolution(pop_manager: PopulationManager):
    clear_screen()
    print("=" * 60)
    print("РЕЖИМ ЭВОЛЮЦИИ")
    print("=" * 60)
    
    try:
        num_gens = int(input("Введите количество поколений для эволюции: "))
    except (ValueError, EOFError):
        print("Неверный ввод!")
        wait_for_key()
        return
    
    print(f"\nЗапуск эволюции на {num_gens} поколений...")
    print("-" * 60)
    
    # Определяем интервал вывода прогресса
    if num_gens > 1000:
        print_interval = num_gens // 100  # Выводить каждые 1% прогресса
    elif num_gens > 500:
        print_interval = 10
    else:
        print_interval = 1
    
    progress = pop_manager.evolve_generation(num_gens, print_interval=print_interval)
    
    for log_entry in progress:
        print(log_entry)
        sys.stdout.flush()  # Принудительная запись в stdout
    
    print("-" * 60)
    best = pop_manager.get_best()
    print(f"\nЭволюция завершена!")
    print(f"Лучшая приспособленность: {best.fitness:.15f}")
    print(f"Сложность: {best.complexity}")
    print(f"Нейроны: {len(best.neurons)}")
    print(f"Связи: {len(best.connections)}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def test_individual(pop_manager: PopulationManager):
    best = pop_manager.get_best()
    
    if best is None or not best.neurons:
        print("\nНет обученной особи! Сначала запустите эволюцию.")
        wait_for_key()
        return
    
    clear_screen()
    print("=" * 60)
    print("РЕЖИМ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Сеть: {len(best.neurons)} нейронов, {len(best.connections)} связей")
    print(f"Размер входа: {best.input_size}, Размер выхода: {best.output_size}")
    print("-" * 60)
    print("Введите входные значения через пробел (или 'q' для выхода)")
    print()
    
    while True:
        try:
            user_input = input(f"Вход ({best.input_size} значений): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            values = [float(x) for x in user_input.split()]
            
            if len(values) != best.input_size:
                print(f"Ошибка: Ожидалось {best.input_size} значений, получено {len(values)}")
                continue
            
            outputs = best.forward(values)
            
            print(f"Выход: {' '.join(f'{o:.10f}' for o in outputs)}")
            print()
            
        except ValueError:
            print("Ошибка: Неверный формат ввода")
        except EOFError:
            break
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def view_status(pop_manager: PopulationManager):
    clear_screen()
    print("=" * 60)
    print("СТАТУС ПОПУЛЯЦИИ")
    print("=" * 60)
    print(f"Поколение: {pop_manager.generation}")
    print(f"Размер популяции: {len(pop_manager.internal_population)}")
    print()
    
    if pop_manager.internal_population:
        sorted_pop = sorted(pop_manager.internal_population, key=lambda x: x.fitness)
        print("Топ 5 особей:")
        print("-" * 60)
        for i, ind in enumerate(sorted_pop[:5]):
            print(f"{i+1}. Приспособленность: {ind.fitness:.15f}, Сложность: {ind.complexity}, "
                  f"Нейроны: {len(ind.neurons)}, Связи: {len(ind.connections)}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def configure_settings(config: ConfigManager):
    clear_screen()
    print("=" * 60)
    print("НАСТРОЙКИ")
    print("=" * 60)
    print("\nНастройки теперь хранятся в файле config.txt")
    print("Вы можете отредактировать этот файл в любом текстовом редакторе.")
    print("\nТекущие значения настроек:")
    print("-" * 40)
    
    descriptions = {
        'population_size': 'Размер внутренней популяции',
        'offspring_per_individual': 'Количество потомков на особь',
        'mutations_per_offspring': 'Количество мутаций на потомка',
        'mutation_rate_weight': 'Вероятность мутации веса',
        'mutation_rate_connection': 'Вероятность мутации связи',
        'mutation_rate_neuron': 'Вероятность мутации нейрона',
        'mutation_rate_activation': 'Вероятность мутации активации',
        'weight_mutation_std': 'Стандартное отклонение мутации веса'
    }
    
    for key, value in config.config.items():
        desc = descriptions.get(key, key)
        print(f"{desc}: {value}")
    
    print("-" * 40)
    print(f"\nДля изменения настроек откройте файл '{config.filename}'")
    print("и измените значения после знака '=' в соответствующих строках.")
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def export_individual(pop_manager: PopulationManager):
    best = pop_manager.get_best()
    
    if best is None:
        print("\nНет особи для экспорта!")
        wait_for_key()
        return
    
    filename = input("Введите имя файла для экспорта: ").strip()
    
    if not filename:
        filename = "best_individual.json"
    
    with open(filename, 'w') as f:
        json.dump(best.save_to_dict(), f, indent=2)
    
    print(f"Экспортировано в {filename}")
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def import_individual(pop_manager: PopulationManager):
    filename = input("Введите имя файла для импорта: ").strip()
    
    if not os.path.exists(filename):
        print(f"Файл не найден: {filename}")
        wait_for_key()
        return
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        individual = Individual.load_from_dict(data)
        
        # Добавить в популяцию или заменить худшую
        if pop_manager.internal_population:
            worst_idx = max(range(len(pop_manager.internal_population)), 
                          key=lambda i: pop_manager.internal_population[i].fitness)
            pop_manager.internal_population[worst_idx] = individual
        else:
            pop_manager.internal_population.append(individual)
        
        print(f"Импортирована особь из {filename}")
    except Exception as e:
        print(f"Ошибка импорта: {e}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def main():
    # Инициализация менеджеров
    config = ConfigManager()
    data_manager = DataManager()
    
    # Создать пример файла данных, если нужно
    if not os.path.exists(DATA_FILE):
        data_manager.save_example()
        print(f"Создан пример файла данных: {DATA_FILE}")
    
    # Загрузить обучающие данные
    if not data_manager.load():
        print(f"Ошибка загрузки данных из {DATA_FILE}")
        print("Убедитесь, что файл существует и содержит корректные данные.")
        return
    
    print(f"Загружено {len(data_manager.data)} обучающих примеров")
    print(f"Размер входа: {data_manager.input_size}, Размер выхода: {data_manager.output_size}")
    
    mutator = Mutator(config)
    fitness_calc = FitnessCalculator(data_manager)
    pop_manager = PopulationManager(config, data_manager, mutator, fitness_calc)
    
    # Попытаться загрузить существующую популяцию
    if pop_manager.load():
        print(f"Загружена популяция из поколения {pop_manager.generation}")
    else:
        print("Инициализация новой популяции...")
        pop_manager.initialize()
    
    # Главный цикл
    while True:
        choice = show_menu()
        
        if choice == 1:
            run_evolution(pop_manager)
        elif choice == 2:
            test_individual(pop_manager)
        elif choice == 3:
            view_status(pop_manager)
        elif choice == 4:
            pop_manager.save()
            print("Популяция сохранена!")
            wait_for_key()
        elif choice == 5:
            if pop_manager.load():
                print(f"Загружена популяция из поколения {pop_manager.generation}")
            else:
                print("Сохранённая популяция не найдена!")
            wait_for_key()
        elif choice == 6:
            configure_settings(config)
        elif choice == 7:
            print(f"\nОткрытие {DATA_FILE} для редактирования...")
            print("Отредактируйте файл и нажмите любую клавишу когда закончите...")
            wait_for_key()
            # Перезагрузить данные после редактирования
            if data_manager.load():
                print(f"Перезагружено {len(data_manager.data)} обучающих примеров")
                # Реинициализировать популяцию с новыми размерами данных
                pop_manager.initialize()
            else:
                print("Ошибка перезагрузки данных!")
            wait_for_key()
        elif choice == 8:
            export_individual(pop_manager)
        elif choice == 9:
            import_individual(pop_manager)
        elif choice == 0:
            print("Сохранение популяции перед выходом...")
            pop_manager.save()
            print("До свидания!")
            break
        else:
            print("Неверный выбор!")
            wait_for_key()


if __name__ == "__main__":
    main()
