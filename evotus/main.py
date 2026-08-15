#!/usr/bin/env python3
"""
Evotus - Universal Neural Network with Evolutionary Strategies
"""

import random
import math
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import copy

# Conditional import for Windows key press detection
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

# Configuration file paths
CONFIG_FILE = "config.json"
DATA_FILE = "data.txt"
POPULATION_FILE = "population.json"

class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"
    STEP = "step"

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

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

@dataclass
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
    
    def __post_init__(self):
        self.complexity = len(self.connections)
    
    def clone(self) -> 'Individual':
        new_neurons = {k: Neuron(v.id, v.activation) for k, v in self.neurons.items()}
        new_connections = [Connection(c.from_neuron, c.to_neuron, c.weight) for c in self.connections]
        return Individual(
            neurons=new_neurons,
            connections=new_connections,
            fitness=self.fitness,
            complexity=self.complexity,
            input_size=self.input_size,
            output_size=self.output_size
        )
    
    def get_input_neurons(self) -> Set[int]:
        return set(range(self.input_size))
    
    def get_output_neurons(self) -> Set[int]:
        return set(range(self.input_size, self.input_size + self.output_size))
    
    def get_hidden_neurons(self) -> Set[int]:
        all_neurons = set(self.neurons.keys())
        return all_neurons - self.get_input_neurons() - self.get_output_neurons()
    
    def forward(self, inputs: List[float]) -> List[float]:
        # Initialize neuron values
        neuron_values: Dict[int, float] = {}
        
        # Set input neurons
        for i, val in enumerate(inputs):
            neuron_values[i] = val
        
        # Set bias neuron to 1.0 if it exists
        bias_id = self.input_size + self.output_size
        if bias_id in self.neurons:
            neuron_values[bias_id] = 1.0
        
        # Get topological order for hidden and output neurons
        hidden = list(self.get_hidden_neurons())
        output = list(self.get_output_neurons())
        
        # Remove bias from hidden (it's already set)
        hidden = [h for h in hidden if h != bias_id]
        
        # Simple iteration (may need multiple passes for recurrent networks)
        max_iterations = 10
        for _ in range(max_iterations):
            changed = False
            # Process all non-input neurons
            for neuron_id in hidden + output:
                if neuron_id not in self.neurons:
                    continue
                
                neuron = self.neurons[neuron_id]
                act_func = ACTIVATION_FUNCTIONS[neuron.activation]
                
                # Sum weighted inputs
                total = 0.0
                for conn in self.connections:
                    if conn.to_neuron == neuron_id:
                        if conn.from_neuron in neuron_values:
                            total += conn.weight * neuron_values[conn.from_neuron]
                
                new_value = act_func(total)
                
                if neuron_id not in neuron_values or abs(neuron_values[neuron_id] - new_value) > 1e-15:
                    neuron_values[neuron_id] = new_value
                    changed = True
            
            if not changed:
                break
        
        # Extract outputs
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
            'neurons': {str(k): {'id': v.id, 'activation': v.activation.value} 
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
        individual.neurons = {int(k): Neuron(v['id'], ActivationFunction(v['activation'])) 
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
                        print(f"Warning: Inconsistent data dimensions in line: {line}")
                        continue
                else:
                    self.input_size = len(inputs)
                    self.output_size = len(outputs)
                
                self.data.append((inputs, outputs))
        
        return len(self.data) > 0
    
    def save_example(self):
        """Create an example data file if it doesn't exist"""
        with open(self.filename, 'w') as f:
            f.write("# Example XOR problem\n")
            f.write("# Format: input1 input2 ... | output1 output2 ...\n")
            f.write("0 0 | 0\n")
            f.write("0 1 | 1\n")
            f.write("1 0 | 1\n")
            f.write("1 1 | 0\n")


class ConfigManager:
    def __init__(self, filename: str = CONFIG_FILE):
        self.filename = filename
        self.config = {
            'population_size': 20,
            'offspring_per_individual': 3,
            'mutations_per_offspring': 5,
            'mutation_rate_weight': 0.3,
            'mutation_rate_connection': 0.2,
            'mutation_rate_neuron': 0.2,
            'mutation_rate_activation': 0.3,
            'weight_mutation_std': 0.5,
        }
        self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                saved_config = json.load(f)
                self.config.update(saved_config)
    
    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        self.save()


class Mutator:
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def mutate(self, individual: Individual, num_mutations: int) -> Individual:
        mutant = individual.clone()
        
        for _ in range(num_mutations):
            mutation_type = random.random()
            
            if mutation_type < self.config.get('mutation_rate_weight'):
                self._mutate_weight(mutant)
            elif mutation_type < self.config.get('mutation_rate_weight') + self.config.get('mutation_rate_connection'):
                if random.random() < 0.5:
                    self._add_connection(mutant)
                else:
                    self._remove_connection(mutant)
            elif mutation_type < self.config.get('mutation_rate_weight') + self.config.get('mutation_rate_connection') + self.config.get('mutation_rate_neuron'):
                if random.random() < 0.5:
                    self._add_neuron(mutant)
                else:
                    self._remove_neuron(mutant)
            else:
                self._mutate_activation(mutant)
        
        mutant.complexity = len(mutant.connections)
        return mutant
    
    def _mutate_weight(self, individual: Individual):
        if not individual.connections:
            return
        
        conn = random.choice(individual.connections)
        delta = random.gauss(0, self.config.get('weight_mutation_std'))
        conn.weight += delta
    
    def _add_connection(self, individual: Individual):
        if len(individual.neurons) < 2:
            return
        
        neurons = list(individual.neurons.keys())
        
        # Try to add a connection from input/hidden to output/hidden
        # This ensures signal can flow
        max_attempts = 50
        for _ in range(max_attempts):
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
    
    def _remove_connection(self, individual: Individual):
        if individual.connections:
            individual.connections.pop(random.randrange(len(individual.connections)))
    
    def _add_neuron(self, individual: Individual):
        # Find a free neuron ID
        existing_ids = set(individual.neurons.keys())
        new_id = 0
        while new_id in existing_ids:
            new_id += 1
        
        # Don't add neurons in input/output range
        input_range = set(range(individual.input_size))
        output_range = set(range(individual.input_size, individual.input_size + individual.output_size))
        
        if new_id in input_range or new_id in output_range:
            new_id = max(existing_ids) + 1
            while new_id in input_range or new_id in output_range:
                new_id += 1
        
        activation = random.choice(list(ActivationFunction))
        individual.neurons[new_id] = Neuron(new_id, activation)
    
    def _remove_neuron(self, individual: Individual):
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        neuron_id = random.choice(list(hidden))
        
        # Remove all connections involving this neuron
        individual.connections = [
            c for c in individual.connections 
            if c.from_neuron != neuron_id and c.to_neuron != neuron_id
        ]
        
        del individual.neurons[neuron_id]
    
    def _mutate_activation(self, individual: Individual):
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        neuron_id = random.choice(list(hidden))
        activations = list(ActivationFunction)
        current = individual.neurons[neuron_id].activation
        new_activations = [a for a in activations if a != current]
        individual.neurons[neuron_id].activation = random.choice(new_activations)


class FitnessCalculator:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def calculate(self, individual: Individual) -> Tuple[float, int]:
        """
        Calculate fitness as sum of squared errors.
        Returns (error, complexity) tuple.
        Lower error is better.
        If errors are equal, lower complexity is better.
        BUT: if one has higher complexity but even slightly lower error, it's better.
        """
        if not self.data_manager.data:
            return float('inf'), individual.complexity
        
        total_error = 0.0
        
        for inputs, expected_outputs in self.data_manager.data:
            actual_outputs = individual.forward(inputs)
            
            for actual, expected in zip(actual_outputs, expected_outputs):
                diff = actual - expected
                total_error += diff * diff
        
        return total_error, individual.complexity
    
    def compare_fitness(self, ind1: Individual, ind2: Individual) -> int:
        """
        Compare two individuals.
        Returns:
          -1 if ind1 is better
           1 if ind2 is better
           0 if equal
        
        Rules:
        - Lower error is always better
        - If errors are equal, lower complexity is better
        - Higher complexity with even slightly lower error is better
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
        self.internal_population: List[Individual] = []
        self.generation = 0
    
    def initialize(self):
        """Create initial zero neural network population with some random connections"""
        self.internal_population = []
        
        for _ in range(self.config.get('population_size')):
            individual = Individual()
            individual.input_size = self.data_manager.input_size
            individual.output_size = self.data_manager.output_size
            
            # Create input neurons
            for i in range(individual.input_size):
                individual.neurons[i] = Neuron(i, ActivationFunction.LINEAR)
            
            # Create output neurons
            for i in range(individual.output_size):
                neuron_id = individual.input_size + i
                individual.neurons[neuron_id] = Neuron(neuron_id, ActivationFunction.SIGMOID)
            
            # Add bias neuron (constant value of 1)
            bias_id = individual.input_size + individual.output_size
            individual.neurons[bias_id] = Neuron(bias_id, ActivationFunction.LINEAR)
            
            # Add some initial random connections to kickstart evolution
            # Connect each input to each output with small random weights
            for inp_id in range(individual.input_size):
                for out_id in range(individual.input_size, individual.input_size + individual.output_size):
                    weight = random.gauss(0, 0.5)
                    individual.connections.append(Connection(inp_id, out_id, weight))
            
            # Connect bias to outputs
            for out_id in range(individual.input_size, individual.input_size + individual.output_size):
                weight = random.gauss(0, 0.5)
                individual.connections.append(Connection(bias_id, out_id, weight))
            
            individual.fitness = float('inf')
            individual.complexity = len(individual.connections)
            
            self.internal_population.append(individual)
        
        self.generation = 0
    
    def evolve_generation(self, num_generations: int) -> List[str]:
        """Run evolution for specified number of generations"""
        progress_log = []
        
        for gen in range(num_generations):
            self.generation += 1
            
            # Each individual produces offspring
            new_population = []
            
            for parent_idx, parent in enumerate(self.internal_population):
                num_offspring = self.config.get('offspring_per_individual')
                
                best_offspring = None
                best_offspring_error = float('inf')
                
                for _ in range(num_offspring):
                    mutations = self.config.get('mutations_per_offspring')
                    offspring = self.mutator.mutate(parent, mutations)
                    
                    # Ensure offspring has at least some connections if parent has none
                    if len(parent.connections) == 0 and len(offspring.connections) == 0:
                        # Force add a connection
                        self._force_add_connection(offspring)
                    
                    # Evaluate offspring fitness
                    error, _ = self.fitness_calc.calculate(offspring)
                    
                    if error < best_offspring_error:
                        best_offspring = offspring
                        best_offspring_error = error
                
                # Compare best offspring with parent
                parent_error = parent.fitness
                if parent_error == float('inf'):
                    parent_error, _ = self.fitness_calc.calculate(parent)
                    parent.fitness = parent_error
                
                # Direct comparison: offspring replaces parent only if strictly better
                if best_offspring_error < parent_error - 1e-15:
                    new_population.append(best_offspring)
                elif abs(best_offspring_error - parent_error) <= 1e-15:
                    # Equal errors, prefer lower complexity
                    if best_offspring.complexity < parent.complexity:
                        new_population.append(best_offspring)
                    else:
                        new_population.append(parent.clone())
                else:
                    new_population.append(parent.clone())
            
            self.internal_population = new_population
            
            # Log progress
            best = min(self.internal_population, key=lambda x: x.fitness)
            avg_fitness = sum(ind.fitness for ind in self.internal_population) / len(self.internal_population)
            progress_log.append(
                f"Generation {self.generation}: Best={best.fitness:.10f}, Avg={avg_fitness:.10f}"
            )
        
        return progress_log
    
    def _force_add_connection(self, individual: Individual):
        """Force add a connection between random neurons, ensuring input can reach output"""
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
        msvcrt.getch()
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def show_menu() -> int:
    clear_screen()
    print("=" * 60)
    print("EVOTUS - Universal Neural Network with Evolutionary Strategies")
    print("=" * 60)
    print("\nMain Menu:")
    print("1. Start Evolution")
    print("2. Test Current Best Individual")
    print("3. View Population Status")
    print("4. Save Population")
    print("5. Load Population")
    print("6. Configure Settings")
    print("7. View/Edit Training Data")
    print("8. Export Best Individual to File")
    print("9. Import Individual from File")
    print("0. Exit")
    print("\n" + "=" * 60)
    
    try:
        choice = int(input("Enter choice: "))
        return choice
    except (ValueError, EOFError):
        return -1


def run_evolution(pop_manager: PopulationManager):
    clear_screen()
    print("=" * 60)
    print("EVOLUTION MODE")
    print("=" * 60)
    
    try:
        num_gens = int(input("Enter number of generations to evolve: "))
    except (ValueError, EOFError):
        print("Invalid input!")
        wait_for_key()
        return
    
    print(f"\nStarting evolution for {num_gens} generations...")
    print("-" * 60)
    
    progress = pop_manager.evolve_generation(num_gens)
    
    for log_entry in progress:
        print(log_entry)
    
    print("-" * 60)
    best = pop_manager.get_best()
    print(f"\nEvolution complete!")
    print(f"Best fitness: {best.fitness:.15f}")
    print(f"Complexity: {best.complexity}")
    print(f"Neurons: {len(best.neurons)}")
    print(f"Connections: {len(best.connections)}")
    
    print("\nPress any key to return to menu...")
    wait_for_key()


def test_individual(pop_manager: PopulationManager):
    best = pop_manager.get_best()
    
    if best is None or not best.neurons:
        print("\nNo trained individual available! Run evolution first.")
        wait_for_key()
        return
    
    clear_screen()
    print("=" * 60)
    print("TESTING MODE")
    print("=" * 60)
    print(f"Network: {len(best.neurons)} neurons, {len(best.connections)} connections")
    print(f"Input size: {best.input_size}, Output size: {best.output_size}")
    print("-" * 60)
    print("Enter input values separated by spaces (or 'q' to quit)")
    print()
    
    while True:
        try:
            user_input = input(f"Input ({best.input_size} values): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            values = [float(x) for x in user_input.split()]
            
            if len(values) != best.input_size:
                print(f"Error: Expected {best.input_size} values, got {len(values)}")
                continue
            
            outputs = best.forward(values)
            
            print(f"Output: {' '.join(f'{o:.10f}' for o in outputs)}")
            print()
            
        except ValueError:
            print("Error: Invalid input format")
        except EOFError:
            break
    
    print("\nPress any key to return to menu...")
    wait_for_key()


def view_status(pop_manager: PopulationManager):
    clear_screen()
    print("=" * 60)
    print("POPULATION STATUS")
    print("=" * 60)
    print(f"Generation: {pop_manager.generation}")
    print(f"Population size: {len(pop_manager.internal_population)}")
    print()
    
    if pop_manager.internal_population:
        sorted_pop = sorted(pop_manager.internal_population, key=lambda x: x.fitness)
        print("Top 5 individuals:")
        print("-" * 60)
        for i, ind in enumerate(sorted_pop[:5]):
            print(f"{i+1}. Fitness: {ind.fitness:.15f}, Complexity: {ind.complexity}, "
                  f"Neurons: {len(ind.neurons)}, Connections: {len(ind.connections)}")
    
    print("\nPress any key to return to menu...")
    wait_for_key()


def configure_settings(config: ConfigManager):
    clear_screen()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    
    for key, value in config.config.items():
        print(f"{key}: {value}")
    
    print("\nEnter setting name and value to change (or 'q' to quit)")
    
    while True:
        try:
            user_input = input("\nSetting name: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            if user_input not in config.config:
                print(f"Unknown setting: {user_input}")
                continue
            
            current_type = type(config.config[user_input])
            value_input = input(f"New value (current: {config.config[user_input]}): ").strip()
            
            try:
                new_value = current_type(value_input)
                config.set(user_input, new_value)
                print(f"Updated {user_input} = {new_value}")
            except ValueError:
                print(f"Invalid value type. Expected {current_type.__name__}")
            
        except EOFError:
            break
    
    print("\nPress any key to return to menu...")
    wait_for_key()


def export_individual(pop_manager: PopulationManager):
    best = pop_manager.get_best()
    
    if best is None:
        print("\nNo individual to export!")
        wait_for_key()
        return
    
    filename = input("Enter filename to export to: ").strip()
    
    if not filename:
        filename = "best_individual.json"
    
    with open(filename, 'w') as f:
        json.dump(best.save_to_dict(), f, indent=2)
    
    print(f"Exported to {filename}")
    print("\nPress any key to return to menu...")
    wait_for_key()


def import_individual(pop_manager: PopulationManager):
    filename = input("Enter filename to import from: ").strip()
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        wait_for_key()
        return
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        individual = Individual.load_from_dict(data)
        
        # Add to population or replace worst
        if pop_manager.internal_population:
            worst_idx = max(range(len(pop_manager.internal_population)), 
                          key=lambda i: pop_manager.internal_population[i].fitness)
            pop_manager.internal_population[worst_idx] = individual
        else:
            pop_manager.internal_population.append(individual)
        
        print(f"Imported individual from {filename}")
    except Exception as e:
        print(f"Error importing: {e}")
    
    print("\nPress any key to return to menu...")
    wait_for_key()


def main():
    # Initialize managers
    config = ConfigManager()
    data_manager = DataManager()
    
    # Create example data file if needed
    if not os.path.exists(DATA_FILE):
        data_manager.save_example()
        print(f"Created example data file: {DATA_FILE}")
    
    # Load training data
    if not data_manager.load():
        print(f"Error loading data from {DATA_FILE}")
        print("Please ensure the file exists and contains valid data.")
        return
    
    print(f"Loaded {len(data_manager.data)} training samples")
    print(f"Input size: {data_manager.input_size}, Output size: {data_manager.output_size}")
    
    mutator = Mutator(config)
    fitness_calc = FitnessCalculator(data_manager)
    pop_manager = PopulationManager(config, data_manager, mutator, fitness_calc)
    
    # Try to load existing population
    if pop_manager.load():
        print(f"Loaded population from generation {pop_manager.generation}")
    else:
        print("Initializing new population...")
        pop_manager.initialize()
    
    # Main loop
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
            print("Population saved!")
            wait_for_key()
        elif choice == 5:
            if pop_manager.load():
                print(f"Loaded population from generation {pop_manager.generation}")
            else:
                print("No saved population found!")
            wait_for_key()
        elif choice == 6:
            configure_settings(config)
        elif choice == 7:
            print(f"\nOpening {DATA_FILE} for editing...")
            print("Edit the file and press any key when done...")
            wait_for_key()
            # Reload data after editing
            if data_manager.load():
                print(f"Reloaded {len(data_manager.data)} training samples")
                # Reinitialize population with new data dimensions
                pop_manager.initialize()
            else:
                print("Error reloading data!")
            wait_for_key()
        elif choice == 8:
            export_individual(pop_manager)
        elif choice == 9:
            import_individual(pop_manager)
        elif choice == 0:
            print("Saving population before exit...")
            pop_manager.save()
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            wait_for_key()


if __name__ == "__main__":
    main()
