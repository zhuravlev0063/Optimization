import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
import pickle
import os
from optimization_methods.hybrid_gapso import adapt_function
from optimization_methods.genetic_algorithm import GeneticAlgorithm
from optimization_methods.particle_swarm import ParticleSwarmOptimization
from optimization_methods.hybrid_gapso import HybridGAPSO

ga_params = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "bounds": [-5.12, 5.12]
}
pso_params = {
    "swarmsize": 20,
    "current_velocity_ratio": 0.5,
    "local_velocity_ratio": 2.0,
    "global_velocity_ratio": 5.0,
    "minvalues": [-5.12, -5.12],
    "maxvalues": [5.12, 5.12]
}
hybrid_params = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "bounds": [-5.12, 5.12],
    "swarmsize": 10,
    "current_velocity_ratio": 0.5,
    "local_velocity_ratio": 2.0,
    "global_velocity_ratio": 5.0
}


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, f, final_point, trajectory, method_name, constraints=None):
        self.ax.clear()
        x = np.linspace(-5.12, 5.12, 100)
        y = np.linspace(-5.12, 5.12, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[f(x_i, y_i) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
        self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        if trajectory:
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]
            traj_z = [f(x, y) for x, y in zip(traj_x, traj_y)]
            self.ax.plot(traj_x, traj_y, traj_z, 'r.-', label='Траектория')
        self.ax.scatter([final_point[0]], [final_point[1]], [f(final_point[0], final_point[1])], color='red', s=100,
                        label='Найденная точка')
        if constraints:
            for constraint in constraints:
                a, b, c = constraint['a'], constraint['b'], constraint['c']
                if b != 0:
                    y_con = np.linspace(-5.12, 5.12, 100)
                    x_con = (c - b * y_con) / a
                    z_con = np.array([f(x_i, y_i) for x_i, y_i in zip(x_con, y_con)])
                    self.ax.plot(x_con, y_con, z_con, 'k-', linewidth=2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('f(X, Y)')
        self.ax.set_title(f'Оптимизация: {method_name}')
        self.ax.legend()
        self.draw()


def run_algorithm(algorithm_class, f, dimension, max_iterations, **kwargs):
    adapted_f = adapt_function(f, dimension)
    algo = algorithm_class(adapted_f, np.zeros(dimension), max_iterations, dimension=dimension, **kwargs)
    result = algo.run()
    trajectory = result[1]
    fitness = [adapted_f(pos) for pos in trajectory]
    print(f"Algorithm: {algorithm_class.__name__}, Initial Iterations: {len(trajectory)}, Fitness: {fitness}")
    return trajectory, fitness


def plot_fitness_vs_iterations(f, dimension=2, max_iterations=20, save_path="fitness_vs_iterations.png"):
    cache_file = f"cache_fitness_vs_iterations_{f.__name__}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as cf:
            ga_fitness, pso_fitness, hybrid_fitness = pickle.load(cf)
    else:
        ga_traj, ga_fitness = run_algorithm(GeneticAlgorithm, f, dimension, max_iterations, **ga_params)
        pso_traj, pso_fitness = run_algorithm(ParticleSwarmOptimization, f, dimension, max_iterations, **pso_params)
        hybrid_traj, hybrid_fitness = run_algorithm(HybridGAPSO, f, dimension, max_iterations, **hybrid_params)
        with open(cache_file, 'wb') as cf:
            pickle.dump((ga_fitness, pso_fitness, hybrid_fitness), cf)

    # Определяем максимальное число итераций
    max_iter = max(len(ga_fitness), len(pso_fitness), len(hybrid_fitness))
    print(f"Maximum iterations detected: {max_iter}")

    # Дополняем данные до максимального числа итераций
    def extend_fitness(fitness, target_length):
        if len(fitness) >= target_length:
            return fitness[:target_length]
        last_value = fitness[-1] if fitness else 0
        return fitness + [last_value] * (target_length - len(fitness))

    ga_fitness_extended = extend_fitness(ga_fitness, max_iter)
    pso_fitness_extended = extend_fitness(pso_fitness, max_iter)
    hybrid_fitness_extended = extend_fitness(hybrid_fitness, max_iter)

    return {
        'ga': {'iterations': list(range(1, max_iter + 1)), 'fitness': ga_fitness_extended},
        'pso': {'iterations': list(range(1, max_iter + 1)), 'fitness': pso_fitness_extended},
        'hybrid': {'iterations': list(range(1, max_iter + 1)), 'fitness': hybrid_fitness_extended},
        'title': 'Зависимость значения функции от числа итераций',
        'xlabel': 'Номер итерации',
        'ylabel': 'Значение функции f(x)',
        'save_path': save_path
    }


def plot_fitness_vs_population(f, dimension=2, max_iterations=20, sizes=range(10, 101, 10),
                               save_path="fitness_vs_population.png"):
    cache_file = f"cache_fitness_vs_population_{f.__name__}.pkl"
    expected_length = len(sizes)
    print(f"Expected sizes length: {expected_length}")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as cf:
            ga_fitness, pso_fitness, hybrid_fitness = pickle.load(cf)
        print(
            f"Loaded from cache: ga_fitness length={len(ga_fitness)}, pso_fitness length={len(pso_fitness)}, hybrid_fitness length={len(hybrid_fitness)}")
        if (len(ga_fitness) != expected_length or
                len(pso_fitness) != expected_length or
                len(hybrid_fitness) != expected_length):
            print("Cache data length mismatch, recalculating...")
        else:
            return {
                'ga': {'sizes': list(sizes), 'fitness': ga_fitness},
                'pso': {'sizes': list(sizes), 'fitness': pso_fitness},
                'hybrid': {'sizes': list(sizes), 'fitness': hybrid_fitness},
                'title': 'Зависимость точности от размера популяции/роя',
                'xlabel': 'Размер популяции/роя',
                'ylabel': 'Конечное значение функции f(x)',
                'save_path': save_path
            }

    ga_fitness = []
    pso_fitness = []
    hybrid_fitness = []
    for size in sizes:
        ga_params['population_size'] = size
        pso_params['swarmsize'] = size
        hybrid_params['population_size'] = size
        traj, fitness = run_algorithm(GeneticAlgorithm, f, dimension, max_iterations, **ga_params)
        ga_fitness.append(fitness[-1])
        traj, fitness = run_algorithm(ParticleSwarmOptimization, f, dimension, max_iterations, **pso_params)
        pso_fitness.append(fitness[-1])
        traj, fitness = run_algorithm(HybridGAPSO, f, dimension, max_iterations, **hybrid_params)
        hybrid_fitness.append(fitness[-1])
    print(
        f"Calculated: ga_fitness length={len(ga_fitness)}, pso_fitness length={len(pso_fitness)}, hybrid_fitness length={len(hybrid_fitness)}")

    with open(cache_file, 'wb') as cf:
        pickle.dump((ga_fitness, pso_fitness, hybrid_fitness), cf)

    return {
        'ga': {'sizes': list(sizes), 'fitness': ga_fitness},
        'pso': {'sizes': list(sizes), 'fitness': pso_fitness},
        'hybrid': {'sizes': list(sizes), 'fitness': hybrid_fitness},
        'title': 'Зависимость точности от размера популяции/роя',
        'xlabel': 'Размер популяции/роя',
        'ylabel': 'Конечное значение функции f(x)',
        'save_path': save_path
    }


def plot_time_vs_population(f, dimension=2, max_iterations=20, sizes=range(10, 101, 10),
                            save_path="time_vs_population.png"):
    cache_file = f"cache_time_vs_population_{f.__name__}.pkl"
    expected_length = len(sizes)
    print(f"Expected sizes length: {expected_length}")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as cf:
            ga_times, pso_times, hybrid_times = pickle.load(cf)
        print(
            f"Loaded from cache: ga_times length={len(ga_times)}, pso_times length={len(pso_times)}, hybrid_times length={len(hybrid_times)}")
        if (len(ga_times) != expected_length or
                len(pso_times) != expected_length or
                len(hybrid_times) != expected_length):
            print("Cache data length mismatch, recalculating...")
        else:
            return {
                'ga': {'sizes': list(sizes), 'times': ga_times},
                'pso': {'sizes': list(sizes), 'times': pso_times},
                'hybrid': {'sizes': list(sizes), 'times': hybrid_times},
                'title': 'Зависимость времени выполнения от размера популяции/роя',
                'xlabel': 'Размер популяции/роя',
                'ylabel': 'Время выполнения (секунды)',
                'save_path': save_path
            }

    def measure_time(algorithm_class, f, dimension, max_iterations, runs=1, **kwargs):
        dim_times = []
        for _ in range(runs):
            start_time = time.time()
            adapted_f = adapt_function(f, dimension)
            algo = algorithm_class(adapted_f, np.zeros(dimension), max_iterations, dimension=dimension, **kwargs)
            algo.run()
            end_time = time.time()
            print(f"Run time: {end_time - start_time}")
            dim_times.append(end_time - start_time)
        return np.mean(dim_times)

    ga_times = []
    pso_times = []
    hybrid_times = []
    for size in sizes:
        ga_params['population_size'] = size
        pso_params['swarmsize'] = size
        hybrid_params['population_size'] = size
        ga_times.append(measure_time(GeneticAlgorithm, f, dimension, max_iterations, runs=1, **ga_params))
        pso_times.append(measure_time(ParticleSwarmOptimization, f, dimension, max_iterations, runs=1, **pso_params))
        hybrid_times.append(measure_time(HybridGAPSO, f, dimension, max_iterations, runs=1, **hybrid_params))
    print(
        f"Calculated: ga_times length={len(ga_times)}, pso_times length={len(pso_times)}, hybrid_times length={len(hybrid_times)}")

    with open(cache_file, 'wb') as cf:
        pickle.dump((ga_times, pso_times, hybrid_times), cf)

    return {
        'ga': {'sizes': list(sizes), 'times': ga_times},
        'pso': {'sizes': list(sizes), 'times': pso_times},
        'hybrid': {'sizes': list(sizes), 'times': hybrid_times},
        'title': 'Зависимость времени выполнения от размера популяции/роя',
        'xlabel': 'Размер популяции/роя',
        'ylabel': 'Время выполнения (секунды)',
        'save_path': save_path
    }