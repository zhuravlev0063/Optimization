import numpy as np
from .base import OptimizationMethod

class GeneticAlgorithm(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.dimension = self.kwargs.get("dimension", 2)

    def run(self):
        population_size = self.kwargs.get("population_size", 200)
        generations = self.max_iterations
        mutation_rate = self.kwargs.get("mutation_rate", 0.1)
        bounds = self.kwargs.get("bounds", (-5.12, 5.12))
        dimension = self.dimension

        population = np.random.uniform(bounds[0], bounds[1], (population_size, dimension))
        best_individual = None
        best_fitness = float('inf')
        history = []

        for generation in range(generations):
            fitness = np.array([self.f(ind) for ind in population])
            current_best_idx = np.argmin(fitness)

            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()
                history.append((generation, best_individual, best_fitness))

            selected_indices = []
            for _ in range(population_size):
                candidates = np.random.choice(population_size, size=5, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                selected_indices.append(winner)
            selected_population = population[selected_indices]

            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    parent1, parent2 = selected_population[i], selected_population[i + 1]
                    alpha = np.random.rand()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                    selected_population[i], selected_population[i + 1] = child1, child2

            mutation_strength = 0.1 * (0.99 ** generation)
            for i in range(population_size):
                if np.random.rand() < mutation_rate:
                    selected_population[i] += np.random.normal(0, mutation_strength, dimension)
                    selected_population[i] = np.clip(selected_population[i], bounds[0], bounds[1])

            if best_individual is not None:
                worst_idx = np.argmax([self.f(ind) for ind in selected_population])
                selected_population[worst_idx] = best_individual

            population = selected_population

        trajectory = [ind for _, ind, _ in history]
        iterations_log = [f"Итерация {gen}: x={ind}, f(x)={fit}" for gen, ind, fit in history]
        return best_individual, trajectory, "Генетический алгоритм завершён", iterations_log