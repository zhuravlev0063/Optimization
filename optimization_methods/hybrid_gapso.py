# optimization_methods/hybrid_gapso.py
import numpy as np
from .base import OptimizationMethod
from .genetic_algorithm import GeneticAlgorithm
from .particle_swarm import ParticleSwarmOptimization


class HybridGAPSO(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        # Параметры GA
        self.population_size = self.kwargs.get("population_size", 100)
        self.mutation_rate = self.kwargs.get("mutation_rate", 0.1)
        self.bounds = self.kwargs.get("bounds", (-5.12, 5.12))
        # Параметры PSO
        self.pso_swarmsize = self.kwargs.get("pso_swarmsize", 10)  # Уточняем 10% популяции
        self.minvalues = np.array(self.kwargs.get("minvalues", [self.bounds[0], self.bounds[0]]))
        self.maxvalues = np.array(self.kwargs.get("maxvalues", [self.bounds[1], self.bounds[1]]))
        self.current_velocity_ratio = self.kwargs.get("current_velocity_ratio", 0.5)
        self.local_velocity_ratio = self.kwargs.get("local_velocity_ratio", 2.0)
        self.global_velocity_ratio = self.kwargs.get("global_velocity_ratio", 2.0)
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"

    def run(self):
        # Инициализация популяции GA
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, 2))
        best_individual = None
        best_fitness = float('inf')
        history = []

        for generation in range(self.max_iterations):
            # Оценка популяции
            fitness = np.array([self.f(ind[0], ind[1]) for ind in population])
            current_best_idx = np.argmin(fitness)

            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()
                history.append((generation, best_individual, best_fitness))

            # Выбираем лучшие особи для PSO (10% популяции)
            sorted_indices = np.argsort(fitness)[:self.pso_swarmsize]
            best_individuals = population[sorted_indices].tolist()

            # Запуск PSO для уточнения лучших особей (1 итерация)
            pso = ParticleSwarmOptimization(self.f, best_individual, 1,
                                            swarmsize=self.pso_swarmsize,
                                            minvalues=self.minvalues,
                                            maxvalues=self.maxvalues,
                                            current_velocity_ratio=self.current_velocity_ratio,
                                            local_velocity_ratio=self.local_velocity_ratio,
                                            global_velocity_ratio=self.global_velocity_ratio)
            # Заменяем начальный рой PSO на выбранные особи
            pso.swarm = [(lambda p: type("", (), {
                "position": np.array(p),
                "velocity": np.random.rand(2) * 0.1,
                "best_position": np.array(p),
                "best_value": self.f(p[0], p[1]),
                "update": pso.swarm[0][0].update
            }))(ind) for ind in best_individuals]
            pso_result, pso_trajectory, _, pso_log = pso.run()

            # Обновляем популяцию улучшенными особями
            for i, idx in enumerate(sorted_indices):
                population[idx] = pso.swarm[i].position

            # Турнирный отбор
            selected_indices = []
            for _ in range(self.population_size):
                candidates = np.random.choice(self.population_size, size=5, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                selected_indices.append(winner)
            selected_population = population[selected_indices]

            # Скрещивание
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    parent1, parent2 = selected_population[i], selected_population[i + 1]
                    alpha = np.random.rand()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                    selected_population[i], selected_population[i + 1] = child1, child2

            # Мутация
            mutation_strength = 0.1 * (0.99 ** generation)
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    selected_population[i] += np.random.normal(0, mutation_strength, 2)
                    selected_population[i] = np.clip(selected_population[i], self.bounds[0], self.bounds[1])

            # Элитизм: замена худшей особи на лучшую
            if best_individual is not None:
                worst_idx = np.argmax([self.f(ind[0], ind[1]) for ind in selected_population])
                selected_population[worst_idx] = best_individual

            population = selected_population

        trajectory = [ind for _, ind, _ in history]
        iterations_log = [f"Итерация {gen}: x={ind}, f(x)={fit}" for gen, ind, fit in history]
        return best_individual, trajectory, "Гибридный GA-PSO завершён", iterations_log
