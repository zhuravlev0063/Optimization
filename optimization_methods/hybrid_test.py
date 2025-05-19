import numpy as np
from .base import OptimizationMethod

# Адаптер для функций
def adapt_function(f, dimension):
    if dimension == 2:
        def wrapper(x):
            return f(x[0], x[1])
        return wrapper
    else:
        def wrapper(x):
            return np.sum(x**2)  # Сферическая функция для n-мерности
        return wrapper

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

        fitness = np.array([self.f(ind) for ind in population])
        top_indices = np.argsort(fitness)[:10]
        top_individuals = population[top_indices]

        trajectory = [ind for _, ind, _ in history]
        iterations_log = [f"Итерация {gen}: x={ind}, f(x)={fit}" for gen, ind, fit in history]
        return best_individual, trajectory, "Генетический алгоритм завершён", iterations_log, top_individuals

class ParticleSwarmOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.swarmsize = self.kwargs.get("swarmsize", 50)
        self.bounds = np.array(self.kwargs.get("bounds", [-5.12, 5.12]))
        self.dimension = self.kwargs.get("dimension", 2)
        self.current_velocity_ratio = self.kwargs.get("current_velocity_ratio", 0.5)
        self.local_velocity_ratio = self.kwargs.get("local_velocity_ratio", 2.0)
        self.global_velocity_ratio = self.kwargs.get("global_velocity_ratio", 5.0)
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4
        self.initial_positions = self.kwargs.get("initial_positions", None)
        self.swarm = self._create_swarm()

    def _create_swarm(self):
        class Particle:
            def __init__(self, outer, position=None):
                if position is None:
                    self.position = np.random.rand(outer.dimension) * (outer.bounds[1] - outer.bounds[0]) + outer.bounds[0]
                else:
                    self.position = position.copy()
                self.velocity = np.random.rand(outer.dimension) * (outer.bounds[1] - outer.bounds[0]) - (outer.bounds[1] - outer.bounds[0])
                self.best_position = self.position.copy()
                self.best_value = outer.f(self.position)

            def update(self, outer, global_best_position):
                rnd_local = np.random.rand(outer.dimension)
                rnd_global = np.random.rand(outer.dimension)
                velo_ratio = outer.local_velocity_ratio + outer.global_velocity_ratio
                common_ratio = 2.0 * outer.current_velocity_ratio / abs(
                    2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))

                new_velocity = (common_ratio * self.velocity +
                                common_ratio * outer.local_velocity_ratio * rnd_local * (
                                            self.best_position - self.position) +
                                common_ratio * outer.global_velocity_ratio * rnd_global * (
                                            global_best_position - self.position))

                self.velocity = new_velocity
                self.position += self.velocity
                self.position = np.clip(self.position, outer.bounds[0], outer.bounds[1])
                value = outer.f(self.position)
                if value < self.best_value:
                    self.best_value = value
                    self.best_position = self.position.copy()

        swarm = []
        if self.initial_positions is not None:
            for pos in self.initial_positions:
                swarm.append(Particle(self, pos))
            for _ in range(self.swarmsize - len(self.initial_positions)):
                swarm.append(Particle(self))
        else:
            swarm = [Particle(self) for _ in range(self.swarmsize)]

        global_best_value = min(p.best_value for p in swarm)
        global_best_position = next(p.best_position for p in swarm if p.best_value == global_best_value)
        return swarm, global_best_position, global_best_value

    def run(self):
        swarm, global_best_position, global_best_value = self.swarm
        trajectory = [global_best_position.copy()]
        iterations_log = []

        for i in range(self.max_iterations):
            for particle in swarm:
                particle.update(self, global_best_position)
                if particle.best_value < global_best_value:
                    global_best_value = particle.best_value
                    global_best_position = particle.best_position.copy()

            trajectory.append(global_best_position.copy())
            f_val = self.f(global_best_position)
            iterations_log.append(f"Итерация {i}: x={global_best_position}, f(x)={f_val}")

        return global_best_position, trajectory, "PSO завершён", iterations_log

class HybridGAPSO(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.ga_iterations = self.kwargs.get("ga_iterations", 50)
        self.pso_iterations = self.kwargs.get("pso_iterations", 50)
        self.bounds = self.kwargs.get("bounds", [-5.12, 5.12])
        self.dimension = self.kwargs.get("dimension", 2)

    def run(self):
        ga_kwargs = {
            "population_size": self.kwargs.get("population_size", 200),
            "mutation_rate": self.kwargs.get("mutation_rate", 0.1),
            "bounds": self.bounds,
            "dimension": self.dimension
        }
        ga = GeneticAlgorithm(self.f, self.initial_point, self.ga_iterations, **ga_kwargs)
        ga_best, ga_trajectory, ga_status, ga_log, top_individuals = ga.run()

        pso_kwargs = {
            "swarmsize": self.kwargs.get("swarmsize", 50),
            "bounds": self.bounds,
            "dimension": self.dimension,
            "current_velocity_ratio": self.kwargs.get("current_velocity_ratio", 0.5),
            "local_velocity_ratio": self.kwargs.get("local_velocity_ratio", 2.0),
            "global_velocity_ratio": self.kwargs.get("global_velocity_ratio", 5.0),
            "initial_positions": top_individuals
        }
        pso = ParticleSwarmOptimization(self.f, self.initial_point, self.pso_iterations, **pso_kwargs)
        pso_best, pso_trajectory, pso_status, pso_log = pso.run()

        trajectory = ga_trajectory + pso_trajectory
        iterations_log = ga_log + pso_log
        return pso_best, trajectory, "Гибридный ГА+PSO завершён", iterations_log