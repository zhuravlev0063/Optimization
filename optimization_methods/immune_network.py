# optimization_methods/immune_network.py
import numpy as np
from .base import OptimizationMethod

class ImmuneNetworkOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.pop_size = self.kwargs.get("pop_size", 50)
        self.n_b = self.kwargs.get("n_b", 10)
        self.n_c = self.kwargs.get("n_c", 5)
        self.b_s = self.kwargs.get("b_s", 0.2)
        self.b_b = self.kwargs.get("b_b", 0.01)
        self.b_r = self.kwargs.get("b_r", 0.1)
        self.b_n = self.kwargs.get("b_n", 0.1)
        self.mutation_rate = self.kwargs.get("mutation_rate", 0.1)
        self.range_lower = self.kwargs.get("range_lower", -5)
        self.range_upper = self.kwargs.get("range_upper", 5)

        # Параметры для обнаружения стагнации
        self.stagnation_threshold = self.kwargs.get("stagnation_threshold", 10)
        self.stagnation_counter = 0
        self.prev_best_fitness = None

    class Antibody:
        def __init__(self, x, y, outer):
            self.x = x
            self.y = y
            self.bg_affinity = 1 / (1 + outer.f(x, y))

    def compute_bb_affinity(self, ab1, ab2):
        return np.sqrt((ab1.x - ab2.x)**2 + (ab1.y - ab2.y)**2)

    def initialize_population(self):
        return [self.Antibody(np.random.uniform(self.range_lower, self.range_upper),
                              np.random.uniform(self.range_lower, self.range_upper), self)
                for _ in range(self.pop_size)]

    def clone_antibody(self, antibody):
        num_clones = int(1 + (self.n_c - 1) * antibody.bg_affinity)
        return [self.Antibody(antibody.x, antibody.y, self) for _ in range(num_clones)]

    def mutate_antibody(self, antibody):
        x_new = antibody.x + self.mutation_rate * np.random.uniform(-0.5, 0.5)
        y_new = antibody.y + self.mutation_rate * np.random.uniform(-0.5, 0.5)
        x_new = np.clip(x_new, self.range_lower, self.range_upper)
        y_new = np.clip(y_new, self.range_lower, self.range_upper)
        return self.Antibody(x_new, y_new, self)

    def run(self):
        S_b = self.initialize_population()
        S_m = []
        best_solution = None
        trajectory = []
        iterations_log = []

        for iteration in range(self.max_iterations):
            # Шаг 2.1: Отбор n_b лучших антител
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)
            selected = S_b[:self.n_b]

            # Шаг 2.2: Клонирование и мутация
            clones = []
            for ab in selected:
                clones.extend(self.clone_antibody(ab))
            clones = [self.mutate_antibody(clone) for clone in clones]

            # Шаг 2.3: Отбор лучших клонов
            clones = sorted(clones, key=lambda ab: ab.bg_affinity, reverse=True)
            n_d = int(self.b_s * len(clones))
            new_memory = clones[:n_d]
            new_memory = [ab for ab in new_memory if ab.bg_affinity >= self.b_b]
            S_m.extend(new_memory)

            # Шаг 2.4: Сжатие памяти
            i = 0
            while i < len(S_m):
                j = i + 1
                while j < len(S_m):
                    if self.compute_bb_affinity(S_m[i], S_m[j]) < self.b_r:
                        del S_m[j]
                    else:
                        j += 1
                i += 1
            S_b.extend(S_m)

            # Шаг 3: Сжатие сети
            i = 0
            while i < len(S_b):
                j = i + 1
                while j < len(S_b):
                    if self.compute_bb_affinity(S_b[i], S_b[j]) < self.b_r:
                        del S_b[j]
                    else:
                        j += 1
                i += 1

            # Шаг 4: Обновление части популяции случайными антителами
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)
            num_replace = int(self.b_n * self.pop_size)
            S_b = S_b[:self.pop_size - num_replace]
            S_b.extend(self.initialize_population()[:num_replace])

            # Ограничение размера популяции
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)[:self.pop_size]

            # Логирование
            current_best = S_b[0]
            current_fitness = self.f(current_best.x, current_best.y)
            if best_solution is None or current_fitness < self.f(best_solution.x, best_solution.y):
                best_solution = current_best
            trajectory.append([best_solution.x, best_solution.y])
            iterations_log.append(f"Итерация {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")

            # Проверка на стагнацию
            if self.prev_best_fitness is not None:
                if abs(self.prev_best_fitness - current_fitness) < 1e-8:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
            else:
                self.prev_best_fitness = current_fitness

            self.prev_best_fitness = current_fitness

            # Реакция на стагнацию
            if self.stagnation_counter >= self.stagnation_threshold:
                self.n_c = max(1, int(self.n_c * 0.7))  # уменьшить число клонов
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)  # увеличить мутацию
                self.b_n = min(0.5, self.b_n + 0.05)  # увеличить долю случайной замены
                self.stagnation_counter = 0

            # Условие остановки
            if current_fitness < 1e-10:
                iterations_log.append("Достигнут минимум!")
                break

        final_point = [best_solution.x, best_solution.y]
        return final_point, trajectory, "Иммунная сеть завершена", iterations_log
