import numpy as np
from scipy.optimize import minimize
from .base import OptimizationMethod

class ImmuneNetworkOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.pop_size = self.kwargs.get("pop_size", 50)  # Уменьшено для скорости
        self.n_b = self.kwargs.get("n_b", 20)
        self.n_c = self.kwargs.get("n_c", 10)
        self.b_s = self.kwargs.get("b_s", 0.2)
        self.b_b = self.kwargs.get("b_b", 0.005)
        self.b_r = self.kwargs.get("b_r", 0.3)
        self.b_n = self.kwargs.get("b_n", 0.3)  # Уменьшено для меньшей замены
        self.mutation_rate = self.kwargs.get("mutation_rate", 0.8)
        self.range_lower = self.kwargs.get("range_lower", -5)
        self.range_upper = self.kwargs.get("range_upper", 5)
        self.stagnation_limit = self.kwargs.get("stagnation_limit", 3)  # Уменьшено для частых перезапусков
        self.tolerance = self.kwargs.get("tolerance", 1e-6)
        self.max_memory_size = self.kwargs.get("max_memory_size", 50)  # Увеличено

    class Antibody:
        def __init__(self, x, y, outer):
            self.x = x
            self.y = y
            fitness = outer.f(x, y)
            if np.isinf(fitness) or np.isnan(fitness):
                self.bg_affinity = 0.0
            else:
                self.bg_affinity = 1 / (1 + fitness)

    def compute_bb_affinity(self, ab1, ab2):
        return np.sqrt((ab1.x - ab2.x)**2 + (ab1.y - ab2.y)**2)

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            x = np.random.uniform(self.range_lower, self.range_upper)
            y = np.random.uniform(self.range_lower, self.range_upper)
            population.append(self.Antibody(x, y, self))
        return population

    def clone_antibody(self, antibody, fitness):
        if np.isinf(fitness) or np.isnan(fitness):
            fitness = 1e10
        dynamic_n_c = max(5, self.n_c - int(5 * np.log10(max(1, min(fitness, 1e10)))))
        dynamic_n_c = min(dynamic_n_c, 50)
        clones = []
        for _ in range(dynamic_n_c):
            clones.append(self.Antibody(antibody.x, antibody.y, self))
        return clones

    def mutate_antibody(self, antibody, mutation_rate):
        fitness = self.f(antibody.x, antibody.y)
        if np.isinf(fitness) or np.isnan(fitness):
            fitness = 1e10
        # Адаптивные мутации для f(x) ≈ 0.286
        mutation_range_x = 1e-5 if fitness < 0.01 else 0.005 if fitness < 0.5 else 0.01 if fitness < 1.0 else 0.1
        mutation_range_y = mutation_range_x * 10  # Увеличенный шаг по y
        x_new = antibody.x + mutation_rate * np.random.normal(0, mutation_range_x)
        y_new = antibody.y + mutation_rate * np.random.normal(0, mutation_range_y)
        x_new = np.clip(x_new, self.range_lower, self.range_upper)
        y_new = np.clip(y_new, self.range_lower, self.range_upper)
        return self.Antibody(x_new, y_new, self)

    def gradient_descent_step(self, antibody):
        x, y = antibody.x, antibody.y
        fitness = self.f(x, y)
        if np.isinf(fitness) or np.isnan(fitness):
            return antibody
        h = 1e-6
        grad_x = (self.f(x + h, y) - self.f(x - h, y)) / (2 * h)
        grad_y = (self.f(x, y + h) - self.f(x, y - h)) / (2 * h)
        if np.isinf(grad_x) or np.isnan(grad_x) or np.isinf(grad_y) or np.isnan(grad_y):
            return antibody
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        if grad_norm > 0:
            grad_x /= grad_norm
            grad_y /= grad_norm
        step_size = 1e-5 if fitness < 0.01 else 0.05 if fitness < 0.5 else 0.01 if fitness < 1.0 else 0.1
        x_new = np.clip(x - step_size * grad_x, self.range_lower, self.range_upper)
        y_new = np.clip(y - step_size * grad_y, self.range_lower, self.range_upper)
        return self.Antibody(x_new, y_new, self)

    def nelder_mead_step(self, antibody):
        def objective(point):
            fitness = self.f(point[0], point[1])
            return fitness if not (np.isinf(fitness) or np.isnan(fitness)) else 1e10
        result = minimize(objective, [antibody.x, antibody.y], method='Nelder-Mead', bounds=[(self.range_lower, self.range_upper)]*2)
        x_new, y_new = result.x
        if np.isinf(result.fun) or np.isnan(result.fun):
            return antibody
        return self.Antibody(x_new, y_new, self)

    def turbo_mutation(self, antibody):
        fitness = self.f(antibody.x, antibody.y)
        if np.isinf(fitness) or np.isnan(fitness):
            fitness = 1e10
        jump_range = 0.01 if fitness < 0.01 else 0.2 if fitness < 0.5 else 0.5 if fitness < 1.0 else 1.0
        x_new = np.clip(antibody.x + np.random.uniform(-jump_range, jump_range), self.range_lower, self.range_upper)
        y_new = np.clip(antibody.y + np.random.uniform(-jump_range * 10, jump_range * 10), self.range_lower, self.range_upper)
        return self.Antibody(x_new, y_new, self)

    def run(self):
        S_b = self.initialize_population()
        S_m = []
        best_solution = None
        best_fitness = float('inf')
        trajectory = []
        iterations_log = []
        stagnation_count = 0
        mutation_cycle = 0

        for iteration in range(self.max_iterations):

            # Динамическое управление параметрами
            dynamic_n_c = self.n_c - int(5 * stagnation_count / self.stagnation_limit)
            dynamic_pop_size = max(50, self.pop_size - int(150 * np.log10(max(1, min(best_fitness, 1e10)))))
            dynamic_b_s = self.b_s * (1 + stagnation_count / self.stagnation_limit)
            dynamic_b_n = self.b_n * (1 + 2 * stagnation_count / self.stagnation_limit)
            dynamic_b_r = 1e-4 if best_fitness < 0.5 else self.b_r * (1 + 0.5 * iteration / self.max_iterations)

            # Шаг 2.1: Отбор n_b лучших антител
            if not S_b:
                iterations_log.append("Ошибка: S_b пустой, переинициализация популяции")
                S_b = self.initialize_population()
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)
            selected = S_b[:min(self.n_b, len(S_b))]

            # Гибридный локальный поиск
            if selected:
                fitness = self.f(selected[0].x, selected[0].y)
                if np.isinf(fitness) or np.isnan(fitness):
                    selected[0] = self.turbo_mutation(selected[0])
                elif fitness < 0.5:  # Нелдер-Мид для f(x) < 0.5
                    selected[0] = self.nelder_mead_step(selected[0])
                else:
                    selected[0] = self.gradient_descent_step(selected[0])

            # Шаг 2.2: Клонирование и мутация
            clones = []
            for ab in selected:
                fitness = self.f(ab.x, ab.y)
                clones.extend(self.clone_antibody(ab, fitness))

            # Адаптивная мутация
            adaptive_mutation_rate = self.mutation_rate * (1 + 0.5 * stagnation_count / self.stagnation_limit)
            min_mutation_rate = 0.5 if best_fitness < 0.5 else 0.2
            if stagnation_count > 2:
                mutation_cycle = (mutation_cycle + 1) % 5
                adaptive_mutation_rate = min(2.0 * (1 + mutation_cycle / 5), 2.0)
            adaptive_mutation_rate = max(adaptive_mutation_rate, min_mutation_rate)
            clones = [self.mutate_antibody(clone, adaptive_mutation_rate) for clone in clones]

            # Турбо-мутация при стагнации
            if stagnation_count > 1:  # Ускоряем турбо-мутации
                for i in range(len(clones)):
                    if np.random.random() < 0.7:
                        clones[i] = self.turbo_mutation(clones[i])

            # Шаг 2.3: Отбор лучших клонов
            if clones:
                clones = sorted(clones, key=lambda ab: ab.bg_affinity, reverse=True)
                n_d = int(dynamic_b_s * len(clones))
                new_memory = clones[:n_d]
                new_memory = [ab for ab in new_memory if ab.bg_affinity >= self.b_b]
                S_m.extend(new_memory)

            # Шаг 2.4: Сжатие памяти
            S_m = sorted(S_m, key=lambda ab: ab.bg_affinity, reverse=True)[:self.max_memory_size]
            S_m_new = []
            for i, ab1 in enumerate(S_m):
                keep = True
                for j in range(i + 1, len(S_m)):
                    if self.compute_bb_affinity(ab1, S_m[j]) < dynamic_b_r:
                        keep = False
                        break
                if keep:
                    S_m_new.append(ab1)
            S_m = S_m_new
            if len(S_m) < 10 and best_fitness < 0.5:
                S_m.extend(clones[:max(0, 10 - len(S_m))])

            # Шаг 3: Сжатие популяции
            S_b.extend(S_m)
            S_b_new = []
            for i, ab1 in enumerate(S_b):
                keep = True
                for j in range(i + 1, len(S_b)):
                    if self.compute_bb_affinity(ab1, S_b[j]) < 0.001:
                        keep = False
                        break
                if keep:
                    S_b_new.append(ab1)
            S_b = S_b_new

            # Шаг 4: Динамическая замена
            if not S_b:
                iterations_log.append("Ошибка: S_b пустой после сжатия, переинициализация")
                S_b = self.initialize_population()
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)
            num_replace = int(dynamic_b_n * dynamic_pop_size)
            S_b = S_b[:max(dynamic_pop_size - num_replace, 0)]
            new_antibodies = []
            for _ in range(min(num_replace, dynamic_pop_size - len(S_b))):
                if np.random.random() < 0.9 and best_fitness < 0.5:
                    x = np.random.normal(best_solution.x if best_solution else 0, 0.005)
                    y = np.random.normal(best_solution.y if best_solution else 0, 0.05)
                elif np.random.random() < 0.8 and best_fitness < 1.0:
                    x = np.random.normal(0, 0.01)
                    y = np.random.normal(0, 0.1)
                else:
                    x = np.random.uniform(-0.1, 0.1)
                    y = np.random.uniform(-0.1, 0.1)
                x = np.clip(x, self.range_lower, self.range_upper)
                y = np.clip(y, self.range_lower, self.range_upper)
                new_antibodies.append(self.Antibody(x, y, self))
            S_b.extend(new_antibodies)

            # Добавление лучшего решения
            if best_solution is not None:
                S_b.append(self.Antibody(best_solution.x, best_solution.y, self))

            # Ограничение размера популяции
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)[:dynamic_pop_size]

            # Логирование и проверка стагнации
            if not S_b:
                iterations_log.append("Ошибка: S_b пустой перед логированием, переинициализация")
                S_b = self.initialize_population()
            current_best = S_b[0]
            current_fitness = self.f(current_best.x, current_best.y)
            if np.isinf(current_fitness) or np.isnan(current_fitness):
                current_fitness = 1e10
            if current_fitness < best_fitness - 1e-12:
                best_solution = current_best
                best_fitness = current_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            trajectory.append([best_solution.x, best_solution.y])
            iterations_log.append(f"Итерация {iteration}: x=[{best_solution.x:.10f}, {best_solution.y:.10f}], f(x)={best_fitness:.10f}")

            # Перезапуск при стагнации
            if stagnation_count > self.stagnation_limit:
                keep_count = int(0.1 * dynamic_pop_size)
                S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)[:keep_count]
                new_antibodies = []
                for _ in range(dynamic_pop_size - len(S_b)):
                    if np.random.random() < 0.9 and best_fitness < 0.5:
                        x = np.random.normal(best_solution.x if best_solution else 0, 0.005)
                        y = np.random.normal(best_solution.y if best_solution else 0, 0.05)
                    elif np.random.random() < 0.8 and best_fitness < 1.0:
                        x = np.random.normal(0, 0.01)
                        y = np.random.normal(0, 0.1)
                    else:
                        x = np.random.uniform(-0.1, 0.1)
                        y = np.random.uniform(-0.1, 0.1)
                    x = np.clip(x, self.range_lower, self.range_upper)
                    y = np.clip(y, self.range_lower, self.range_upper)
                    new_antibodies.append(self.Antibody(x, y, self))
                S_b.extend(new_antibodies)
                S_m = sorted(S_m, key=lambda ab: ab.bg_affinity, reverse=True)[:self.max_memory_size // 4]
                stagnation_count = 0
                mutation_cycle = 0

            # Условие остановки
            if np.isclose(best_fitness, 0.0, atol=self.tolerance) or best_fitness < self.tolerance:
                iterations_log.append("Достигнут минимум!")
                break

        final_point = [best_solution.x, best_solution.y]
        return final_point, trajectory, "Иммунная сеть завершена", iterations_log