import numpy as np
from optimization import OptimizationMethod, gradient_f
import sympy as sp
import re

class GradientDescent(OptimizationMethod):
    def run(self):
        x = self.initial_point.copy()
        trajectory = [x.copy()]
        k = 0
        iterations_log = []
        learning_rate = self.kwargs.get("learning_rate", 0.1)
        epsilon1 = self.kwargs.get("epsilon1", 1e-3)
        epsilon2 = self.kwargs.get("epsilon2", 1e-3)

        while k < self.max_iterations:
            grad = gradient_f(self.f, x[0], x[1])
            f_x = self.f(x[0], x[1])
            iterations_log.append(f"Итерация {k}: x={x}, f(x)={f_x}")

            if abs(f_x) < epsilon1:
                return x, trajectory, "|f(x*)| < epsilon1", iterations_log

            t_k = learning_rate
            x_next = x - t_k * grad
            f_x_next = self.f(x_next[0], x_next[1])

            step_attempts = 0
            while f_x_next >= f_x and step_attempts < 2:
                t_k /= 2
                x_next = x - t_k * grad
                f_x_next = self.f(x_next[0], x_next[1])
                step_attempts += 1

            if step_attempts == 2:
                return x, trajectory, "Шаг стал слишком маленьким", iterations_log

            if np.linalg.norm(x_next - x) < epsilon2 and abs(f_x_next - f_x) < epsilon2:
                return x_next, trajectory, "||xk+1 - xk|| < epsilon2", iterations_log

            x = x_next
            trajectory.append(x.copy())
            k += 1

        return x, trajectory, "Превышено число итераций", iterations_log

class QuadraticSimplex(OptimizationMethod):
    def __init__(self, f, max_iterations, **kwargs):
        super().__init__(f, None, max_iterations, **kwargs)
        self.constraints = self.kwargs.get("constraints", [{"a": 1.0, "b": 2.0, "c": 2.0}])

    def dop(self, x, idx):
        return (self.constraints[idx]["a"] * x[0] +
                self.constraints[idx]["b"] * x[1] -
                self.constraints[idx]["c"])  # Отрицательное значение — ограничение выполняется

    def is_linear(self):
        x0, x1 = sp.symbols('x0 x1')
        f_expr = self.f(x0, x1)
        df_dx0 = sp.diff(f_expr, x0)
        df_dx1 = sp.diff(f_expr, x1)
        return df_dx0.is_constant() and df_dx1.is_constant()

    def vect(self):
        x0, x1 = sp.symbols('x0 x1')
        f_expr = self.f(x0, x1)
        df_dx0 = sp.diff(f_expr, x0)
        df_dx1 = sp.diff(f_expr, x1)
        return [float(df_dx0), float(df_dx1)] + [0] * len(self.constraints)

    def extract_and_modify2(self):
        a = []
        b = []
        for i in range(len(self.constraints)):
            row = [self.constraints[i]["a"], self.constraints[i]["b"]] + \
                  [1 if i == j else 0 for j in range(len(self.constraints))]
            a.append(row)
            b.append(self.constraints[i]["c"])
        return a, b

    def maxVal(self, a, c, b):
        min_val = min([x for x in c if x < 0], default=0)
        if min_val >= 0:
            return None, None
        max_index = c.index(min_val)
        ratios = [(b[i] / a[i][max_index], i) for i in range(len(a)) if a[i][max_index] > 0]
        if not ratios:
            return None, None
        min_ratio, min_index = min(ratios, key=lambda x: x[0])
        if min_ratio < 0:
            return None, None
        return min_index, max_index

    def ch_ab(self, a, m, c, b):
        id1, id2 = m
        el = a[id1][id2]
        if el == 0:
            return a, c, b
        for i in range(len(a[0])):
            a[id1][i] /= el
        b[id1] /= el
        timeC = c[id2]
        for i in range(len(c)):
            c[i] -= timeC * a[id1][i]
        for i in range(len(a)):
            if i != id1:
                time = a[i][id2]
                for j in range(len(a[0])):
                    a[i][j] -= time * a[id1][j]
                b[i] -= time * b[id1]
        return a, c, b

    def run_simplex(self):
        c = self.vect()
        a, b = self.extract_and_modify2()
        c1 = c.copy()
        d = b.copy()

        iterations = 0
        tt = list(range(2, 2 + len(self.constraints)))

        trajectory = []
        iterations_log = []

        while iterations < self.max_iterations:
            m = self.maxVal(a, c1, b)
            if m[0] is None or m[1] is None:
                iterations_log.append("Программа завершена: оптимальное решение или ошибка.")
                break

            id1, id2 = m
            tt[id1] = id2
            a, c1, b = self.ch_ab(a, m, c1, b)
            d = b.copy()

            result = [0.0] * (2 + len(self.constraints))
            for idx, val in zip(tt, d):
                result[idx] = val
            x = result[:2]
            f_val = self.f(x[0], x[1])
            constraint_vals = [self.dop(x, i) for i in range(len(self.constraints))]
            trajectory.append(np.array(x))
            iterations_log.append(f"Итерация {iterations}: x={x}, f(x)={f_val}, constraints={constraint_vals}")

            if all(x >= 0 for x in c1):
                break

            iterations += 1

        result = [0.0] * (2 + len(self.constraints))
        for idx, val in zip(tt, d):
            result[idx] = val
        final_point = np.array(result[:2])
        f_val = self.f(final_point[0], final_point[1])
        constraint_vals = [self.dop(final_point, i) for i in range(len(self.constraints))]
        trajectory.append(final_point)
        iterations_log.append(f"Финальная итерация: x={final_point}, f(x)={f_val}, constraints={constraint_vals}")
        return final_point, trajectory, "Симплекс завершён", iterations_log

    def run_qp(self):
        x0, x1 = sp.symbols('x0 x1')
        f_expr = self.f(x0, x1)
        Q = np.array([[float(sp.diff(f_expr, x0, 2)), float(sp.diff(f_expr, x0, x1))],
                      [float(sp.diff(f_expr, x1, x0)), float(sp.diff(f_expr, x1, 2))]]) / 2
        c = np.array([float(sp.diff(f_expr, x0).subs({x0: 0, x1: 0})),
                      float(sp.diff(f_expr, x1).subs({x0: 0, x1: 0}))])
        A = np.array([[c["a"], c["b"]] for c in self.constraints])
        b = np.array([c["c"] for c in self.constraints])

        # Начальная точка: строго внутри области
        x = np.array([0.1, 0.1])
        while not (A @ x <= b).all():
            x *= 0.5
        if not (A @ x <= b).all():
            x = np.array([0.0, 0.0])

        trajectory = [x.copy()]
        iterations_log = []

        # Лучшая точка и значение функции
        best_x = x.copy()
        best_f = self.f(best_x[0], best_x[1])

        # Метод внутренней точки
        mu = 1.0
        for i in range(self.max_iterations):
            # Слак-переменные: A x + s = b, s ≥ 0
            s = b - A @ x
            s = np.maximum(s, 1e-6)

            # Градиент и гессиан
            grad = 2 * Q @ x + c - A.T @ (mu / s)
            hess = 2 * Q + A.T @ np.diag(mu / (s ** 2)) @ A

            # Шаг Ньютона
            step = np.linalg.solve(hess, -grad)

            # Ограничиваем шаг
            alpha = 1.0
            x_new = x + step
            s_new = b - A @ x_new
            while not (A @ x_new <= b).all() or (x_new < 0).any() or (s_new <= 0).any():
                alpha *= 0.5
                x_new = x + alpha * step
                s_new = b - A @ x_new
                if alpha < 1e-6:
                    break

            x = np.maximum(x_new, 0)
            f_val = self.f(x[0], x[1])
            constraint_vals = [self.dop(x, j) for j in range(len(self.constraints))]
            iterations_log.append(f"Итерация {i}: x={x}, f(x)={f_val}, constraints={constraint_vals}")
            trajectory.append(x.copy())

            # Обновляем лучшую точку
            if f_val < best_f and (A @ x <= b + 1e-6).all():
                best_f = f_val
                best_x = x.copy()

            # Проверяем условия KKT для всех комбинаций активных ограничений
            active_constraints = [j for j in range(len(b)) if abs(s[j]) < 0.8]
            for active_set in [[]] + [[j] for j in range(len(b))] + [[0, 1]]:
                if active_set and not all(j in active_constraints for j in active_set):
                    continue
                A_active = A[active_set]
                b_active = b[active_set]
                if len(active_set) == 0:
                    x_kkt = np.linalg.solve(2 * Q, -c)
                else:
                    n = 2
                    m = A_active.shape[0]
                    KKT = np.zeros((n + m, n + m))
                    KKT[:n, :n] = 2 * Q
                    KKT[:n, n:] = A_active.T
                    KKT[n:, :n] = A_active
                    rhs = np.concatenate([-c, b_active])
                    try:
                        sol = np.linalg.solve(KKT, rhs)
                        x_kkt = sol[:n]
                    except np.linalg.LinAlgError:
                        continue

                if (A @ x_kkt <= b).all() and (x_kkt >= 0).all():
                    lambda_ = sol[n:] if len(active_set) > 0 else np.zeros(len(b))
                    if (lambda_ >= -1e-5).all():
                        x = x_kkt
                        f_val = self.f(x[0], x[1])
                        if f_val < best_f and (A @ x <= b + 1e-6).all():
                            best_f = f_val
                            best_x = x.copy()
                        break

            if np.linalg.norm(step) * alpha < 1e-6:
                break
            mu *= 0.001

        # Финальная корректировка: если точка близка к границе, минимизируем на границе
        s = b - A @ x
        for j in range(len(b)):
            if abs(s[j]) < 0.8:
                a, b_val, c = A[j][0], A[j][1], b[j]
                for x1 in np.linspace(0, c / a if a > 0 else 1, 1000):
                    x2 = (c - a * x1) / b_val
                    if x2 >= 0 and (A @ np.array([x1, x2]) <= b + 1e-6).all():
                        f_val = self.f(x1, x2)
                        if f_val < best_f:
                            best_f = f_val
                            best_x = np.array([x1, x2])

        final_point = best_x
        f_val = self.f(final_point[0], final_point[1])
        constraint_vals = [self.dop(final_point, i) for i in range(len(self.constraints))]
        iterations_log.append(f"Финальная итерация: x={final_point}, f(x)={f_val}")
        return final_point, trajectory, "QP завершён", iterations_log

    def run(self):
        if self.is_linear():
            return self.run_simplex()
        else:
            return self.run_qp()

class GeneticAlgorithm(OptimizationMethod):
    def run(self):
        population_size = self.kwargs.get("population_size", 200)
        generations = self.max_iterations  # Используем max_iterations как generations
        mutation_rate = self.kwargs.get("mutation_rate", 0.1)
        bounds = self.kwargs.get("bounds", (-2, 2))

        population = np.random.uniform(bounds[0], bounds[1], (population_size, 2))
        best_individual = None  # лучшее найденное решение
        best_fitness = float('inf')
        history = []

        for generation in range(generations):
            fitness = np.array([self.f(ind[0], ind[1]) for ind in population])  # оценка популяции
            current_best_idx = np.argmin(fitness)  #индекс лучшей особи

            if fitness[current_best_idx] < best_fitness:  #обновление лучшего решения
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()
                history.append((generation, best_individual, best_fitness))

            selected_indices = []
            for _ in range(population_size): # турнирный отбор
                candidates = np.random.choice(population_size, size=5, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                selected_indices.append(winner)
            selected_population = population[selected_indices]

            for i in range(0, population_size, 2):  # скрещивание
                if i + 1 < population_size:
                    parent1, parent2 = selected_population[i], selected_population[i + 1]
                    alpha = np.random.rand()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                    selected_population[i], selected_population[i + 1] = child1, child2

            mutation_strength = 0.1 * (0.99 ** generation) # мутация
            for i in range(population_size):
                if np.random.rand() < mutation_rate:
                    selected_population[i] += np.random.normal(0, mutation_strength, 2)
                    selected_population[i] = np.clip(selected_population[i], bounds[0], bounds[1])

            if best_individual is not None:  # замена худшей особи на лучшую
                worst_idx = np.argmax([self.f(ind[0], ind[1]) for ind in selected_population])
                selected_population[worst_idx] = best_individual

            population = selected_population

        trajectory = [ind for _, ind, _ in history]
        iterations_log = [f"Итерация {gen}: x={ind}, f(x)={fit}" for gen, ind, fit in history]
        return best_individual, trajectory, "Генетический алгоритм завершён", iterations_log


class ParticleSwarmOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.swarmsize = self.kwargs.get("swarmsize", 50)
        # Преобразуем списки в numpy массивы
        self.minvalues = np.array(self.kwargs.get("minvalues", [-5.12, -5.12]))
        self.maxvalues = np.array(self.kwargs.get("maxvalues", [5.12, 5.12]))
        self.current_velocity_ratio = self.kwargs.get("current_velocity_ratio", 0.5)
        self.local_velocity_ratio = self.kwargs.get("local_velocity_ratio", 2.0)  # Изменено с 2.0 на 2.05
        self.global_velocity_ratio = self.kwargs.get("global_velocity_ratio", 5.0)  # Изменено с 2.0 на 2.05
        # Проверяем условие (можно ослабить до >= 4, если это допустимо для алгоритма)
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"
        self.swarm = self._create_swarm()

    def _create_swarm(self):
        class Particle:
            def __init__(self, outer):
                self.position = np.random.rand(2) * (outer.maxvalues - outer.minvalues) + outer.minvalues
                self.velocity = np.random.rand(2) * (outer.maxvalues - outer.minvalues) - (
                            outer.maxvalues - outer.minvalues)
                self.best_position = self.position.copy()
                self.best_value = outer.f(self.position[0], self.position[1])

            def update(self, outer, global_best_position):
                rnd_local = np.random.rand(2)
                rnd_global = np.random.rand(2)
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
                value = outer.f(self.position[0], self.position[1])
                if value < self.best_value:
                    self.best_value = value
                    self.best_position = self.position.copy()

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
            f_val = self.f(global_best_position[0], global_best_position[1])
            iterations_log.append(f"Итерация {i}: x={global_best_position}, f(x)={f_val}")

        return global_best_position, trajectory, "PSO завершён", iterations_log

optimization_methods = {
    "Градиентный спуск": GradientDescent,
    "Квадратичный симплекс": QuadraticSimplex,
    "Генетический алгоритм": GeneticAlgorithm,
    "Рой частиц": ParticleSwarmOptimization
}