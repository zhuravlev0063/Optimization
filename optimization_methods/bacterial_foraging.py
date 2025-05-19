import numpy as np
from .base import OptimizationMethod

class BacterialForagingOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.num_bacteria = kwargs.get("num_bacteria", 50)  # Число бактерий
        self.chem_steps = kwargs.get("chem_steps", 100)     # Шаги хемотаксиса
        self.repro_steps = kwargs.get("repro_steps", 4)     # Шаги репродукции
        self.elim_steps = kwargs.get("elim_steps", 2)       # Шаги ликвидации
        self.step_size = kwargs.get("step_size", 0.1)       # Начальная величина шага
        self.elim_prob = kwargs.get("elim_prob", 0.25)      # Вероятность ликвидации
        self.elim_count = kwargs.get("elim_count", 10)      # Число ликвидируемых бактерий
        self.bounds_lower = kwargs.get("bounds_lower", -5)  # Нижняя граница
        self.bounds_upper = kwargs.get("bounds_upper", 5)   # Верхняя граница

        # Validate initial_point
        if isinstance(initial_point, (list, np.ndarray)):
            initial_point = np.array(initial_point, dtype=float)
            if initial_point.shape == (2,):  # Valid 2D point
                self.initial_point = initial_point
            else:
                raise ValueError(f"initial_point must be a 2D array/list, got shape {initial_point.shape}")
        else:
            # Fallback for scalar or invalid input
            print(f"Warning: Invalid initial_point {initial_point}. Using random 2D point within bounds.")
            self.initial_point = np.random.uniform(self.bounds_lower, self.bounds_upper, 2)

    class Bacterium:
        def __init__(self, outer):
            self.dim = 2  # 2D пространство
            self.bounds = [outer.bounds_lower, outer.bounds_upper]
            self.position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            self.health = 0

        def update_health(self, fitness):
            self.health += fitness

    def run(self):
        bacteria = [self.Bacterium(self) for _ in range(self.num_bacteria)]
        # Initialize best_position with validated initial_point
        best_position = self.initial_point.copy()
        best_fitness = self.f(best_position[0], best_position[1])
        trajectory = [best_position.copy()]
        iterations_log = [f"Начальная точка: x={best_position}, f(x)={best_fitness}"]

        for l in range(self.elim_steps):
            # Проверка на четное количество бактерий
            if self.num_bacteria % 2 != 0:
                iterations_log.append(f"Программа завершена: четное количество бактерий ({self.num_bacteria})")
                return best_position, trajectory, "BFO завершён из-за четного числа бактерий", iterations_log

            for r in range(self.repro_steps):
                for t in range(self.chem_steps):
                    current_step_size = self.step_size / (t + 1)  # Уменьшение шага
                    for bacterium in bacteria:
                        current_fitness = self.f(bacterium.position[0], bacterium.position[1])
                        bacterium.update_health(current_fitness)

                        direction = np.random.uniform(-1, 1, 2)  # случайное направление
                        direction = direction / np.linalg.norm(direction)  # нормализуем вектор
                        new_position = bacterium.position + current_step_size * direction  # новая позиция
                        new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)  # проверяем новую позицию в пределах
                        new_fitness = self.f(new_position[0], new_position[1])

                        # Плавание: до 3 шагов при улучшении
                        if new_fitness < current_fitness:  # сравниваем
                            bacterium.position = new_position
                            for _ in range(2):
                                new_position = bacterium.position + current_step_size * direction
                                new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)
                                if self.f(new_position[0], new_position[1]) >= new_fitness:
                                    break
                                bacterium.position = new_position
                                new_fitness = self.f(new_position[0], new_position[1])
                        else:
                            # Кувырок
                            direction = np.random.uniform(-1, 1, 2)
                            direction = direction / np.linalg.norm(direction)
                            bacterium.position = bacterium.position + current_step_size * direction
                            bacterium.position = np.clip(bacterium.position, self.bounds_lower, self.bounds_upper)

                # Репродукция
                bacteria.sort(key=lambda b: b.health)  # сортируем по возрастанию здоровья
                survivors = bacteria[:self.num_bacteria // 2]  # выживает лучшая половина бактерий
                bacteria = survivors + [self.Bacterium(self) for _ in range(self.num_bacteria // 2)]  # остальные заменяются на другие
                for i in range(self.num_bacteria // 2):
                    bacteria[self.num_bacteria // 2 + i].position = survivors[i].position.copy()  # но заменяются с другой позицией (с позицией выжившей)

            # Ликвидация и рассеивание
            elim_indices = np.random.choice(self.num_bacteria, self.elim_count, replace=False)  # берем elim_count рандомных бактерий
            for i in elim_indices:
                if np.random.random() < self.elim_prob:  # заменяем позицию на новую случайную с вероятностью elim_prob
                    bacteria[i] = self.Bacterium(self)

            # Обновление лучшего решения
            current_best = min(bacteria, key=lambda b: self.f(b.position[0], b.position[1]))
            current_fitness = self.f(current_best.position[0], current_best.position[1])
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_position = current_best.position.copy()
            trajectory.append(best_position.copy())
            iterations_log.append(f"Итерация {l}: x={best_position}, f(x)={best_fitness}")

        return best_position, trajectory, "BFO завершён", iterations_log