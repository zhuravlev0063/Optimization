import numpy as np
from .base import OptimizationMethod

class ParticleSwarmOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.swarmsize = self.kwargs.get("swarmsize", 50) # размер роя (частиц)
        # Преобразуем списки в numpy массивы
        self.minvalues = np.array(self.kwargs.get("minvalues", [-5.12, -5.12]))
        self.maxvalues = np.array(self.kwargs.get("maxvalues", [5.12, 5.12]))
        self.current_velocity_ratio = self.kwargs.get("current_velocity_ratio", 0.5) # (инерция) — насколько частица сохраняет свою текущую скорость.
        self.local_velocity_ratio = self.kwargs.get("local_velocity_ratio", 2.0)  # насколько сильно частица тянется к своей личной лучшей точке.
        self.global_velocity_ratio = self.kwargs.get("global_velocity_ratio", 5.0) #  насколько сильно частица тянется к глобальной лучшей точке.
        # Проверяем условие (можно ослабить до >= 4, если это допустимо для алгоритма)
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"
        self.swarm = self._create_swarm() #вызов метода для создания роя частиц

    def _create_swarm(self):
        class Particle:
            def __init__(self, outer):
                self.position = np.random.rand(2) * (outer.maxvalues - outer.minvalues) + outer.minvalues #кординаты частицы
                self.velocity = np.random.rand(2) * (outer.maxvalues - outer.minvalues) - (outer.maxvalues - outer.minvalues) #скорость частицы
                self.best_position = self.position.copy()  #запоминаем начальную позицию как лучшую
                self.best_value = outer.f(self.position[0], self.position[1]) #вычислем целевую функцию

            def update(self, outer, global_best_position):
                rnd_local = np.random.rand(2)
                rnd_global = np.random.rand(2)  # генерируем случайные числа для случайного компонента вектора скорости
                velo_ratio = outer.local_velocity_ratio + outer.global_velocity_ratio #сумма коэф притяжения
                common_ratio = 2.0 * outer.current_velocity_ratio / abs(
                    2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))  #коэффициент сходимости по Кларку и Кеннеди

                new_velocity = (common_ratio * self.velocity +
                                common_ratio * outer.local_velocity_ratio * rnd_local * (
                                            self.best_position - self.position) +
                                common_ratio * outer.global_velocity_ratio * rnd_global * (
                                            global_best_position - self.position))   #Обновление скорости

                self.velocity = new_velocity
                self.position += self.velocity #Обновление позиции
                value = outer.f(self.position[0], self.position[1])  #Если новая позиция лучше — сохраняем её как новую лучшую.
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
                particle.update(self, global_best_position) #Обновляем каждую частицу.
                if particle.best_value < global_best_value:  #Если какая-то из них нашла новую лучшую точку
                    global_best_value = particle.best_value    # то обновляем глобальный минимум.
                    global_best_position = particle.best_position.copy()

            trajectory.append(global_best_position.copy())
            f_val = self.f(global_best_position[0], global_best_position[1])
            iterations_log.append(f"Итерация {i}: x={global_best_position}, f(x)={f_val}")

        return global_best_position, trajectory, "PSO завершён", iterations_log
