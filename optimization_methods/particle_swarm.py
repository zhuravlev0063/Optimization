import numpy as np
from .base import OptimizationMethod

class ParticleSwarmOptimization(OptimizationMethod):
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        super().__init__(f, initial_point, max_iterations, **kwargs)
        self.swarmsize = self.kwargs.get("swarmsize", 50)
        self.minvalues = np.array(self.kwargs.get("minvalues", [-5.12, -5.12]))
        self.maxvalues = np.array(self.kwargs.get("maxvalues", [5.12, 5.12]))
        self.bounds = np.array([self.minvalues[0], self.maxvalues[0]])
        self.dimension = len(self.minvalues) if len(self.minvalues) > 1 else self.kwargs.get("dimension", 2)
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
                    self.position = np.random.rand(outer.dimension) * (outer.maxvalues - outer.minvalues) + outer.minvalues
                else:
                    self.position = position.copy()
                self.velocity = np.random.rand(outer.dimension) * (outer.maxvalues - outer.minvalues) - (outer.maxvalues - outer.minvalues)
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
                self.position = np.clip(self.position, outer.minvalues, outer.maxvalues)
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