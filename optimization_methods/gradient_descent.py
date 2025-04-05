# optimization_methods/gradient_descent.py
import numpy as np
from .base import OptimizationMethod, gradient_f

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