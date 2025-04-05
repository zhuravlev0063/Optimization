# optimization_methods/base.py
import numpy as np

class OptimizationMethod:
    def __init__(self, f, initial_point, max_iterations, **kwargs):
        self.f = f
        self.initial_point = np.array(initial_point, dtype=float) if initial_point is not None else None
        self.max_iterations = max_iterations
        self.kwargs = kwargs

    def run(self):
        raise NotImplementedError("Метод run должен быть реализован в подклассе")

def diff_manual(f, var_idx, vars_vals, h=1e-5):
    vars_vals_forward = vars_vals.copy()
    vars_vals_backward = vars_vals.copy()
    vars_vals_forward[var_idx] += h
    vars_vals_backward[var_idx] -= h
    return (f(*vars_vals_forward) - f(*vars_vals_backward)) / (2 * h)

def gradient_f(f, x1, x2):
    vars_vals = [x1, x2]
    df_dx1 = diff_manual(f, 0, vars_vals)
    df_dx2 = diff_manual(f, 1, vars_vals)
    return np.array([df_dx1, df_dx2])