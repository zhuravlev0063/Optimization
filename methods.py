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
        super().__init__(f, None, max_iterations, **kwargs)  # initial_point=None
        self.x = sp.symbols('x0 x1')
        self.l = sp.symbols('l')
        self.v = [sp.symbols('v0'), sp.symbols('v1')]
        self.z = [sp.symbols('z0'), sp.symbols('z1')]
        self.w = sp.symbols('w')
        self.a = self.kwargs.get("a", 1.0)
        self.b = self.kwargs.get("b", 2.0)
        self.c = self.kwargs.get("c", 2.0)

    def dop(self, x):
        """Ограничение: ax0 + bx1 - c = 0"""
        return self.a * x[0] + self.b * x[1] - self.c

    def lagrange_function(self, x, l):
        """Функция Лагранжа"""
        return self.f(x[0], x[1]) + l * self.dop(x)

    def compute_derivatives(self):
        """Вычисление производных Лагранжиана"""
        L = self.lagrange_function(self.x, self.l)
        return sp.diff(L, self.x[0]), sp.diff(L, self.x[1]), sp.diff(L, self.l)

    def modify_derivatives(self, dL_dx0, dL_dx1, dL_dl): #добавляет переменные v0, v1 (возможно для условий неотрицательности) и z0,z1(слэк-переменные)
        """Модификация производных"""
        return (dL_dx0 - self.v[0] + self.z[0],
                dL_dx1 - self.v[1] + self.z[1],
                dL_dl + self.w)

    def modify_and_sum_derivatives(self, modified_dL_dx0, modified_dL_dx1): #суммирует первые производные с противопол знаком , убираем z0, z1(ставит их равным 0)
        """Сумма модифицированных производных"""
        return (-modified_dL_dx0.subs({self.z[0]: 0, self.z[1]: 0}) +
                -modified_dL_dx1.subs({self.z[0]: 0, self.z[1]: 0}))

    def reorder_coefficients(self, expression): #Помогают преобразовать символические выражения в числовые коэффициенты.
        """Упорядочивание коэффициентов"""
        expression = str(expression).replace(' ', '')
        terms = re.findall(r'[+-]?[\d]*\.?[\d]+\*?[a-zA-Z]+(?:\^?\d*)?|[+-]?[a-zA-Z]+\d*|[+-]?[\d]*\.?[\d]+', expression)
        t = not any(re.match(r'^-[\d]+(\.[\d]+)?$', term) for term in terms)

        coeffs = {'const': [], 'x0': [], 'x1': [], 'l': [], 'v0': [], 'v1': [], 'w': []}
        for term in terms:
            if re.search(r'[a-zA-Z]', term):
                sign = '-' if (t and not term.startswith('-')) or (not t and term.startswith('-')) else '+'
                term = sign + term.lstrip('+-')
                for key in coeffs:
                    if key in term:
                        coeffs[key].append(term)
                        break
            else:
                coeffs['const'].append('+' + term.lstrip('-') if term.startswith('-') else term)

        return [coeffs[k][0] if coeffs[k] else '+0' for k in ['const', 'x0', 'x1', 'l', 'v0', 'v1', 'w']]

    def extract_all_values(self, polynomial):
        """Извлечение числовых значений"""
        if isinstance(polynomial, list):
            polynomial = ''.join(polynomial)
        terms = re.findall(r'[+-]?[\d]*\.?[\d]*\*?[a-zA-Z]?\^?\d*', polynomial)
        values = []
        for term in terms:
            if not term:
                continue
            coef_match = re.match(r'([+-]?[\d]*\.?[\d]*)', term)
            if coef_match:
                coef = coef_match.group(0)
                values.append(1.0 if coef in ('', '+') else -1.0 if coef == '-' else float(coef))
        return values

    def vect(self): #Извлекает коэффициенты из суммы производных для вектора c (целевая функция симплекса).
        """Получение вектора коэффициентов"""
        dL_dx0, dL_dx1, dL_dl = self.compute_derivatives()
        mod_dx0, mod_dx1, _ = self.modify_derivatives(dL_dx0, dL_dx1, dL_dl)
        summed = self.modify_and_sum_derivatives(mod_dx0, mod_dx1)
        return self.extract_all_values(self.reorder_coefficients(summed))

    def extract_and_modify2(self): #Создаёт матрицу A из модифицированных производных — это ограничения симплекс-метода.
        """Получение матрицы ограничений"""
        dL_dx0, dL_dx1, dL_dl = self.compute_derivatives()
        mod_dx0, mod_dx1, mod_dl = self.modify_derivatives(dL_dx0, dL_dx1, dL_dl)
        return [self.extract_all_values(self.reorder_coefficients(str(mod_dx0))),
                self.extract_all_values(self.reorder_coefficients(str(mod_dx1))),
                self.extract_all_values(self.reorder_coefficients(str(mod_dl)))]

    def maxVal(self, a, c):
        """Поиск ведущего элемента: возвращает индексы строки и столбца"""
        up_f = max(c[1:], default=0)
        if up_f <= 0:
            return None, None
        max_index = c[1:].index(up_f) + 1
        max_val = [(a[i][0] / a[i][max_index], i) for i in range(len(a)) if a[i][max_index] > 0]
        if not max_val:
            return None, None
        min_ratio, min_index = min(max_val, key=lambda x: x[0])
        if min_ratio < 0:
            return None, None
        return min_index, max_index

    def ch_ab(self, a, m, c1):
        """Преобразование симплекс-таблицы"""
        id1, id2 = m
        el = a[id1][id2]
        if el == 0:
            return a, c1
        timeC = -c1[id2]
        for i in range(len(a[0])):
            a[id1][i] /= el
            c1[i] += a[id1][i] * timeC
        for i in range(len(a)):
            if i != id1:
                time = -a[i][id2]
                for j in range(len(a[0])):
                    a[i][j] += a[id1][j] * time
        return a, c1

    def run(self):
        """Запуск симплекс-метода"""
        c = self.vect()
        a = self.extract_and_modify2()
        d = [0, 0, 0]
        c1 = c.copy()

        iterations = 0
        tt = [7, 8, 6]  # Индексы базисных переменных
        trajectory = []
        iterations_log = []

        while iterations < self.max_iterations:
            m = self.maxVal(a, c1)
            if m[0] is None or m[1] is None:
                iterations_log.append("Программа завершена: либо зацикливание, либо оптимальное решение.")
                break

            id1, id2 = m
            tt[id1] = id2
            a, c1 = self.ch_ab(a, m, c1)
            d = [row[0] for row in a]

            result = [0.0] * 9
            for idx, val in zip(tt, d):
                result[idx - 1] = val
            x = result[0:2]
            f_val = self.f(x[0], x[1])
            trajectory.append(np.array(x))
            iterations_log.append(f"Итерация {iterations}: x={x}, f(x)={f_val}")

            if all(x <= 0 for x in c1):
                break

            iterations += 1

        result = [0.0] * 9
        for idx, val in zip(tt, d):
            result[idx - 1] = val
        final_point = np.array(result[0:2])
        f_val = self.f(final_point[0], final_point[1])
        trajectory.append(final_point)
        iterations_log.append(f"Финальная итерация: x={final_point}, f(x)={f_val}")
        return final_point, trajectory, "Симплекс завершён", iterations_log

optimization_methods = {
    "Градиентный спуск": GradientDescent,
    "Квадратичный симплекс": QuadraticSimplex
}