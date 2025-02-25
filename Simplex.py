import numpy as np
from scipy.optimize import linprog
from sympy import symbols, diff, lambdify


def extract_coefficients(func, vars):
    """
    Извлекает коэффициенты для квадратичной функции от двух переменных.

    :param func: Функция, от которой нужно извлечь коэффициенты
    :param vars: Переменные, по которым будем вычислять производные
    :return: Коэффициенты для целевой функции
    """
    # Определяем символические переменные
    x1, x2 = symbols('x1 x2')

    # Вычисляем частные производные функции по каждой переменной
    grad = [diff(func(x1, x2), var) for var in [x1, x2]]

    # Преобразуем символические выражения в числовые функции
    grad_func = [lambdify((x1, x2), g) for g in grad]

    # Преобразуем выражения в массив коэффициентов
    grad_vals = np.array([grad_func[0](0, 0), grad_func[1](0, 0)])  # Это для проверки
    print("Первые частные производные при (x1, x2) = (0, 0):", grad_vals)

    # Для коэффициентов квадратичной функции (вторые производные) используем тот же принцип
    hess = [[diff(grad[0], x1), diff(grad[0], x2)],
            [diff(grad[1], x1), diff(grad[1], x2)]]

    # Преобразуем их в функции
    hess_func = [[lambdify((x1, x2), h) for h in row] for row in hess]

    # Получаем числовые значения коэффициентов
    hess_vals = np.array([[hess_func[i][j](0, 0) for j in range(2)] for i in range(2)])
    print("Гессиан (матрица вторых производных):", hess_vals)

    return grad_vals, hess_vals


def solve_function(func, constraints):
    """
    Решает задачу минимизации квадратичной функции с заданными ограничениями.

    :param func: Целевая функция в виде lambda
    :param constraints: Ограничения в виде списка: [(A_ub, b_ub), (bounds)]
    :return: Результаты оптимизации
    """
    # Генерация коэффициентов для целевой функции
    grad_vals, hess_vals = extract_coefficients(func, ['x1', 'x2'])

    # Ограничения
    A_ub, b_ub = constraints[0]
    bounds = constraints[1]

    # Для решения задачи симплексом, нам нужно представить задачу линейной
    # Преобразуем коэффициенты целевой функции в формат для линейного программирования
    c = np.concatenate([grad_vals, np.zeros_like(grad_vals)])

    # Решаем задачу с использованием симплекс-метода
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')

    if res.success:
        print("Оптимальное решение найдено:")
        print(f"Значение переменных x: {res.x}")
        print(f"Минимальное значение целевой функции: {res.fun}")
    else:
        print("Не удалось найти оптимальное решение")


# Пример использования

# Целевая функция: f(x1, x2) = x1^2 + 2*x1*x2 + x2^2 - 4*x1 - 6*x2
# Просто задаем ее как lambda-функцию
f = lambda x1, x2: x1 ** 2 + 2 * x1 * x2 + x2 ** 2 - 4 * x1 - 6 * x2

# Ограничения:
# Ограничение 1: x1 + x2 <= 3
A_ub = [[1, 1]]
b_ub = [3]

# Ограничения для переменных:
bounds = [(0, None), (0, None)]  # x1 >= 0, x2 >= 0

# Решаем задачу
solve_function(f, [(A_ub, b_ub), bounds])
