import numpy as np

def f_quad_simple(x1, x2):
    """Квадратичная функция: 2x1² + x1x2 + x2²"""
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2

def f_himmelblau(x1, x2):
    """Функция Химмельблау: (x1² + x2 - 11)² + (x1 + x2² - 7)²"""
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def f_quad_constraineded(x1, x2):
    """Квадратичная функция с ограничением: 2x1² + 2x1x2 + 2x2² - 4x1 - 6x2"""
    return 2 * x1**2 + 3 * x2 **2 + 4 * x1*x2 - 6 * x1 - 3 * x2

def f_quad_constrained(x1, x2):
    """Квадратичная функция с ограничением: 2x1² + 2x1x2 + 2x2² - 4x1 - 6x2"""
    return 2 * x1**2 + 2 * x1 * x2 + 2 * x2**2 - 4 * x1 - 6 * x2

def f_quad_constraineds(x1, x2):
    """Квадратичная функция с ограничением: 2x1² + 2x1x2 + 2x2² - 4x1 - 6x2"""
    return x1**2 - 10 * x1 + x2**2 - 20 * x2 + 125

def f_linear(x1, x2):
    return -2 * x1 - 3 * x2  # Максимизация прибыли, например

def f_linearsss(x1, x2):
    return -6 * x1 - 3 * x2 + 0.5*x1**2 + x1*x2 + x2**2  # Максимизация прибыли, например

def rosenbrock(x1, x2):
    """Функция Розенброка с минимумом в точке (1, 1)"""
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

# Словарь функций для интерфейса
available_functions = {
    "Квадратичная простая": f_quad_simple,
    "Химмельблау": f_himmelblau,
    "Линейная": f_linear,
    "Линейнаяda": f_linearsss,
    "Квадратичная с ограничением": f_quad_constrained,
    "Квадратичная с ограничениями": f_quad_constraineded,
    "Квадратичная с ограничениями c": f_quad_constraineds,
    "Розенброк": rosenbrock
}