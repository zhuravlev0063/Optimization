def f_quad_simple(x1, x2):
    """Квадратичная функция: 2x1² + x1x2 + x2²"""
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2

def f_himmelblau(x1, x2):
    """Функция Химмельблау: (x1² + x2 - 11)² + (x1 + x2² - 7)²"""
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def f_quad_constrained(x1, x2):
    """Квадратичная функция с ограничением: 2x1² + 2x1x2 + 2x2² - 4x1 - 6x2"""
    return 2 * x1**2 + 2 * x1 * x2 + 2 * x2**2 - 4 * x1 - 6 * x2

# Словарь функций для интерфейса
available_functions = {
    "Квадратичная простая": f_quad_simple,
    "Химмельблау": f_himmelblau,
    "Квадратичная с ограничением": f_quad_constrained
}