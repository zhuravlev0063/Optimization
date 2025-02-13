import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция f(x1, x2)
def f(x1, x2):
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2

# Градиент функции f(x1, x2)
def gradient_f(x1, x2):
    df_dx1 = 4 * x1 + x2
    df_dx2 = x1 + 2 * x2
    return np.array([df_dx1, df_dx2])

# Метод градиентного спуска
def gradient_descent(learning_rate, num_iterations, initial_point):
    x = np.array(initial_point)

    for _ in range(num_iterations):
        grad = gradient_f(x[0], x[1])
        x = x - learning_rate * grad

    return x

# Параметры градиентного спуска
learning_rate = 0.1  # Шаг обучения
num_iterations = 10  # Количество итераций (M=10)
initial_point = [0.5, 1]  # Начальная точка (x0 = (0.5, 1))

# Запуск метода градиентного спуска
final_point = gradient_descent(learning_rate, num_iterations, initial_point)

# Итоговые значения
final_value = f(final_point[0], final_point[1])

# Вывод итогов
print(f"Итоговые значения после {num_iterations} итераций:")
print(f"x1 = {final_point[0]:.4f}, x2 = {final_point[1]:.4f}")
print(f"f(x1, x2) = {final_value:.4f}")

# Для 3D-графика
x1_range = np.linspace(-2, 2, 400)
x2_range = np.linspace(-2, 2, 400)

# Сетка для осей
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = f(X1, X2)

# Создание графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

# Подписи осей
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')

# Выделяем финальную точку градиентного спуска на графике
ax.scatter(final_point[0], final_point[1], final_value, color='r', s=50, label='Final point')

# Добавление легенды
ax.legend()

# Отображение графика с возможностью вращения
plt.show()
