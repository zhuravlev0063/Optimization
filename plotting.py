import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functions import available_functions

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 6))
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)
        self.f = available_functions["Простая квадратичная"]  # Устанавливаем функцию по умолчанию
        self.surface_data = None
        self.last_f = None

    def plot(self, f, final_point, trajectory, method_name, constraints=None):
        self.ax.clear()

        # Определяем границы в зависимости от функции
        if f.__name__ == "rosenbrock":
            x1_range = np.linspace(-2, 2, 100)
            x2_range = np.linspace(-1, 3, 100)
        else:
            x1_range = np.linspace(-5, 5, 100)
            x2_range = np.linspace(-5, 5, 100)

        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = np.array([[f(xi, yi) for xi in x1_range] for yi in x2_range])
        self.surface_data = (X1, X2, Z)
        self.last_f = f

        self.ax.plot_surface(X1, X2, Z, cmap='ocean', edgecolor='none', alpha=0.7)

        # Траектория
        trajectory = np.array(trajectory)
        if method_name == "Градиентный спуск":
            self.ax.scatter(trajectory[:, 0], trajectory[:, 1], [f(x, y) for x, y in trajectory],
                           color='red', s=10)
            self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]),
                           color='r', s=50, label='Final point (Gradient Descent)')
        elif method_name == "Квадратичный симплекс":
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], [f(x, y) for x, y in trajectory],
                        color='green', linewidth=1)
            self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]),
                           color='g', s=50, label='Final point (Quadratic Simplex)')
        elif method_name == "Генетический алгоритм":
            self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]),
                            color='b', s=50, label='Genetic final point')
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], [f(x, y) for x, y in trajectory],
                         color='blue', linewidth=1, label='Trajectory')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('f(X, Y)')
        self.ax.set_title(f'Оптимизация: {method_name}')
        self.ax.legend()
        self.draw()