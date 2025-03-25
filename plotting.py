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
        self.f = available_functions["Квадратичная простая"]  # Устанавливаем функцию по умолчанию
        self.surface_data = None
        self.last_f = None

    def plot(self, f, final_point, trajectory, method_name):
        self.ax.clear()

        if f != self.last_f or self.surface_data is None:
            x1_range = np.linspace(-5, 5, 100)
            x2_range = np.linspace(-5, 5, 100)
            X1, X2 = np.meshgrid(x1_range, x2_range)
            Z = f(X1, X2)
            self.surface_data = (X1, X2, Z)
            self.last_f = f
        else:
            X1, X2, Z = self.surface_data

        self.ax.plot_surface(X1, X2, Z, cmap='ocean', edgecolor='none', alpha=0.7)

        trajectory = np.array(trajectory)
        if method_name == "Градиентный спуск":
            self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]),
                           color='r', s=50, label='Final point')
            self.ax.scatter(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]),
                           color='red', s=10)
        elif method_name == "Квадратичный симплекс":
            self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]),
                           color='g', s=50, label='Simplex final point')
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]),
                        color='green', linewidth=1)

        self.ax.legend()
        self.draw()