import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, QMessageBox, \
    QHBoxLayout, QTextEdit, QSlider, QDialog
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Функции
functions = {
    "f1": lambda x1, x2: 2 * x1 ** 2 + x1 * x2 + x2 ** 2,
    "f2": lambda x1, x2: np.sin(x1) + np.cos(x2),
    "f3": lambda x1, x2: x1 ** 2 - x2 ** 2
}


# Функция для приближенного вычисления производной
def diff_manual(f, var_idx, vars_vals, h=1e-5):
    vars_vals_forward = vars_vals.copy()
    vars_vals_backward = vars_vals.copy()

    vars_vals_forward[var_idx] += h
    vars_vals_backward[var_idx] -= h

    return (f(*vars_vals_forward) - f(*vars_vals_backward)) / (2 * h)


# Функция для вычисления градиента
def gradient_f(f, x1, x2):
    vars_vals = [x1, x2]
    df_dx1 = diff_manual(f, 0, vars_vals)
    df_dx2 = diff_manual(f, 1, vars_vals)
    return np.array([df_dx1, df_dx2])


# Метод градиентного спуска
def gradient_descent(f, learning_rate, max_iterations, initial_point, epsilon1, epsilon2):
    x = np.array(initial_point, dtype=float)
    trajectory = [x.copy()]
    k = 0
    iterations_log = []

    while k < max_iterations:
        grad = gradient_f(f, x[0], x[1])
        f_x = f(x[0], x[1])
        iterations_log.append(f"Итерация {k}: x={x}, f(x)={f_x}")

        if abs(f_x) < epsilon1:
            return x, trajectory, "|f(x*)| < epsilon1", iterations_log

        t_k = learning_rate
        x_next = x - t_k * grad
        f_x_next = f(x_next[0], x_next[1])

        if np.linalg.norm(x_next - x) < epsilon2 and abs(f_x_next - f_x) < epsilon2:
            return x_next, trajectory, "||xk+1 - xk|| < epsilon2", iterations_log

        x = x_next
        trajectory.append(x.copy())
        k += 1

    return x, trajectory, "Превышено число итераций", iterations_log


# Класс для окна с итерациями
class IterationWindow(QDialog):
    def __init__(self, iterations_log):
        super().__init__()
        self.setWindowTitle("Итерации градиентного спуска")
        self.setGeometry(150, 150, 400, 500)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText("\n".join(iterations_log))
        layout.addWidget(self.text_edit)
        self.setLayout(layout)


# Класс для отображения 3D-графика
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 6))
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)
        self.f = functions["f1"]

    def plot(self, f, final_point, trajectory, angle_x=30, angle_y=30, angle_z=30):
        self.ax.clear()
        x1_range = np.linspace(-2, 2, 400)
        x2_range = np.linspace(-2, 2, 400)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = f(X1, X2)

        self.ax.plot_surface(X1, X2, Z, cmap='ocean', edgecolor='none')
        self.ax.view_init(elev=angle_x, azim=angle_y)
        self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]), color='r', s=50,
                        label='Final point')
        trajectory = np.array(trajectory)
        self.ax.scatter(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]), color='red', s=10)
        self.ax.legend()
        self.draw()


# Основной класс приложения
class FunctionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор функции и метода")
        self.setGeometry(100, 100, 900, 600)

        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        self.function_selector = QComboBox()
        self.function_selector.addItems(functions.keys())
        control_layout.addWidget(QLabel("Выберите функцию:"))
        control_layout.addWidget(self.function_selector)

        self.run_button = QPushButton("Запустить")
        self.run_button.clicked.connect(self.run_calculation)
        control_layout.addWidget(self.run_button)

        self.iteration_button = QPushButton("Показать итерации")
        self.iteration_button.clicked.connect(self.show_iterations)
        control_layout.addWidget(self.iteration_button)

        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(0, 360)
        self.slider_x.setValue(30)
        self.slider_x.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Вращение X:"))
        control_layout.addWidget(self.slider_x)

        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 360)
        self.slider_y.setValue(30)
        self.slider_y.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Вращение Y:"))
        control_layout.addWidget(self.slider_y)

        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(0, 360)
        self.slider_z.setValue(30)
        self.slider_z.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Вращение Z:"))
        control_layout.addWidget(self.slider_z)

        self.plot_canvas = PlotCanvas(self)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.plot_canvas)

        self.setLayout(main_layout)

    def run_calculation(self):
        f = functions[self.function_selector.currentText()]
        final_point, trajectory, stop_reason, self.iterations_log = gradient_descent(f, 0.1, 100, [0.5, 1], 1e-3, 1e-3)
        self.plot_canvas.plot(f, final_point, trajectory, self.slider_x.value(), self.slider_y.value(),
                              self.slider_z.value())

    def update_plot(self):
        self.plot_canvas.plot(self.plot_canvas.f, [0, 0], [], self.slider_x.value(), self.slider_y.value(),
                              self.slider_z.value())

    def show_iterations(self):
        IterationWindow(self.iterations_log).exec()


# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FunctionSelector()
    window.show()
    sys.exit(app.exec())
