import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, QMessageBox, \
    QHBoxLayout, QTextEdit, QSlider
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def diff_manual(f, var_idx, vars_vals, h=1e-5):
    vars_vals_forward = vars_vals.copy()
    vars_vals_backward = vars_vals.copy()
    vars_vals_forward[var_idx] += h
    vars_vals_backward[var_idx] -= h
    return (f(*vars_vals_forward) - f(*vars_vals_backward)) / (2 * h)


def f(x1, x2):
    return 2 * x1 ** 2 + x1 * x2 + x2 ** 2


def gradient_f(x1, x2):
    vars_vals = [x1, x2]
    df_dx1 = diff_manual(f, 0, vars_vals)
    df_dx2 = diff_manual(f, 1, vars_vals)
    return np.array([df_dx1, df_dx2])


def gradient_descent(learning_rate, num_iterations, initial_point, epsilon1=1e-5, epsilon2=1e-5):
    x = np.array(initial_point, dtype=float)
    trajectory = [x.copy()]

    for k in range(num_iterations):
        grad = gradient_f(x[0], x[1])
        grad_norm = np.linalg.norm(grad)

        if grad_norm < epsilon1:
            break

        new_x = x - learning_rate * grad

        if np.linalg.norm(new_x - x) < epsilon2:
            break

        if abs(f(*new_x) - f(*x)) < epsilon2:
            break

        x = new_x
        trajectory.append(x.copy())

    return x, trajectory


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 7))
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, final_point, trajectory, angle_x=30, angle_y=30, angle_z=30):
        self.ax.clear()
        x1_range = np.linspace(-2, 2, 400)
        x2_range = np.linspace(-2, 2, 400)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = f(X1, X2)

        self.ax.plot_surface(X1, X2, Z, cmap='ocean', edgecolor='none')
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_zlabel('f(x1, x2)')
        self.ax.view_init(elev=angle_x, azim=angle_y)
        self.ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]), color='r', s=50,
                        label='Final point')

        trajectory = np.array(trajectory)
        self.ax.scatter(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]), color='red', s=10)
        self.ax.legend()
        self.draw()


class FunctionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор функции и метода")
        self.setGeometry(100, 100, 800, 600)
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        self.function_label = QLabel("Выберите функцию:")
        control_layout.addWidget(self.function_label)
        self.function_combo = QComboBox()
        self.function_combo.addItems(["f(x1, x2)=2 * x1 ** 2 + x1 * x2 + x2 ** 2", "Другая функция"])
        control_layout.addWidget(self.function_combo)

        self.method_label = QLabel("Выберите метод:")
        control_layout.addWidget(self.method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Градиентный спуск", "Другой метод"])
        control_layout.addWidget(self.method_combo)

        self.run_button = QPushButton("Запустить")
        self.run_button.clicked.connect(self.run_calculation)
        control_layout.addWidget(self.run_button)

        self.points_display = QTextEdit()
        self.points_display.setReadOnly(True)
        control_layout.addWidget(self.points_display)

        self.minimum_display = QTextEdit()
        self.minimum_display.setReadOnly(True)
        control_layout.addWidget(self.minimum_display)

        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(0, 360)
        self.slider_x.setValue(30)
        self.slider_x.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Вращение вокруг оси X:"))
        control_layout.addWidget(self.slider_x)

        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 360)
        self.slider_y.setValue(30)
        self.slider_y.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Вращение вокруг оси Y:"))
        control_layout.addWidget(self.slider_y)

        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(0, 360)
        self.slider_z.setValue(30)
        self.slider_z.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Вращение вокруг оси Z:"))
        control_layout.addWidget(self.slider_z)

        main_layout.addLayout(control_layout)
        self.plot_canvas = PlotCanvas(self)
        main_layout.addWidget(self.plot_canvas)
        self.setLayout(main_layout)

        self.final_point = None
        self.trajectory = None

    def run_calculation(self):
        if self.function_combo.currentText() == "f(x1, x2)=2 * x1 ** 2 + x1 * x2 + x2 ** 2" and self.method_combo.currentText() == "Градиентный спуск":
            self.final_point, self.trajectory = gradient_descent(0.1, 10, [0.5, 1])
            self.plot_canvas.plot(self.final_point, self.trajectory, angle_x=self.slider_x.value(),
                                  angle_y=self.slider_y.value(), angle_z=self.slider_z.value())
        else:
            QMessageBox.information(self, "Информация", "Выбрана функция или метод, который еще не реализован.")

    def update_plot(self):
        if self.final_point is not None and self.trajectory is not None:
            self.plot_canvas.plot(self.final_point, self.trajectory, angle_x=self.slider_x.value(),
                                  angle_y=self.slider_y.value(), angle_z=self.slider_z.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FunctionSelector()
    window.show()
    sys.exit(app.exec())
