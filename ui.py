from PySide6.QtWidgets import QWidget, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PySide6.QtCore import Qt
from functions import available_functions
from methods import optimization_methods
from plotting import PlotCanvas
from iteration_window import IterationWindow

class FunctionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор функции и метода")
        self.setGeometry(100, 100, 900, 600)
        self.iterations_log = []  # Инициализация iterations_log

        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)

        # Выбор функции
        self.function_selector = QComboBox()
        self.function_selector.addItems(available_functions.keys())
        control_layout.addWidget(QLabel("Выберите функцию:"))
        control_layout.addWidget(self.function_selector)

        # Выбор метода
        self.method_combo = QComboBox()
        self.method_combo.addItems(optimization_methods.keys())
        self.method_combo.currentTextChanged.connect(self.update_params_visibility)
        control_layout.addWidget(QLabel("Выберите метод:"))
        control_layout.addWidget(self.method_combo)

        # Поля ввода для градиентного спуска
        self.gradient_params = QVBoxLayout()
        self.initial_x0 = QLineEdit("0.5")
        self.initial_x1 = QLineEdit("1.0")
        self.max_iter = QLineEdit("100")
        self.epsilon1 = QLineEdit("0.001")
        self.epsilon2 = QLineEdit("0.001")
        self.learning_rate = QLineEdit("0.1")

        self.gradient_params.addWidget(QLabel("Начальная точка x0:"))
        self.gradient_params.addWidget(self.initial_x0)
        self.gradient_params.addWidget(QLabel("Начальная точка x1:"))
        self.gradient_params.addWidget(self.initial_x1)
        self.gradient_params.addWidget(QLabel("Максимум итераций:"))
        self.gradient_params.addWidget(self.max_iter)
        self.gradient_params.addWidget(QLabel("Эпсилон 1:"))
        self.gradient_params.addWidget(self.epsilon1)
        self.gradient_params.addWidget(QLabel("Эпсилон 2:"))
        self.gradient_params.addWidget(self.epsilon2)
        self.gradient_params.addWidget(QLabel("Шаг (learning rate):"))
        self.gradient_params.addWidget(self.learning_rate)
        control_layout.addLayout(self.gradient_params)

        # Поля ввода для симплекса
        self.simplex_params = QVBoxLayout()
        self.simplex_initial_x0 = QLineEdit("0.5")
        self.simplex_initial_x1 = QLineEdit("1.0")
        self.simplex_max_iter = QLineEdit("10")
        self.simplex_a = QLineEdit("1")
        self.simplex_b = QLineEdit("2")
        self.simplex_c = QLineEdit("2")

        self.simplex_params.addWidget(QLabel("Начальная точка x0:"))
        self.simplex_params.addWidget(self.simplex_initial_x0)
        self.simplex_params.addWidget(QLabel("Начальная точка x1:"))
        self.simplex_params.addWidget(self.simplex_initial_x1)
        self.simplex_params.addWidget(QLabel("Максимум итераций:"))
        self.simplex_params.addWidget(self.simplex_max_iter)
        self.simplex_params.addWidget(QLabel("Ограничение: a*x0 + b*x1 = c"))
        self.simplex_params.addWidget(QLabel("Коэффициент a:"))
        self.simplex_params.addWidget(self.simplex_a)
        self.simplex_params.addWidget(QLabel("Коэффициент b:"))
        self.simplex_params.addWidget(self.simplex_b)
        self.simplex_params.addWidget(QLabel("Свободный член c:"))
        self.simplex_params.addWidget(self.simplex_c)
        control_layout.addLayout(self.simplex_params)
        self.update_params_visibility()

        # Кнопки
        self.run_button = QPushButton("Запустить")
        self.run_button.clicked.connect(self.run_calculation)
        control_layout.addWidget(self.run_button)

        self.iteration_button = QPushButton("Показать итерации")
        self.iteration_button.clicked.connect(self.show_iterations)
        control_layout.addWidget(self.iteration_button)

        control_layout.addStretch()

        self.plot_canvas = PlotCanvas(self)
        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.plot_canvas, 3)

        self.setLayout(main_layout)

    def update_params_visibility(self):
        method = self.method_combo.currentText()
        is_gradient = method == "Градиентный спуск"
        for i in range(self.gradient_params.count()):
            self.gradient_params.itemAt(i).widget().setVisible(is_gradient)
        for i in range(self.simplex_params.count()):
            self.simplex_params.itemAt(i).widget().setVisible(not is_gradient)

    def run_calculation(self):
        f = available_functions[self.function_selector.currentText()]
        method_name = self.method_combo.currentText()
        method_class = optimization_methods[method_name]

        if method_name == "Градиентный спуск":
            try:
                initial_point = [float(self.initial_x0.text()), float(self.initial_x1.text())]
                max_iter = int(self.max_iter.text())
                epsilon1 = float(self.epsilon1.text())
                epsilon2 = float(self.epsilon2.text())
                learning_rate = float(self.learning_rate.text())
                method = method_class(f, initial_point, max_iter,
                                    learning_rate=learning_rate, epsilon1=epsilon1, epsilon2=epsilon2)
            except ValueError as e:
                print(f"Ошибка ввода параметров: {e}")
                return
        else:  # Квадратичный симплекс
            try:
                initial_point = [float(self.simplex_initial_x0.text()), float(self.simplex_initial_x1.text())]
                max_iter = int(self.simplex_max_iter.text())
                a = float(self.simplex_a.text())
                b = float(self.simplex_b.text())
                c = float(self.simplex_c.text())
                method = method_class(f, initial_point, max_iter, a=a, b=b, c=c)
            except ValueError as e:
                print(f"Ошибка ввода параметров: {e}")
                return

        final_point, trajectory, stop_reason, self.iterations_log = method.run()
        self.plot_canvas.plot(f, final_point, trajectory, method_name)

    def show_iterations(self):
        IterationWindow(self.iterations_log).exec()