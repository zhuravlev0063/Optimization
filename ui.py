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
        self.iterations_log = []

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
        self.simplex_max_iter = QLineEdit("10")
        self.constraints_layout = QVBoxLayout()
        self.constraints = []
        self.add_constraint()
        self.simplex_params.addWidget(QLabel("Максимум итераций:"))
        self.simplex_params.addWidget(self.simplex_max_iter)
        self.simplex_params.addWidget(QLabel("Ограничения: a*x0 + b*x1 ≤ c"))
        self.simplex_params.addLayout(self.constraints_layout)
        self.add_constraint_button = QPushButton("Добавить ограничение")
        self.add_constraint_button.clicked.connect(self.add_constraint)
        self.simplex_params.addWidget(self.add_constraint_button)
        control_layout.addLayout(self.simplex_params)

        # Поля ввода для генетического алгоритма
        self.genetic_params = QVBoxLayout()
        self.genetic_max_iter = QLineEdit("100")
        self.population_size = QLineEdit("200")
        self.mutation_rate = QLineEdit("0.1")
        self.bounds_lower = QLineEdit("-2")
        self.bounds_upper = QLineEdit("2")
        self.genetic_params.addWidget(QLabel("Максимум итераций (поколений):"))
        self.genetic_params.addWidget(self.genetic_max_iter)
        self.genetic_params.addWidget(QLabel("Размер популяции:"))
        self.genetic_params.addWidget(self.population_size)
        self.genetic_params.addWidget(QLabel("Вероятность мутации:"))
        self.genetic_params.addWidget(self.mutation_rate)
        self.genetic_params.addWidget(QLabel("Нижняя граница области:"))
        self.genetic_params.addWidget(self.bounds_lower)
        self.genetic_params.addWidget(QLabel("Верхняя граница области:"))
        self.genetic_params.addWidget(self.bounds_upper)
        control_layout.addLayout(self.genetic_params)

        # Поля ввода для роя частиц
        self.pso_params = QVBoxLayout()
        self.pso_swarmsize = QLineEdit("3000")
        self.pso_max_iter = QLineEdit("1000")
        self.pso_current_velocity = QLineEdit("0.3")
        self.pso_local_velocity = QLineEdit("2.0")
        self.pso_global_velocity = QLineEdit("5.0")
        self.pso_min_bound = QLineEdit("-5.12")
        self.pso_max_bound = QLineEdit("5.12")
        self.pso_params.addWidget(QLabel("Размер роя:"))
        self.pso_params.addWidget(self.pso_swarmsize)
        self.pso_params.addWidget(QLabel("Максимум итераций:"))
        self.pso_params.addWidget(self.pso_max_iter)
        self.pso_params.addWidget(QLabel("Инерция (current velocity):"))
        self.pso_params.addWidget(self.pso_current_velocity)
        self.pso_params.addWidget(QLabel("Локальный коэффициент:"))
        self.pso_params.addWidget(self.pso_local_velocity)
        self.pso_params.addWidget(QLabel("Глобальный коэффициент:"))
        self.pso_params.addWidget(self.pso_global_velocity)
        self.pso_params.addWidget(QLabel("Нижняя граница:"))
        self.pso_params.addWidget(self.pso_min_bound)
        self.pso_params.addWidget(QLabel("Верхняя граница:"))
        self.pso_params.addWidget(self.pso_max_bound)
        control_layout.addLayout(self.pso_params)

        # Поля ввода для алгоритма пчел
        self.bees_params = QVBoxLayout()
        self.bees_max_iter = QLineEdit("1000")
        self.bees_scoutbeecount = QLineEdit("300")
        self.bees_selectedbeecount = QLineEdit("10")
        self.bees_bestbeecount = QLineEdit("30")
        self.bees_selsitescount = QLineEdit("15")
        self.bees_bestsitescount = QLineEdit("5")
        self.bees_range_lower = QLineEdit("-5.12")
        self.bees_range_upper = QLineEdit("5.12")
        self.bees_range_shrink = QLineEdit("0.98")
        self.bees_max_stagnation = QLineEdit("10")
        self.bees_params.addWidget(QLabel("Максимум итераций:"))
        self.bees_params.addWidget(self.bees_max_iter)
        self.bees_params.addWidget(QLabel("Количество разведчиков:"))
        self.bees_params.addWidget(self.bees_scoutbeecount)
        self.bees_params.addWidget(QLabel("Пчел на перспективные участки:"))
        self.bees_params.addWidget(self.bees_selectedbeecount)
        self.bees_params.addWidget(QLabel("Пчел на лучшие участки:"))
        self.bees_params.addWidget(self.bees_bestbeecount)
        self.bees_params.addWidget(QLabel("Количество перспективных участков:"))
        self.bees_params.addWidget(self.bees_selsitescount)
        self.bees_params.addWidget(QLabel("Количество лучших участков:"))
        self.bees_params.addWidget(self.bees_bestsitescount)
        self.bees_params.addWidget(QLabel("Нижняя граница поиска:"))
        self.bees_params.addWidget(self.bees_range_lower)
        self.bees_params.addWidget(QLabel("Верхняя граница поиска:"))
        self.bees_params.addWidget(self.bees_range_upper)
        self.bees_params.addWidget(QLabel("Коэффициент сужения:"))
        self.bees_params.addWidget(self.bees_range_shrink)
        self.bees_params.addWidget(QLabel("Максимум итераций без улучшения:"))
        self.bees_params.addWidget(self.bees_max_stagnation)
        control_layout.addLayout(self.bees_params)

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
        self.update_params_visibility()

    def add_constraint(self):
        constraint_layout = QHBoxLayout()
        a_input = QLineEdit("1")
        b_input = QLineEdit("2")
        c_input = QLineEdit("2")
        constraint_layout.addWidget(QLabel("a:"))
        constraint_layout.addWidget(a_input)
        constraint_layout.addWidget(QLabel("b:"))
        constraint_layout.addWidget(b_input)
        constraint_layout.addWidget(QLabel("c:"))
        constraint_layout.addWidget(c_input)
        remove_button = QPushButton("Удалить")
        remove_button.clicked.connect(lambda: self.remove_constraint(constraint_layout))
        constraint_layout.addWidget(remove_button)
        self.constraints_layout.addLayout(constraint_layout)
        self.constraints.append({"a": a_input, "b": b_input, "c": c_input, "layout": constraint_layout})

    def remove_constraint(self, layout):
        for i, constraint in enumerate(self.constraints):
            if constraint["layout"] == layout:
                self.constraints_layout.removeItem(layout)
                for item in [layout.itemAt(j).widget() for j in range(layout.count())]:
                    item.deleteLater()
                del self.constraints[i]
                break

    def update_params_visibility(self):
        method = self.method_combo.currentText()
        is_gradient = method == "Градиентный спуск"
        is_simplex = method == "Квадратичный симплекс"
        is_genetic = method == "Генетический алгоритм"
        is_pso = method == "Рой частиц"
        is_bees = method == "Алгоритм пчел"

        for i in range(self.gradient_params.count()):
            item = self.gradient_params.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_gradient)

        for i in range(self.simplex_params.count()):
            item = self.simplex_params.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_simplex)
            elif item.layout():
                layout = item.layout()
                for j in range(layout.count()):
                    sub_item = layout.itemAt(j)
                    if sub_item.widget():
                        sub_item.widget().setVisible(is_simplex)
                    elif sub_item.layout():
                        sub_layout = sub_item.layout()
                        for k in range(sub_layout.count()):
                            if sub_layout.itemAt(k).widget():
                                sub_layout.itemAt(k).widget().setVisible(is_simplex)

        for i in range(self.genetic_params.count()):
            item = self.genetic_params.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_genetic)

        for i in range(self.pso_params.count()):
            item = self.pso_params.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_pso)

        for i in range(self.bees_params.count()):
            item = self.bees_params.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_bees)

    def run_calculation(self):
        f = available_functions[self.function_selector.currentText()]
        method_name = self.method_combo.currentText()
        method_class = optimization_methods[method_name]

        try:
            if method_name == "Градиентный спуск":
                initial_point = [float(self.initial_x0.text()), float(self.initial_x1.text())]
                max_iter = int(self.max_iter.text())
                epsilon1 = float(self.epsilon1.text())
                epsilon2 = float(self.epsilon2.text())
                learning_rate = float(self.learning_rate.text())
                method = method_class(f, initial_point, max_iter,
                                    learning_rate=learning_rate, epsilon1=epsilon1, epsilon2=epsilon2)
            elif method_name == "Квадратичный симплекс":
                max_iter = int(self.simplex_max_iter.text())
                constraints = [{"a": float(c["a"].text()), "b": float(c["b"].text()), "c": float(c["c"].text())}
                              for c in self.constraints]
                method = method_class(f, max_iter, constraints=constraints)
            elif method_name == "Генетический алгоритм":
                max_iter = int(self.genetic_max_iter.text())
                population_size = int(self.population_size.text())
                mutation_rate = float(self.mutation_rate.text())
                bounds = (float(self.bounds_lower.text()), float(self.bounds_upper.text()))
                method = method_class(f, None, max_iter,
                                    population_size=population_size, mutation_rate=mutation_rate, bounds=bounds)
            elif method_name == "Рой частиц":
                max_iter = int(self.pso_max_iter.text())
                swarmsize = int(self.pso_swarmsize.text())
                current_velocity_ratio = float(self.pso_current_velocity.text())
                local_velocity_ratio = float(self.pso_local_velocity.text())
                global_velocity_ratio = float(self.pso_global_velocity.text())
                min_bound = float(self.pso_min_bound.text())
                max_bound = float(self.pso_max_bound.text())
                method = method_class(f, None, max_iter,
                                    swarmsize=swarmsize,
                                    minvalues=[min_bound, min_bound],
                                    maxvalues=[max_bound, max_bound],
                                    current_velocity_ratio=current_velocity_ratio,
                                    local_velocity_ratio=local_velocity_ratio,
                                    global_velocity_ratio=global_velocity_ratio)
            elif method_name == "Алгоритм пчел":
                max_iter = int(self.bees_max_iter.text())
                scoutbeecount = int(self.bees_scoutbeecount.text())
                selectedbeecount = int(self.bees_selectedbeecount.text())
                bestbeecount = int(self.bees_bestbeecount.text())
                selsitescount = int(self.bees_selsitescount.text())
                bestsitescount = int(self.bees_bestsitescount.text())
                range_lower = float(self.bees_range_lower.text())
                range_upper = float(self.bees_range_upper.text())
                range_shrink = float(self.bees_range_shrink.text())
                max_stagnation = int(self.bees_max_stagnation.text())
                method = method_class(f, None, max_iter,
                                    scoutbeecount=scoutbeecount,
                                    selectedbeecount=selectedbeecount,
                                    bestbeecount=bestbeecount,
                                    selsitescount=selsitescount,
                                    bestsitescount=bestsitescount,
                                    range_lower=range_lower,
                                    range_upper=range_upper,
                                    range_shrink=range_shrink,
                                    max_stagnation=max_stagnation)

            final_point, trajectory, stop_reason, self.iterations_log = method.run()
            self.plot_canvas.plot(f, final_point, trajectory, method_name, constraints=method.kwargs.get("constraints", []))
        except ValueError as e:
            print(f"Ошибка ввода параметров: {e}")
            return

    def show_iterations(self):
        IterationWindow(self.iterations_log).exec()