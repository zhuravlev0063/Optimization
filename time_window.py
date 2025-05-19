from PySide6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from plotting import plot_fitness_vs_iterations, plot_fitness_vs_population, plot_time_vs_population

class TimeVsDimensionWindow(QDialog):
    def __init__(self, f):
        super().__init__()
        self.setWindowTitle("График сравнения алгоритмов")
        self.setGeometry(150, 150, 800, 600)
        layout = QVBoxLayout()

        # Выбор типа графика
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems([
            "Фитнес vs. Итерации",
            "Фитнес vs. Размер популяции",
            "Время vs. Размер популяции"
        ])
        self.graph_type_combo.currentTextChanged.connect(self.update_plot)
        layout.addWidget(QLabel("Выберите тип графика:"))
        layout.addWidget(self.graph_type_combo)

        # Плейсхолдер для графика
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Кнопка для сохранения графика
        self.save_button = QPushButton("Сохранить график")
        self.save_button.clicked.connect(self.save_current_plot)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.f = f
        self.update_plot()

    def update_plot(self):
        try:
            graph_type = self.graph_type_combo.currentText()
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if graph_type == "Фитнес vs. Итерации":
                data = plot_fitness_vs_iterations(self.f)
                print("Фитнес vs. Итерации:", {k: {sk: len(sv) for sk, sv in v.items()} for k, v in data.items() if k in ['ga', 'pso', 'hybrid']})
                if len(data['ga']['iterations']) > 0:
                    ax.plot(data['ga']['iterations'], data['ga']['fitness'], label='Генетический алгоритм', color='blue')
                if len(data['pso']['iterations']) > 0:
                    ax.plot(data['pso']['iterations'], data['pso']['fitness'], label='Рой частиц', color='red')
                if len(data['hybrid']['iterations']) > 0:
                    ax.plot(data['hybrid']['iterations'], data['hybrid']['fitness'], label='Гибридный ГА+PSO', color='green')
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_title(data['title'])
                ax.set_yscale('log')
            elif graph_type == "Фитнес vs. Размер популяции":
                data = plot_fitness_vs_population(self.f)
                print("Фитнес vs. Размер популяции:", {k: {sk: len(sv) for sk, sv in v.items()} for k, v in data.items() if k in ['ga', 'pso', 'hybrid']})
                if len(data['ga']['sizes']) > 0:
                    ax.plot(data['ga']['sizes'], data['ga']['fitness'], label='Генетический алгоритм', color='blue')
                if len(data['pso']['sizes']) > 0:
                    ax.plot(data['pso']['sizes'], data['pso']['fitness'], label='Рой частиц', color='red')
                if len(data['hybrid']['sizes']) > 0:
                    ax.plot(data['hybrid']['sizes'], data['hybrid']['fitness'], label='Гибридный ГА+PSO', color='green')
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_title(data['title'])
                ax.set_yscale('log')
            elif graph_type == "Время vs. Размер популяции":
                data = plot_time_vs_population(self.f)
                print("Время vs. Размер популяции:", {k: {sk: len(sv) for sk, sv in v.items()} for k, v in data.items() if k in ['ga', 'pso', 'hybrid']})
                if len(data['ga']['sizes']) > 0:
                    ax.plot(data['ga']['sizes'], data['ga']['times'], label='Генетический алгоритм', color='blue')
                if len(data['pso']['sizes']) > 0:
                    ax.plot(data['pso']['sizes'], data['pso']['times'], label='Рой частиц', color='red')
                if len(data['hybrid']['sizes']) > 0:
                    ax.plot(data['hybrid']['sizes'], data['hybrid']['times'], label='Гибридный ГА+PSO', color='green')
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_title(data['title'])

            ax.legend()
            ax.grid(True)
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Ошибка построения графика: {e}")
            import traceback
            traceback.print_exc()

    def save_current_plot(self):
        try:
            graph_type = self.graph_type_combo.currentText()
            if graph_type == "Фитнес vs. Итерации":
                data = plot_fitness_vs_iterations(self.f)
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                if len(data['ga']['iterations']) > 0:
                    ax.plot(data['ga']['iterations'], data['ga']['fitness'], label='Генетический алгоритм', color='blue')
                if len(data['pso']['iterations']) > 0:
                    ax.plot(data['pso']['iterations'], data['pso']['fitness'], label='Рой частиц', color='red')
                if len(data['hybrid']['iterations']) > 0:
                    ax.plot(data['hybrid']['iterations'], data['hybrid']['fitness'], label='Гибридный ГА+PSO', color='green')
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_title(data['title'])
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(data['save_path'])
            elif graph_type == "Фитнес vs. Размер популяции":
                data = plot_fitness_vs_population(self.f)
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                if len(data['ga']['sizes']) > 0:
                    ax.plot(data['ga']['sizes'], data['ga']['fitness'], label='Генетический алгоритм', color='blue')
                if len(data['pso']['sizes']) > 0:
                    ax.plot(data['pso']['sizes'], data['pso']['fitness'], label='Рой частиц', color='red')
                if len(data['hybrid']['sizes']) > 0:
                    ax.plot(data['hybrid']['sizes'], data['hybrid']['fitness'], label='Гибридный ГА+PSO', color='green')
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_title(data['title'])
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(data['save_path'])
            elif graph_type == "Время vs. Размер популяции":
                data = plot_time_vs_population(self.f)
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                if len(data['ga']['sizes']) > 0:
                    ax.plot(data['ga']['sizes'], data['ga']['times'], label='Генетический алгоритм', color='blue')
                if len(data['pso']['sizes']) > 0:
                    ax.plot(data['pso']['sizes'], data['pso']['times'], label='Рой частиц', color='red')
                if len(data['hybrid']['sizes']) > 0:
                    ax.plot(data['hybrid']['sizes'], data['hybrid']['times'], label='Гибридный ГА+PSO', color='green')
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_title(data['title'])
                ax.legend()
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(data['save_path'])
            print(f"График сохранён: {data['save_path']}")
        except Exception as e:
            print(f"Ошибка сохранения графика: {e}")
            import traceback
            traceback.print_exc()