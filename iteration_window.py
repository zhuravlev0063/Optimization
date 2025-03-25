from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit

class IterationWindow(QDialog):
    def __init__(self, iterations_log):
        super().__init__()
        self.setWindowTitle("Итерации метода")
        self.setGeometry(150, 150, 400, 500)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText("\n".join(iterations_log))
        layout.addWidget(self.text_edit)
        self.setLayout(layout)