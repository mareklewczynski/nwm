import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QComboBox, QLineEdit, QFileDialog, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import backendola as backend 
import pandas as pd

class LinearRegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Backend variables
        self.dataset = None
        self.x = None
        self.y = None
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.cost_function = backend.m_s_e
        self.regularization = None
        self.lambda_reg = 0.1
        self.cost_history = []
        self.model_params = None  # Store model parameters

    def init_ui(self):
        self.layout = QVBoxLayout()
        
        # Load Dataset
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.layout.addWidget(self.load_button)

        # Cost Function Dropdown
        self.cost_label = QLabel("Select Cost Function:")
        self.cost_dropdown = QComboBox()
        self.cost_dropdown.addItems(["MSE", "MAE", "R-squared"])
        self.cost_dropdown.currentIndexChanged.connect(self.select_cost_function)
        self.layout.addWidget(self.cost_label)
        self.layout.addWidget(self.cost_dropdown)
        
        # Learning Rate Input
        self.lr_label = QLabel("Learning Rate:")
        self.lr_input = QLineEdit("0.01")
        self.lr_input.textChanged.connect(self.set_learning_rate)
        self.layout.addWidget(self.lr_label)
        self.layout.addWidget(self.lr_input)

        # Regularization Type
        self.reg_label = QLabel("Regularization:")
        self.reg_dropdown = QComboBox()
        self.reg_dropdown.addItems(["None", "Lasso", "Ridge"])
        self.reg_dropdown.currentIndexChanged.connect(self.select_regularization)
        self.layout.addWidget(self.reg_label)
        self.layout.addWidget(self.reg_dropdown)

        # Regularization Parameter
        self.lambda_label = QLabel("Regularization Parameter:")
        self.lambda_input = QLineEdit("0.1")
        self.lambda_input.textChanged.connect(self.set_lambda)
        self.layout.addWidget(self.lambda_label)
        self.layout.addWidget(self.lambda_input)

        # Execution Buttons
        self.run_all_button = QPushButton("Run All Steps")
        self.run_all_button.clicked.connect(self.run_all_steps)
        self.run_one_button = QPushButton("Run One Step")
        self.run_one_button.clicked.connect(self.run_one_step)
        self.layout.addWidget(self.run_all_button)
        self.layout.addWidget(self.run_one_button)

        # Plot Widget
        self.plot = PlotCanvas(self)
        self.layout.addWidget(self.plot)

        # Results Section
        self.results_label = QLabel("Results:")
        self.layout.addWidget(self.results_label)
        
        self.cost_output = QLabel("Cost: N/A")
        self.layout.addWidget(self.cost_output)

        self.params_output = QLabel("Model Parameters: N/A")
        self.layout.addWidget(self.params_output)

        self.setLayout(self.layout)
        self.setWindowTitle("Linear Regression")

    def load_dataset(self):
        # 
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV Files (*.csv)")
        if file_path:
            try:
                data = pd.read_csv(file_path)
                self.x = data['YearsExperience'].values
                self.y = data['Salary'].values
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading dataset: {e}")
        
    def select_cost_function(self, index):
        if index == 0:
            self.cost_function = backend.m_s_e
        elif index == 1:
            self.cost_function = backend.m_a_e
        elif index == 2:
            self.cost_function = backend.r_squared

    def set_learning_rate(self, text):
        try:
            self.learning_rate = float(text)
        except ValueError:
            pass

    def select_regularization(self, index):
        if index == 0:
            self.regularization = None
        elif index == 1:
            self.regularization = "lasso"
        elif index == 2:
            self.regularization = "ridge"

    def set_lambda(self, text):
        try:
            self.lambda_reg = float(text)
        except ValueError:
            pass

    def run_all_steps(self):
        if self.x is None or self.y is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        try:
            m, b, predictions_history, self.cost_history = backend.gradient_descent(
                self.x, self.y, learning_rate=self.learning_rate,
                num_iterations=self.num_iterations, error_func=self.cost_function,
                regularization=self.regularization, lambda_reg=self.lambda_reg,
                verbose=True
            )
            
            # Calculate final predictions
            final_predictions = m * self.x + b
            
            # Store model parameters
            self.model_params = {'m': m, 'b': b}
            
            # Update results in GUI
            self.update_results()
            self.plot.update_plot(self.x, self.y, final_predictions)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during all steps: {e}")

    def run_one_step(self):
        if self.x is None or self.y is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        try:
            # Example one step of gradient descent
            m, b, predictions_history, cost_history = backend.gradient_descent(
                self.x, self.y, learning_rate=self.learning_rate, num_iterations=1,
                error_func=self.cost_function, regularization=self.regularization, lambda_reg=self.lambda_reg
            )
            
            # Update results in GUI
            self.model_params = {'m': m, 'b': b}
            self.cost_history.append(cost_history[0])  # Add cost from this step
            self.update_results()
            self.plot.update_plot(self.x, self.y, predictions_history[-1])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during one step of gradient descent: {e}")

    def update_results(self):
        # Update the cost and model parameters in the GUI
        final_cost = self.cost_history[-1] if self.cost_history else None
        self.cost_output.setText(f"Cost: {final_cost:.4f}" if final_cost is not None else "Cost: N/A")
        self.params_output.setText(f"Model Parameters: {self.model_params}" if self.model_params else "Model Parameters: N/A")


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)  
        super().__init__(self.fig)
        self.setParent(parent)

    def update_plot(self, x, y, predictions=None):
        self.ax.clear()
        self.ax.scatter(x, y, color='blue', label='Actual Data')  # Actual data
        if predictions is not None:
            self.ax.plot(x, predictions, color='red', label='Predicted Data')  # Predictions
        self.ax.set_title("Linear Regression Results")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinearRegressionApp()
    window.show()
    sys.exit(app.exec_())
