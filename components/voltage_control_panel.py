# components/voltage_control_panel.py
import sys
import config
if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit, QComboBox, QMessageBox
from config import config
from .device_controller import DeviceController

class VoltageControlPanel(QWidget):
    def __init__(self, parent=None, device_controller=None):
        super().__init__(parent)
        self.device_controller = device_controller or DeviceController(mock=False)
        self.step_size = config.VOLTAGE_STEP_SIZES[0]  # Default step size is 1V

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Voltage display
        self.voltage_label = QLabel(f"Current Voltage: {self.device_controller.get_voltage()} V")
        layout.addWidget(self.voltage_label)

        # Increase and decrease buttons
        button_layout = QHBoxLayout()
        self.increase_button = QPushButton("↑")
        self.increase_button.clicked.connect(self.increase_voltage)
        button_layout.addWidget(self.increase_button)

        self.decrease_button = QPushButton("↓")
        self.decrease_button.clicked.connect(self.decrease_voltage)
        button_layout.addWidget(self.decrease_button)
        layout.addLayout(button_layout)

        # Entry to manually set voltage
        self.voltage_entry = QLineEdit()
        layout.addWidget(self.voltage_entry)

        # Set voltage button
        self.set_button = QPushButton("Set Voltage")
        self.set_button.clicked.connect(self.set_voltage)
        layout.addWidget(self.set_button)

        # Step size selection
        step_layout = QHBoxLayout()
        self.step_label = QLabel("Step Size:")
        step_layout.addWidget(self.step_label)

        self.step_combo = QComboBox()
        self.step_combo.addItems([str(s) for s in config.VOLTAGE_STEP_SIZES])
        self.step_combo.currentIndexChanged.connect(self.update_step_size)
        step_layout.addWidget(self.step_combo)
        layout.addLayout(step_layout)

        self.setLayout(layout)

    def update_step_size(self):
        self.step_size = float(self.step_combo.currentText())

    def update_voltage_display(self):
        self.voltage_label.setText(f"Current Voltage: {self.device_controller.get_voltage()} V")

    def increase_voltage(self):
        current_voltage = self.device_controller.get_voltage()
        self.device_controller.set_voltage(current_voltage + self.step_size)
        self.update_voltage_display()

    def decrease_voltage(self):
        current_voltage = self.device_controller.get_voltage()
        self.device_controller.set_voltage(current_voltage - self.step_size)
        self.update_voltage_display()

    def set_voltage(self):
        try:
            new_voltage = float(self.voltage_entry.text())
            self.device_controller.set_voltage(new_voltage)
            self.update_voltage_display()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def get_voltage(self):
        return self.device_controller.get_voltage()
