# components/phase_shifter_panel.py
import sys
if "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers" not in sys.path:
    sys.path.append("S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers")

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit, QComboBox, QMessageBox
import numpy as np
from config import config
from LPS import Vaunix_LPS

class PhaseShifterPanel(QWidget):
    def __init__(self, parent, name, device_serial):
        super().__init__(parent)
        self.name = name
        self.device_serial = device_serial
        self.step_size = config.PHASE_SHIFTER_STEP_SIZES[0]
        self.value = 0  # Will be updated from device

        # Initialize the device
        self.device = Vaunix_LPS(name, device_serial, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.device.working_frequency(config.CENTER_FREQ)

        # Read the current value from the device
        try:
            self.value = round(self.device.phase.get_raw(), 1)
        except Exception as e:
            print(f"Error reading initial value for {self.name}: {e}")
            self.value = 0  # Default value if read fails

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel(self.name)
        layout.addWidget(self.label)

        self.value_label = QLabel(f"Current Value: {self.value}°")
        layout.addWidget(self.value_label)

        button_layout = QHBoxLayout()
        self.increase_button = QPushButton("↑")
        self.increase_button.clicked.connect(self.increase_value)
        button_layout.addWidget(self.increase_button)

        self.decrease_button = QPushButton("↓")
        self.decrease_button.clicked.connect(self.decrease_value)
        button_layout.addWidget(self.decrease_button)
        layout.addLayout(button_layout)

        # Entry to manually set value
        self.value_entry = QLineEdit()
        layout.addWidget(self.value_entry)

        # Set value button
        self.set_button = QPushButton("Set Value")
        self.set_button.clicked.connect(self.set_value)
        layout.addWidget(self.set_button)

        # Step size selection
        step_layout = QHBoxLayout()
        self.step_label = QLabel("Step Size:")
        step_layout.addWidget(self.step_label)

        self.step_combo = QComboBox()
        self.step_combo.addItems([str(s) for s in config.PHASE_SHIFTER_STEP_SIZES])
        self.step_combo.currentIndexChanged.connect(self.update_step_size)
        step_layout.addWidget(self.step_combo)
        layout.addLayout(step_layout)

        self.setLayout(layout)

    def update_step_size(self):
        self.step_size = float(self.step_combo.currentText())

    def update_value_display(self):
        self.value_label.setText(f"Current Value: {self.value}°")

    def increase_value(self):
        new_value = (self.value + self.step_size) % 360
        self.set_device_value(new_value)

    def decrease_value(self):
        new_value = (self.value - self.step_size) % 360
        self.set_device_value(new_value)

    def set_value(self):
        try:
            new_value = float(self.value_entry.text()) % 360
            self.set_device_value(new_value)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def set_device_value(self, new_value):
        # Update the device value
        value_rad = np.deg2rad(new_value)
        self.device.phase(value_rad)
        self.value = round(new_value, 2)
        self.update_value_display()
