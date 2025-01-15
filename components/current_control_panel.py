# components/current_control_panel.py

import sys
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QComboBox, QMessageBox
)
from PyQt5.QtCore import QTimer

import config

# Add driver path if necessary
if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)

from CS580 import CS580

class CurrentControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize the current source device
        try:
            self.cs = CS580('cs', 'ASRL4::INSTR')
        except Exception as e:
            QMessageBox.critical(self, "Device Error", f"Failed to connect to CS580: {e}")
            self.cs = None

        self.current_value = 0.0  # Initialize current value as a float
        self.step_size = 1e-5  # Default step size (0.001e-2)
        self.lock = threading.Lock()  # Lock for thread safety

        self.initUI()

        # Start a timer to update the current value display periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_current_display)
        self.timer.start(1000)  # Update every second

    def initUI(self):
        layout = QVBoxLayout()

        self.title_label = QLabel("<b>Current Control</b>")
        layout.addWidget(self.title_label)

        # Display for current value
        self.current_label = QLabel("Current (A):")
        self.current_display = QLineEdit("0.0")
        self.current_display.setReadOnly(True)
        current_layout = QHBoxLayout()
        current_layout.addWidget(self.current_label)
        current_layout.addWidget(self.current_display)
        layout.addLayout(current_layout)

        # Controls for adjusting current
        controls_layout = QHBoxLayout()

        # Down button
        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(self.decrease_current)
        controls_layout.addWidget(self.down_button)

        # Up button
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self.increase_current)
        controls_layout.addWidget(self.up_button)

        # Step size dropdown
        self.step_size_combo = QComboBox()
        self.step_size_combo.addItems(["1.0e-7", "1.0e-6", "1.0e-5", "1.0e-4", "1.0e-3"])
        self.step_size_combo.currentIndexChanged.connect(self.step_size_changed)
        controls_layout.addWidget(QLabel("Step Size:"))
        controls_layout.addWidget(self.step_size_combo)

        layout.addLayout(controls_layout)

        self.setLayout(layout)

    def update_current_display(self):
        if self.cs is not None:
            with self.lock:
                try:
                    current = self.cs.current.get()
                    # Ensure current is treated as a float
                    self.current_value = float(current)
                    self.current_display.setText(f"{self.current_value:.6e}")  # Correct format for display
                except Exception as e:
                    print(f"Error getting current value: {e}")

    def increase_current(self):
        if self.cs is not None:
            with self.lock:
                new_current = float(self.current_value) + float(self.step_size)
                try:
                    self.cs.current.set(new_current)
                    self.current_value = new_current
                    self.current_display.setText(f"{new_current:.6e}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to set current: {e}")

    def decrease_current(self):
        if self.cs is not None:
            with self.lock:
                new_current = float(self.current_value) - float(self.step_size)  # Convert to float before subtraction
                try:
                    self.cs.current.set(new_current)
                    self.current_value = new_current
                    self.current_display.setText(f"{new_current:.6e}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to set current: {e}")

    def step_size_changed(self, index):
        step_size_str = self.step_size_combo.currentText()
        if step_size_str == "1.0e-7":
            self.step_size = 1e-7
        elif step_size_str == "1.0e-6":
            self.step_size = 1e-6
        elif step_size_str == "1.0e-5":
            self.step_size = 1e-5
        elif step_size_str == "1.0e-4":
            self.step_size = 1e-4
        elif step_size_str == "1.0e-3":
            self.step_size = 1e-3
        else:
            self.step_size = 1e-5  # Default

    def closeEvent(self, event):
        # Clean up the device connection if necessary
        if self.cs:
            # Implement any necessary shutdown procedures
            pass
        event.accept()

