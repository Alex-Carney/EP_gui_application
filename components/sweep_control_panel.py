# components/sweep_control_panel.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QProgressBar, QMessageBox
from PyQt5.QtCore import pyqtSignal
from components.sweep_worker import SweepWorker

class SweepControlPanel(QWidget):
    def __init__(self, parent=None, voltage_control_panel=None, vna_control_panel=None, switch_config_panel=None, phase_shifter_container=None, attenuator_container=None):
        super().__init__(parent)
        self.voltage_control_panel = voltage_control_panel
        self.vna_control_panel = vna_control_panel
        self.switch_config_panel = switch_config_panel
        self.phase_shifter_container = phase_shifter_container
        self.attenuator_container = attenuator_container

        self.is_sweeping = False

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Voltage start
        self.voltage_start_label = QLabel("Voltage Start (V):")
        self.voltage_start_input = QLineEdit("0.0")
        layout.addWidget(self.voltage_start_label)
        layout.addWidget(self.voltage_start_input)

        # Voltage end
        self.voltage_end_label = QLabel("Voltage End (V):")
        self.voltage_end_input = QLineEdit("1.0")
        layout.addWidget(self.voltage_end_label)
        layout.addWidget(self.voltage_end_input)

        # Voltage step
        self.voltage_step_label = QLabel("Voltage Step (V):")
        self.voltage_step_input = QLineEdit("0.1")
        layout.addWidget(self.voltage_step_label)
        layout.addWidget(self.voltage_step_input)

        # Database name
        self.db_name_label = QLabel("Database Name:")
        self.db_name_input = QLineEdit("experiment_data")
        layout.addWidget(self.db_name_label)
        layout.addWidget(self.db_name_input)

        # Start Sweep button
        self.start_sweep_button = QPushButton("Start Sweep")
        self.start_sweep_button.clicked.connect(self.start_sweep)
        layout.addWidget(self.start_sweep_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def start_sweep(self):
        if self.is_sweeping:
            QMessageBox.warning(self, "Sweep in Progress", "A sweep is already in progress.")
            return

        # Get sweep parameters
        try:
            voltage_start = float(eval(self.voltage_start_input.text()))
            voltage_end = float(eval(self.voltage_end_input.text()))
            voltage_step = float(eval(self.voltage_step_input.text()))
            db_name = self.db_name_input.text()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", f"Error parsing inputs: {e}")
            raise e

        # Initialize database
        from database import create_database
        session_factory = create_database(db_name)

        # Create the SweepWorker
        self.sweep_worker = SweepWorker(
            voltage_start=voltage_start,
            voltage_end=voltage_end,
            voltage_step=voltage_step,
            session_factory=session_factory,
            voltage_control_panel=self.voltage_control_panel,
            vna_control_panel=self.vna_control_panel,
            switch_config_panel=self.switch_config_panel,
            phase_shifter_container=self.phase_shifter_container,
            attenuator_container=self.attenuator_container
        )

        # Connect signals
        self.sweep_worker.progress.connect(self.progress_bar.setValue)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.failed.connect(self.on_sweep_failed)

        # Start the sweep
        self.is_sweeping = True
        # Pause GUI updates during sweep
        self.vna_control_panel.pause_updates = True
        self.sweep_worker.start()

    def on_sweep_finished(self):
        self.is_sweeping = False
        self.vna_control_panel.pause_updates = False
        QMessageBox.information(self, "Sweep Complete", "The sweep has completed successfully.")

    def on_sweep_failed(self, error_message):
        self.is_sweeping = False
        self.vna_control_panel.pause_updates = False

        # Display the full stack trace in the dialog
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("Sweep Failed")
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText("The sweep failed with an error:")
        error_dialog.setDetailedText(error_message)
        error_dialog.setStandardButtons(QMessageBox.Ok)
        error_dialog.exec_()

