# components/switch_configuration_panel.py
import sys
if "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers" not in sys.path:
    sys.path.append("S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers")

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal
from config import config
from microwave_switch import MicrowaveSwitchController
from LDA import Vaunix_LDA

class SwitchConfigurationPanel(QWidget):
    configuration_changed = pyqtSignal(str)  # Signal to emit when configuration changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.switch_controller = MicrowaveSwitchController()

        # Initialize attenuators
        self.loop_atten = Vaunix_LDA("LDA", 28577, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.loop_atten_back = Vaunix_LDA("LDA4", 34907, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.yig_drive_atten = Vaunix_LDA("LDA2", 33291, dll_path=config.DRIVERS_PATH, test_mode=False)

        # Setup working frequencies
        self.loop_atten.working_frequency(config.CENTER_FREQ)
        self.yig_drive_atten.working_frequency(config.CENTER_FREQ)
        self.loop_atten_back.working_frequency(config.CENTER_FREQ)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Label
        label = QLabel("Switch Configuration")
        layout.addWidget(label)

        # Buttons for each mode
        self.normal_button = QPushButton("Normal Operation")
        self.normal_button.clicked.connect(self.set_normal_operation)
        layout.addWidget(self.normal_button)

        self.cavity_button = QPushButton("Cavity Readout Only")
        self.cavity_button.clicked.connect(self.set_cavity_readout)
        layout.addWidget(self.cavity_button)

        self.yig_button = QPushButton("YIG Readout Only")
        self.yig_button.clicked.connect(self.set_yig_readout)
        layout.addWidget(self.yig_button)

        self.status_label = QLabel("Current Configuration: None")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_both_switches(self, port):
        try:
            self.switch_controller.set_active_port(port, config.DRIVE_SWITCH_SERIAL)
            self.switch_controller.set_active_port(port, config.READOUT_SWITCH_SERIAL)
            print(f"Both switches set to port {port}")
        except Exception as e:
            print(f"Error setting switches: {e}")

    def set_normal_operation(self):
        print("Normal operation selected.")
        LOOP_ATT = config.LOOP_ATT
        self.loop_atten.attenuation(LOOP_ATT)
        self.loop_atten_back.attenuation(LOOP_ATT + config.LOOP_ATT_BACK_OFFSET)
        self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.set_both_switches(1)
        self.status_label.setText("Current Configuration: Normal Operation")
        self.configuration_changed.emit("Normal Operation")  # Emit signal

    def set_cavity_readout(self):
        print("Cavity readout only selected.")
        self.loop_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.loop_atten_back.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.set_both_switches(1)
        self.status_label.setText("Current Configuration: Cavity Readout Only")
        self.configuration_changed.emit("Cavity Readout Only")  # Emit signal

    def set_yig_readout(self):
        print("YIG readout only selected.")
        self.loop_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.loop_atten_back.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_LOW)
        self.set_both_switches(2)
        self.status_label.setText("Current Configuration: YIG Readout Only")
        self.configuration_changed.emit("YIG Readout Only")  # Emit signal

    def set_configuration(self, configuration_name):
        if configuration_name == "Normal Operation":
            self.set_normal_operation()
        elif configuration_name == "Cavity Readout Only":
            self.set_cavity_readout()
        elif configuration_name == "YIG Readout Only":
            self.set_yig_readout()
        else:
            print(f"Unknown configuration: {configuration_name}")

