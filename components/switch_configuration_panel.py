# components/switch_configuration_panel.py
import sys

import config

if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal
import config
from microwave_switch import MicrowaveSwitchController
from LDA import Vaunix_LDA


class SwitchConfigurationPanel(QWidget):
    configuration_changed = pyqtSignal(str)  # Signal to emit when configuration changes

    def __init__(self, parent=None, cavity_feedback_atten=None, yig_feedback_atten=None, attenuator_container=None):
        super().__init__(parent)
        self.switch_controller = MicrowaveSwitchController()

        # Initialize loop and YIG drive attenuators
        self.loop_atten = Vaunix_LDA("LDA", 28577, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.loop_atten_back = Vaunix_LDA("LDA4", 34907, dll_path=config.DRIVERS_PATH, test_mode=False)
        # self.yig_drive_atten = Vaunix_LDA("LDA2", 33291, dll_path=config.DRIVERS_PATH, test_mode=False)

        # Setup working frequencies
        self.loop_atten.working_frequency(config.CENTER_FREQ)
        # self.yig_drive_atten.working_frequency(config.CENTER_FREQ)
        self.loop_atten_back.working_frequency(config.CENTER_FREQ)

        # Feedback attenuators
        self.cavity_feedback_atten = cavity_feedback_atten
        self.yig_feedback_atten = yig_feedback_atten
        self.attenuator_container = attenuator_container

        # Variables to store baseline normal values
        self.normal_cavity_feedback_atten = None
        self.normal_yig_feedback_atten = None

        # We start in normal operation mode by default
        self.current_mode = "Normal Operation"

        # Since we start in normal mode, store the normal values now
        self.store_normal_values()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        label = QLabel("Switch Configuration")
        layout.addWidget(label)

        self.normal_button = QPushButton("Normal Operation")
        self.normal_button.clicked.connect(lambda: self.set_configuration("Normal Operation"))
        layout.addWidget(self.normal_button)

        self.cavity_button = QPushButton("Cavity Readout Only")
        self.cavity_button.clicked.connect(lambda: self.set_configuration("Cavity Readout Only"))
        layout.addWidget(self.cavity_button)

        self.yig_button = QPushButton("YIG Readout Only")
        self.yig_button.clicked.connect(lambda: self.set_configuration("YIG Readout Only"))
        layout.addWidget(self.yig_button)

        self.mixed_button = QPushButton("NR Mode")
        self.mixed_button.clicked.connect(lambda: self.set_configuration("NR Mode"))
        layout.addWidget(self.mixed_button)

        self.combined_button = QPushButton("Combined Mode")
        self.combined_button.clicked.connect(lambda: self.set_configuration("Combined Mode"))
        layout.addWidget(self.combined_button)

        self.status_label = QLabel("Current Configuration: Normal Operation")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def store_normal_values(self):
        """
        Store the current attenuator values as the baseline for 'normal' mode.
        Only call this when we are actually in normal mode and want to update the baseline.
        """
        if self.attenuator_container is not None:
            print('storing normal values now')
            # Retrieve current values from attenuator container
            cavity_val = self.attenuator_container.get_current_value("cavity_att")
            yig_val = self.attenuator_container.get_current_value("yig_att")

            if cavity_val is not None:
                print(f'setting normal cavity value is {cavity_val}')
                self.normal_cavity_feedback_atten = cavity_val
            if yig_val is not None:
                print(f'setting normal yig value is {yig_val}')
                self.normal_yig_feedback_atten = yig_val

    def restore_normal_values(self):
        print('inside of restoring normal values')
        print(f'normal values are {self.normal_cavity_feedback_atten} and {self.normal_yig_feedback_atten}')
        """
        Restore the attenuator values to what they were in normal mode.
        """
        if self.attenuator_container is not None:
            if self.normal_cavity_feedback_atten is not None:
                self.attenuator_container.set_value("cavity_att", self.normal_cavity_feedback_atten)
            if self.normal_yig_feedback_atten is not None:
                self.attenuator_container.set_value("yig_att", self.normal_yig_feedback_atten)

    def enable_yig_drive(self, enable):
        try:
            port = config.DRIVE_YIG_CONFIG["ENABLE"] if enable else config.DRIVE_YIG_CONFIG["DISABLE"]
            self.switch_controller.set_active_port(port, config.DRIVE_YIG_SERIAL)
        except Exception as e:
            print(f"Error setting drive switch: {e}")

    def enable_cavity_drive(self, enable):
        try:
            port = config.DRIVE_CAVITY_CONFIG["ENABLE"] if enable else config.DRIVE_CAVITY_CONFIG["DISABLE"]
            self.switch_controller.set_active_port(port, config.DRIVE_CAVITY_SERIAL)
        except Exception as e:
            print(f"Error setting drive switch: {e}")

    def enable_yig_readout(self, enable):
        try:
            port = config.READOUT_YIG_CONFIG["ENABLE"] if enable else config.READOUT_YIG_CONFIG["DISABLE"]
            self.switch_controller.set_active_port(port, config.READOUT_YIG_SERIAL)
        except Exception as e:
            print(f"Error setting readout switch: {e}")

    def enable_cavity_readout(self, enable):
        try:
            port = config.READOUT_CAVITY_CONFIG["ENABLE"] if enable else config.READOUT_CAVITY_CONFIG["DISABLE"]
            self.switch_controller.set_active_port(port, config.READOUT_CAVITY_SERIAL)
        except Exception as e:
            print(f"Error setting readout switch: {e}")

    def set_drive_switch(self, mode: str):
        """
        :param mode: str | Can be CAVITY, YIG, MIXED
        """
        try:
            port = config.DRIVE_SWITCH_CONFIG[mode]
            self.switch_controller.set_active_port(port, config.DRIVE_SWITCH_SERIAL)
        except Exception as e:
            print(f"Error setting drive switch: {e}")

    def set_readout_switch(self, mode: str):
        """
        :param mode: str | Can be CAVITY, YIG, MIXED
        """
        try:
            port = config.READOUT_SWITCH_CONFIG[mode]
            self.switch_controller.set_active_port(port, config.READOUT_SWITCH_SERIAL)
        except Exception as e:
            print(f"Error setting readout switch: {e}")

    def toggle_loop_coupling(self, turn_on):
        port = 1 if turn_on else 2
        try:
            self.switch_controller.set_active_port(port, config.CAV_TO_YIG_SWITCH_SERIAL)
            self.switch_controller.set_active_port(port, config.YIG_TO_CAV_SWITCH_SERIAL)
        except Exception as e:
            print(f"Error setting switches: {e}")

    def set_combined_mode(self):
        """
        This is the new code that sets up whatever the hardware
        should do for Combined Mode. For example, you might
        just do 'Normal Operation' as the hardware default,
        or no hardware change at all, because the real logic
        is going to happen in the VNA panel (two sweeps).
        """
        # Possibly just restore normal? or do nothing special:
        # self.restore_normal_values()

        self.status_label.setText("Current Configuration: Combined Mode")
        self.configuration_changed.emit("Combined Mode")

    def set_normal_operation(self):
        # In normal operation, restore normal values
        self.restore_normal_values()

        LOOP_ATT = config.LOOP_ATT
        self.loop_atten.attenuation(LOOP_ATT)
        self.loop_atten_back.attenuation(LOOP_ATT + config.LOOP_ATT_BACK_OFFSET)

        # self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)

        self.set_drive_switch("CAVITY")
        self.set_readout_switch("CAVITY")
        # self.set_switch_35353("CAVITY")
        # self.set_switch_34875("YIG")
        # self.enable_cavity_drive(True)
        # self.enable_cavity_readout(True)
        # self.enable_yig_drive(False)
        # self.enable_yig_readout(False)

        # enable coupling
        self.toggle_loop_coupling(True)
        self.status_label.setText("Current Configuration: Normal Operation")
        self.configuration_changed.emit("Normal Operation")

        # Now that we are in normal operation, we can store normal values again
        # to capture any changes made.
        self.store_normal_values()

    def set_normal_operation_silent(self):
        # Normal operation, but not changing anything to VNA/GUI (for combined mode)
        LOOP_ATT = config.LOOP_ATT
        self.loop_atten.attenuation(LOOP_ATT)
        self.loop_atten_back.attenuation(LOOP_ATT + config.LOOP_ATT_BACK_OFFSET)
        self.set_drive_switch("CAVITY")
        self.set_readout_switch("CAVITY")

    def set_nr_mode_silent(self):
        LOOP_ATT = config.LOOP_ATT
        self.loop_atten.attenuation(LOOP_ATT)
        self.loop_atten_back.attenuation(LOOP_ATT + config.LOOP_ATT_BACK_OFFSET)
        self.set_drive_switch("YIG")
        self.set_readout_switch("YIG")

    def set_nr_mode(self):
        self.restore_normal_values()

        LOOP_ATT = config.LOOP_ATT
        self.loop_atten.attenuation(LOOP_ATT)
        self.loop_atten_back.attenuation(LOOP_ATT + config.LOOP_ATT_BACK_OFFSET)

        # self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)

        self.set_drive_switch("CAVITY")
        self.set_readout_switch("YIG")
        # self.enable_cavity_drive(True)
        # self.enable_cavity_readout(True)
        # self.enable_yig_drive(True)
        # self.enable_yig_readout(True)

        # enable coupling
        self.toggle_loop_coupling(True)
        self.status_label.setText("Current Configuration: NR Mode")
        self.configuration_changed.emit("NR Mode")

        # Now that we are in normal (mixed is a normal) operation, we can store normal values again
        # to capture any changes made.
        self.store_normal_values()

    def set_cavity_readout(self):
        # For switching to cavity readout:
        # If we are currently in normal mode, store normal values once
        # If we are in YIG or Cavity mode, first restore normal, then proceed

        # TODO: SWITCH ON CURRENT MODE

        if self.current_mode == "Normal Operation" or self.current_mode == "NR Mode" or self.current_mode == "Combined Mode":
            self.store_normal_values()
        else:
            # Coming from YIG or Cavity mode:
            # Always restore normal first
            self.restore_normal_values()

        # Now apply cavity mode
        self.loop_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.loop_atten_back.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        # self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)

        # When reading out Cavity, turn off YIG (e.g., set YIG feedback to 50)
        # Also, if you want to set cavity feedback high or something else, do so here
        if self.attenuator_container is not None:
            # Example: YIG feedback off, cavity feedback still normal or some setting?
            print('turning off yig')
            self.attenuator_container.set_value("yig_att", 50)

        self.set_drive_switch("CAVITY")
        self.set_readout_switch("CAVITY")

        # self.enable_cavity_drive(True)
        # self.enable_cavity_readout(True)
        # self.enable_yig_drive(False)
        # self.enable_yig_readout(False)

        self.toggle_loop_coupling(False)
        self.status_label.setText("Current Configuration: Cavity Readout Only")
        self.configuration_changed.emit("Cavity Readout Only")

    def set_yig_readout(self):
        # For switching to YIG readout:
        # If we are currently in normal mode, we have baseline normal values
        # If we are in Cavity mode, restore normal first, then set YIG

        # TODO: SWITCH ON CURRENT MODE

        if self.current_mode == "Normal Operation" or self.current_mode == "NR Mode" or self.current_mode == "Combined Mode":
            self.store_normal_values()
        else:
            # Coming from Cavity or YIG:
            # Always restore normal first
            self.restore_normal_values()

        # Now apply YIG mode
        self.loop_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.loop_atten_back.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        # self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_LOW)

        # When reading out YIG, turn off cavity (e.g., set cavity feedback to 50)
        if self.attenuator_container is not None:
            print('turning off cavity')
            self.attenuator_container.set_value("cavity_att", 50)

        # self.set_switch_35353("NULL")
        # self.set_switch_34875("YIG")
        self.set_drive_switch("YIG")
        self.set_readout_switch("YIG")
        # self.enable_cavity_drive(False)
        # self.enable_cavity_readout(False)
        # self.enable_yig_drive(True)
        # self.enable_yig_readout(True)

        self.toggle_loop_coupling(False)
        self.status_label.setText("Current Configuration: YIG Readout Only")
        self.configuration_changed.emit("YIG Readout Only")

    def set_configuration(self, configuration_name, skip_restore=False):
        if configuration_name == self.current_mode:
            # Already in this mode, do nothing
            return

        if configuration_name == "Normal Operation":
            # Switching to normal from either cavity or yig
            self.set_normal_operation()
        elif configuration_name == "Cavity Readout Only":
            # Switching to cavity from either normal or yig
            self.set_cavity_readout()
        elif configuration_name == "YIG Readout Only":
            # Switching to YIG from either normal or cavity
            self.set_yig_readout()
        elif configuration_name == "NR Mode":
            # Switching to mixed mode
            self.set_nr_mode()
        elif configuration_name == "Combined Mode":
            # Switching to combined mode
            self.set_combined_mode()
        else:
            print(f"Unknown configuration: {configuration_name}")

        # Update current mode
        self.current_mode = configuration_name
