# components/attenuator_panel_container.py
import sys
import config
if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from .attenuator_panel import AttenuatorPanel
from config import config

class AttenuatorPanelContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels = []

        layout = QVBoxLayout()

        # Use devices from config
        devices = config.ATTENUATOR_DEVICES

        for device in devices:
            panel = AttenuatorPanel(self, device["name"], device["serial"])
            layout.addWidget(panel)
            self.panels.append(panel)

        self.setLayout(layout)

    def get_device(self, name):
        """
        Retrieve the device associated with the given attenuator name.
        Returns the device object if found, or None if no match is found.
        """
        for panel in self.panels:
            if panel.name == name:
                return panel.device
        return None

    def get_current_value(self, name):
        """
        Retrieve the current attenuation value from the specified attenuator panel.
        Returns the current attenuation value (float) or None if not found.
        """
        for panel in self.panels:
            if panel.name == name:
                return panel.value
        return None

    def set_value(self, name, val):
        """
        Set the attenuation value for the specified attenuator.
        Updates both the panel's memory of the value and the device's attenuation.
        """
        for panel in self.panels:
            if panel.name == name:
                panel.set_device_value(val)
                return
