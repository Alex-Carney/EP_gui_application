# components/phase_shifter_panel_container.py
import sys
if "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers" not in sys.path:
    sys.path.append("S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers")

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from config import config
from .phase_shifter_panel import PhaseShifterPanel

class PhaseShifterPanelContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels = []

        layout = QVBoxLayout()

        # Use devices from config
        devices = config.PHASE_SHIFTER_DEVICES

        for device in devices:
            panel = PhaseShifterPanel(self, device["name"], device["serial"])
            layout.addWidget(panel)
            self.panels.append(panel)

        self.setLayout(layout)
