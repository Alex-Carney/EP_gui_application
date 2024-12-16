# main.py
import sys
if "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers" not in sys.path:
    sys.path.append("S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers")

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea
from config import config

# Import components
from components import (
    VoltageControlPanel,
    CurrentControlPanel,
    SwitchConfigurationPanel,
    PhaseShifterPanelContainer,
    AttenuatorPanelContainer,
    VNAControlPanel,
    ConfigPanel,
    SweepControlPanel  
)

class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined GUI Application")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left layout for controls
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_scroll_area.setWidget(left_widget)
        main_layout.addWidget(left_scroll_area, stretch=1)

        # Right layout for VNA
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        # Add Voltage Control Panel
        self.voltage_control_panel = VoltageControlPanel()
        left_layout.addWidget(self.voltage_control_panel)

        # Add Current Control Panel
        self.current_control_panel = CurrentControlPanel()
        left_layout.addWidget(self.current_control_panel)



        # Add Phase Shifter Panels
        self.phase_shifter_panel = PhaseShifterPanelContainer()
        left_layout.addWidget(self.phase_shifter_panel)

        # Add Attenuator Panels
        self.attenuator_panel = AttenuatorPanelContainer()
        left_layout.addWidget(self.attenuator_panel)

        # Get the cavity att and yig att from the attenuator panel container
        cavity_feedback_atten = self.attenuator_panel.get_device("cavity_att")
        yig_feedback_atten = self.attenuator_panel.get_device("yig_att")

        # Add Switch Configuration Panel
        self.switch_config_panel = SwitchConfigurationPanel(cavity_feedback_atten=cavity_feedback_atten, yig_feedback_atten=yig_feedback_atten, attenuator_container=self.attenuator_panel)
        left_layout.addWidget(self.switch_config_panel)

        # Add the ConfigPanel
        self.config_panel = ConfigPanel(
            phase_shifter_container=self.phase_shifter_panel,
            attenuator_container=self.attenuator_panel
        )
        left_layout.addWidget(self.config_panel)

        # Add a stretch to left layout to push components to the top
        left_layout.addStretch()

        # Add VNA Control Panel
        self.vna_control_panel = VNAControlPanel(voltage_control_panel=self.voltage_control_panel)
        right_layout.addWidget(self.vna_control_panel)

        # Add Sweep Control Panel
        self.sweep_control_panel = SweepControlPanel(
            voltage_control_panel=self.voltage_control_panel,
            vna_control_panel=self.vna_control_panel,
            switch_config_panel=self.switch_config_panel,
            phase_shifter_container=self.phase_shifter_panel,
            attenuator_container=self.attenuator_panel,
            current_control_panel=self.current_control_panel
        )
        right_layout.addWidget(self.sweep_control_panel)

        # Connect configuration_changed signal to VNA control panel
        self.switch_config_panel.configuration_changed.connect(self.vna_control_panel.on_configuration_changed)

    def closeEvent(self, event):
        # Close devices properly
        self.vna_control_panel.closeEvent(event)
        super().closeEvent(event)

# Run the application
def main():
    app = QApplication(sys.argv)
    window = MainApplication()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
