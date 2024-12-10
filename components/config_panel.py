# components/config_panel.py

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QLabel
import os
import json

class ConfigPanel(QWidget):
    def __init__(self, parent=None, phase_shifter_container=None, attenuator_container=None):
        super().__init__(parent)
        self.phase_shifter_container = phase_shifter_container
        self.attenuator_container = attenuator_container

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Add a description 
        description = "DO NOT SAVE OR LOAD UNLESS IN NORMAL OPERATION!!"
        layout.addWidget(QLabel(description))

        # Save buttons
        save_layout = QHBoxLayout()
        for i in range(1, 4):
            btn = QPushButton(f"Save Config {i}")
            btn.clicked.connect(lambda checked, idx=i: self.save_config(idx))
            save_layout.addWidget(btn)
        layout.addLayout(save_layout)

        # Load buttons
        load_layout = QHBoxLayout()
        for i in range(1, 4):
            btn = QPushButton(f"Load Config {i}")
            btn.clicked.connect(lambda checked, idx=i: self.load_config(idx))
            load_layout.addWidget(btn)
        layout.addLayout(load_layout)

        self.setLayout(layout)

    def save_config(self, idx):
        config_data = {
            "phase_shifters": {},
            "attenuators": {}
        }

        # Get values from phase shifters
        for panel in self.phase_shifter_container.panels:
            config_data["phase_shifters"][panel.name] = panel.value

        # Get values from attenuators
        for panel in self.attenuator_container.panels:
            config_data["attenuators"][panel.name] = panel.value

        # Save to file
        filename = f"config_{idx}.json"
        with open(filename, "w") as f:
            json.dump(config_data, f)

        print(f"Configuration saved to {filename}")

    def load_config(self, idx):
        filename = f"config_{idx}.json"
        if not os.path.exists(filename):
            print(f"Configuration file {filename} does not exist.")
            return

        with open(filename, "r") as f:
            config_data = json.load(f)

        # Set values to phase shifters
        for panel in self.phase_shifter_container.panels:
            value = config_data["phase_shifters"].get(panel.name)
            if value is not None:
                panel.set_device_value(value)

        # Set values to attenuators
        for panel in self.attenuator_container.panels:
            value = config_data["attenuators"].get(panel.name)
            if value is not None:
                panel.set_device_value(value)

        print(f"Configuration loaded from {filename}")
