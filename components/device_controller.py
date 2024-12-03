# components/device_controller.py

import config
from components.dc205 import SRSDC205 as dc

class DeviceController:
    def __init__(self, mock=False):
        self.voltage = 0.0  # Initial voltage
        if mock:
            self.inst = None
        else:
            self.inst = dc.open_serial(config.VOLTAGE_CONTROLLER_PORT, config.VOLTAGE_CONTROLLER_BAUDRATE)
        
    def set_voltage(self, voltage):
        self.voltage = round(voltage, 8)
        if self.inst:
            self.inst.channel[0].voltage = voltage
        else:
            print(f"Voltage set to {voltage} V")
        
    def get_voltage(self):
        return self.voltage
