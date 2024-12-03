# components/sweep_worker.py

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import uuid
import time
import traceback
import config

class SweepWorker(QThread):
    # Define signals
    progress = pyqtSignal(int)  # Progress percentage
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, voltage_start, voltage_end, voltage_step, session_factory, voltage_control_panel, vna_control_panel, switch_config_panel, phase_shifter_container, attenuator_container):
        super().__init__()
        self.voltage_start = voltage_start
        self.voltage_end = voltage_end
        self.voltage_step = voltage_step
        self.session_factory = session_factory
        self.voltage_control_panel = voltage_control_panel
        self.vna_control_panel = vna_control_panel
        self.switch_config_panel = switch_config_panel
        self.phase_shifter_container = phase_shifter_container
        self.attenuator_container = attenuator_container

        self.is_running = True  # Flag to control the thread

    def run(self):
        try:
            voltages = self.generate_voltage_list(self.voltage_start, self.voltage_end, self.voltage_step)
            total_steps = len(voltages) * 3  # Multiply by 3 for the three readout types

            current_step = 0

            # Loop over voltages
            for voltage in voltages:
                if not self.is_running:
                    break

                # Set voltage
                self.voltage_control_panel.device_controller.set_voltage(voltage)

                # Perform measurements for each readout type
                for readout_type in ["Cavity Readout Only", "YIG Readout Only", "Normal Operation"]:
                    if not self.is_running:
                        break

                    # Set configuration
                    self.switch_config_panel.set_configuration(readout_type)

                    # Wait for configuration to settle
                    time.sleep(0.1)  # Adjust as needed

                    # Get data from VNA
                    freqs, power_dbm = self.vna_control_panel.get_current_trace()

                    # Get additional parameters
                    omega_C = self.vna_control_panel.omega_C
                    omega_Y = self.vna_control_panel.omega_Y
                    kappa_C = self.vna_control_panel.kappa_C
                    kappa_Y = self.vna_control_panel.kappa_Y
                    Delta = omega_C - omega_Y if omega_C and omega_Y else None
                    K = kappa_C - kappa_Y if kappa_C and kappa_Y else None

                    # Save data to database
                    self.save_data_to_db(freqs, power_dbm, voltage, readout_type, omega_C, omega_Y, kappa_C, kappa_Y, Delta, K)

                    # Update progress
                    current_step += 1
                    progress_percent = int((current_step / total_steps) * 100)
                    self.progress.emit(progress_percent)

            self.finished.emit()
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(traceback_str)
            self.failed.emit(traceback_str)

    def stop(self):
        self.is_running = False

    def generate_voltage_list(self, start, end, step):
        return np.around(np.arange(start, end + step, step), decimals=8)

    def save_data_to_db(self, freqs, power_dbm, voltage, readout_type, omega_C, omega_Y, kappa_C, kappa_Y, Delta, K):
        from database import EPMeasurementModel

        experiment_id = str(uuid.uuid4())

        # Get settings from attenuators and phase shifters
        loop_phase = self.get_phase_shifter_value("loop_phase")

        # get loop from config
        loop_att = config.LOOP_ATT
        loopback_att = loop_att + config.LOOP_ATT_BACK_OFFSET

        yig_fb_phase = self.get_phase_shifter_value("yig_phase")
        yig_fb_att = self.get_attenuator_value("yig_att")

        cavity_fb_phase = self.get_phase_shifter_value("cavity_phase")
        cavity_fb_att = self.get_attenuator_value("cavity_att")

        # convert readout type
        if readout_type == "Cavity Readout Only":
            readout_type = "cavity"
        elif readout_type == "YIG Readout Only":
            readout_type = "yig"
        else:
            readout_type = "normal"

        measurements = [
            EPMeasurementModel(
                experiment_id=experiment_id,
                frequency_hz=freq,
                power_dBm=power,
                set_voltage=voltage,
                readout_type=readout_type,
                omega_C=omega_C,
                omega_Y=omega_Y,
                kappa_C=kappa_C,
                kappa_Y=kappa_Y,
                Delta=Delta,
                K=K,
                set_loop_phase_deg=loop_phase,
                set_loop_att=loop_att,
                set_loopback_att=loopback_att,
                set_yig_fb_phase_deg=yig_fb_phase,
                set_yig_fb_att=yig_fb_att,
                set_cavity_fb_phase_deg=cavity_fb_phase,
                set_cavity_fb_att=cavity_fb_att
            )
            for freq, power in zip(freqs, power_dbm)
        ]

        session = self.session_factory()
        try:
            session.add_all(measurements)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_phase_shifter_value(self, name):
        for panel in self.phase_shifter_container.panels:
            if panel.name == name:
                return panel.value
        return None

    def get_attenuator_value(self, name):
        for panel in self.attenuator_container.panels:
            if panel.name == name:
                return panel.value
        return None
