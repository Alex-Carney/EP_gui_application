# components/sweep_worker.py

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import uuid
import time
import traceback
import config
import threading  # Import threading for locks

delta_tolerance = .00005 * .9  # Tolerance for detuning in GHz
max_adjustments = 5  # Limit the number of adjustments per voltage step
current_adjustment_step = 0.001e-2  # Current adjustment step in Amperes

ENABLE_HASH_MAP = False  # Set to True to enable the adjustment logic

class SweepWorker(QThread):

    """

    DO NOT ASK CHAT GPT TO UPDATE THIS CODE WITHOUT GIVING IT THIS UPDATED VERSION

    """

    # Define signals
    progress = pyqtSignal(int)  # Progress percentage
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self, voltage_start, voltage_end, voltage_step, session_factory,
        voltage_control_panel, current_control_panel, vna_control_panel,
        switch_config_panel, phase_shifter_container, attenuator_container
    ):
        super().__init__()
        self.voltage_start = voltage_start
        self.voltage_end = voltage_end
        self.voltage_step = voltage_step
        self.session_factory = session_factory
        self.voltage_control_panel = voltage_control_panel
        self.current_control_panel = current_control_panel  # Added current control panel
        self.vna_control_panel = vna_control_panel
        self.switch_config_panel = switch_config_panel
        self.phase_shifter_container = phase_shifter_container
        self.attenuator_container = attenuator_container

        self.is_running = True  # Flag to control the thread

        self.experiment_id = str(uuid.uuid4())

        # Initialize current source device and initial current value
        self.current_source = self.current_control_panel.cs  # CS580 device
        self.current_value = self.current_control_panel.current_value  # Initial current value
        self.current_lock = threading.Lock()  # Lock for thread safety

    def run(self):
        try:
            voltages = self.generate_voltage_list(self.voltage_start, self.voltage_end, self.voltage_step)
            total_steps = len(voltages) * 3  # Multiply by 3 for the three readout types

            current_step = 0

            for voltage in voltages:
                if not self.is_running:
                    break

                # Set voltage
                self.voltage_control_panel.device_controller.set_voltage(voltage)
                print(f"Setting voltage to {voltage} V")

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

                    print("Omega_C: ", omega_C)
                    print("Omega_Y: ", omega_Y)
                    print("Delta: ", Delta)

                    # If readout_type is "YIG Readout Only", adjust current if needed
                    if readout_type == "YIG Readout Only" and Delta is not None and ENABLE_HASH_MAP:
                        # Initialize variables for adjustment
                        best_current = self.current_value
                        best_Delta = Delta
                        adjustment_attempts = 0

                        print("Adjusting current...")

                        print(f"Are we going in the loop? {abs(Delta)} > {delta_tolerance} and {adjustment_attempts} < {max_adjustments}")

                        while abs(Delta) > delta_tolerance and adjustment_attempts < max_adjustments:
                            print(f'Inside adjustment loop. Delta: {Delta}, tolerance: {delta_tolerance}, adjustment attempts: {adjustment_attempts}')

                            if Delta < -delta_tolerance:
                                # Detuning is negative; decrease current
                                print('Subtracting current_adjustment_step')
                                new_current = self.current_value - current_adjustment_step
                            elif Delta > delta_tolerance:
                                # Detuning is positive; increase current
                                print('Adding current_adjustment_step')
                                new_current = self.current_value + current_adjustment_step

                            # Set new current
                            print(f"Setting current to {new_current} A")
                            with self.current_lock:
                                self.current_source.current.set(new_current)
                                self.current_value = new_current  # Update current value

                            # Wait for current source to settle
                            time.sleep(0.1)

                            # Measure again in "YIG Readout Only" configuration
                            self.switch_config_panel.set_configuration("YIG Readout Only")
                            time.sleep(0.1)
                            freqs, power_dbm = self.vna_control_panel.get_current_trace()

                            # Recalculate parameters
                            omega_C = self.vna_control_panel.omega_C
                            omega_Y = self.vna_control_panel.omega_Y
                            Delta = omega_C - omega_Y if omega_C and omega_Y else None

                            # Check if this is the best Delta so far
                            if abs(Delta) < abs(best_Delta):
                                best_Delta = Delta
                                best_current = new_current

                            adjustment_attempts += 1

                        # After adjustment loop, set current to best_current
                        print(f"Setting current to best value: {best_current} A")
                        with self.current_lock:
                            self.current_source.current.set(best_current)
                            self.current_value = best_current  # Update current value

                        # Optionally, get final measurement at best current
                        time.sleep(0.1)
                        freqs, power_dbm = self.vna_control_panel.get_current_trace()
                        omega_C = self.vna_control_panel.omega_C
                        omega_Y = self.vna_control_panel.omega_Y
                        Delta = omega_C - omega_Y if omega_C and omega_Y else None

                    # Save data to database
                    self.save_data_to_db(
                        freqs, power_dbm, voltage, readout_type,
                        omega_C, omega_Y, kappa_C, kappa_Y, Delta, K
                    )

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

    def save_data_to_db(self, freqs, power_dbm, voltage, readout_type,
                        omega_C, omega_Y, kappa_C, kappa_Y, Delta, K):
        from database import EPMeasurementModel

        # Get settings from attenuators and phase shifters
        loop_phase = self.get_phase_shifter_value("loop_phase")

        loop_att = config.LOOP_ATT
        loopback_att = loop_att + config.LOOP_ATT_BACK_OFFSET

        yig_fb_phase = self.get_phase_shifter_value("yig_phase")
        yig_fb_att = self.get_attenuator_value("yig_att")

        cavity_fb_phase = self.get_phase_shifter_value("cavity_phase")
        cavity_fb_att = self.get_attenuator_value("cavity_att")

        # Convert readout type
        if readout_type == "Cavity Readout Only":
            readout_type = "cavity"
        elif readout_type == "YIG Readout Only":
            readout_type = "yig"
        else:
            readout_type = "normal"

        measurements = [
            EPMeasurementModel(
                experiment_id=self.experiment_id,
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
