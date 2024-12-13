# components/sweep_worker.py

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import uuid
import time
import traceback
import config
import threading  # Import threading for locks

# Adjust tolerance a bit because we now re-measure the cavity each iteration
delta_tolerance = 1e-6 * 1.0  # Tolerance for detuning in GHz
max_adjustments = 20  # Limit the number of adjustments per voltage step
ENABLE_HASH_MAP = True  # Enable YIG feedback logic

current_adjustment_step = 1.0e-7  # Current adjustment step (if needed), or used in proportional control
K_p = 2.5  # Proportional gain for YIG feedback

# Threshold to detect spurious peaks if they jump too far from the last known peak
# If the new measurement is more than spurious_threshold away from the last known, we discard it
spurious_threshold = 0.01  # GHz (example value)

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

        # Keep track of last known peak frequencies to avoid spurious results
        self.last_omega_c = None
        self.last_omega_y = None

    def run(self):
        try:
            voltages = self.generate_voltage_list(self.voltage_start, self.voltage_end, self.voltage_step)
            total_steps = len(voltages) * 3  # Multiply by 3 for the three readout types

            current_step_count = 0

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

                    # 1) Set configuration
                    self.switch_config_panel.set_configuration(readout_type)
                    time.sleep(0.1)  # Wait for hardware settle

                    # 2) Acquire data from VNA
                    freqs, power_dbm = self.vna_control_panel.get_current_trace()

                    # 3) Grab the newly measured peaks
                    new_omega_c = self.validate_peak(self.vna_control_panel.omega_C, self.last_omega_c, "cavity")
                    new_omega_y = self.validate_peak(self.vna_control_panel.omega_Y, self.last_omega_y, "yig")

                    # 4) Update last_omega_c and last_omega_y if valid
                    if new_omega_c is not None:
                        self.last_omega_c = new_omega_c
                    if new_omega_y is not None:
                        self.last_omega_y = new_omega_y

                    omega_C = self.last_omega_c
                    omega_Y = self.last_omega_y

                    kappa_C = self.vna_control_panel.kappa_C
                    kappa_Y = self.vna_control_panel.kappa_Y
                    Delta = None
                    K = None
                    if omega_C is not None and omega_Y is not None:
                        Delta = omega_C - omega_Y
                    if kappa_C is not None and kappa_Y is not None:
                        K = kappa_C - kappa_Y

                    print(f"Readout = {readout_type}")
                    print("Omega_C: ", omega_C)
                    print("Omega_Y: ", omega_Y)
                    print("Delta: ", Delta)

                    # 5) If YIG readout, apply feedback
                    if readout_type == "YIG Readout Only" and Delta is not None and ENABLE_HASH_MAP:
                        best_current = self.current_value
                        best_Delta = Delta
                        attempt = 0
                        print("Starting YIG feedback with proportional control...")

                        while attempt < max_adjustments:
                            print(f"Attempt {attempt + 1}: Delta={Delta}, best_Delta={best_Delta}")

                            # Adjust current using proportional control
                            new_current = self.current_value + K_p * Delta
                            print(f"Setting current to {new_current} A")
                            with self.current_lock:
                                print('inside of the actual setting code')
                                self.current_source.current.set(new_current)
                                self.current_value = new_current

                            time.sleep(0.2)  # wait for current to settle

                            # Re-measure YIG readout
                            self.switch_config_panel.set_configuration("YIG Readout Only")
                            time.sleep(0.1)
                            freqs, power_dbm = self.vna_control_panel.get_current_trace()

                            # Validate new YIG peak
                            new_omega_y = self.validate_peak(self.vna_control_panel.omega_Y, self.last_omega_y, "yig")
                            if new_omega_y is not None:
                                self.last_omega_y = new_omega_y
                            omega_Y = self.last_omega_y

                            # Re-measure cavity readout as well so that Delta is consistent
                            self.switch_config_panel.set_configuration("Cavity Readout Only")
                            time.sleep(0.1)
                            freqs_cav, power_dbm_cav = self.vna_control_panel.get_current_trace()
                            new_omega_c = self.validate_peak(self.vna_control_panel.omega_C, self.last_omega_c, "cavity")
                            if new_omega_c is not None:
                                self.last_omega_c = new_omega_c
                            omega_C = self.last_omega_c

                            Delta = None
                            if omega_C is not None and omega_Y is not None:
                                Delta = omega_C - omega_Y

                            if Delta is not None and abs(Delta) < abs(best_Delta):
                                best_Delta = Delta
                                best_current = new_current

                            attempt += 1
                            if Delta is not None and abs(Delta) < delta_tolerance:
                                print(f"Delta within tolerance: {Delta}. Stopping adjustments.")
                                break

                        print(f"Final best current for YIG: {best_current}, best Delta={best_Delta}")
                        with self.current_lock:
                            self.current_source.current.set(best_current)
                            self.current_value = best_current

                        # Optionally do final re-measure
                        time.sleep(0.2)
                        self.switch_config_panel.set_configuration("YIG Readout Only")
                        time.sleep(0.1)
                        freqs_final, power_final = self.vna_control_panel.get_current_trace()

                        # Re-validate final YIG
                        new_omega_y = self.validate_peak(self.vna_control_panel.omega_Y, self.last_omega_y, "yig")
                        if new_omega_y is not None:
                            self.last_omega_y = new_omega_y
                        self.switch_config_panel.set_configuration("Cavity Readout Only")
                        time.sleep(0.1)
                        freqs_cav, power_cav = self.vna_control_panel.get_current_trace()
                        new_omega_c = self.validate_peak(self.vna_control_panel.omega_C, self.last_omega_c, "cavity")
                        if new_omega_c is not None:
                            self.last_omega_c = new_omega_c

                        # Recompute Delta final
                        if self.last_omega_c is not None and self.last_omega_y is not None:
                            Delta = self.last_omega_c - self.last_omega_y

                        # store final YIG data
                        omega_C = self.last_omega_c
                        omega_Y = self.last_omega_y

                    # Save data to database
                    self.save_data_to_db(
                        freqs, power_dbm, voltage, readout_type,
                        omega_C, omega_Y, kappa_C, kappa_Y, Delta, K
                    )

                    # Update progress
                    current_step_count += 1
                    progress_percent = int((current_step_count / total_steps) * 100)
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

    def validate_peak(self, new_peak, last_peak, mode="cavity"):
        """
        If new_peak is None, returns None.
        If last_peak is None, returns new_peak (first measurement).
        Otherwise, checks if new_peak is too far from last_peak. If so, ignore it as spurious.
        """
        if new_peak is None:
            return None
        if last_peak is None:
            return new_peak
        # if the difference is too large, treat it as spurious
        if abs(new_peak - last_peak) > spurious_threshold:
            print(f"WARNING: {mode} peak jumped from {last_peak:.4f} to {new_peak:.4f} GHz, ignoring as spurious.")
            return last_peak
        return new_peak

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
        rtype = readout_type
        if readout_type == "Cavity Readout Only":
            rtype = "cavity"
        elif readout_type == "YIG Readout Only":
            rtype = "yig"
        elif readout_type == "Normal Operation":
            rtype = "normal"

        measurements = []
        from database import EPMeasurementModel
        for freq, power in zip(freqs, power_dbm):
            measurements.append(EPMeasurementModel(
                experiment_id=self.experiment_id,
                frequency_hz=freq,
                power_dBm=power,
                set_voltage=voltage,
                readout_type=rtype,
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
            ))

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
