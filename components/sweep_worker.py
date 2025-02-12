# components/sweep_worker.py

# from PyQt5.QtCore import QThread, pyqtSignal
# import numpy as np
# import uuid
# import time
# import traceback
# import config
# import threading

# delta_tolerance = 1e-6
# max_adjustments = 20
# ENABLE_HASH_MAP = True
# K_p = 0.5  # Proportional gain
# # If your QCoDeS current parameter has a valid range [-0.1, 0.1], clamp accordingly:
# CURRENT_MIN = -0.1
# CURRENT_MAX =  0.1
# spurious_threshold = 0.05  # GHz

# class SweepWorker(QThread):
#     progress = pyqtSignal(int)
#     finished = pyqtSignal()
#     failed = pyqtSignal(str)

#     def __init__(
#             self, voltage_start, voltage_end, voltage_step, session_factory,
#             voltage_control_panel, current_control_panel, vna_control_panel,
#             switch_config_panel, phase_shifter_container, attenuator_container
#     ):
#         super().__init__()
#         self.voltage_start = voltage_start
#         self.voltage_end = voltage_end
#         self.voltage_step = voltage_step
#         self.session_factory = session_factory
#         self.voltage_control_panel = voltage_control_panel
#         self.current_control_panel = current_control_panel
#         self.vna_control_panel = vna_control_panel
#         self.switch_config_panel = switch_config_panel
#         self.phase_shifter_container = phase_shifter_container
#         self.attenuator_container = attenuator_container

#         self.is_running = True
#         self.experiment_id = str(uuid.uuid4())

#         # Current source device
#         self.current_source = self.current_control_panel.cs
#         self.current_value = self.current_control_panel.current_value
#         self.current_lock = threading.Lock()

#         # Last known cavity and YIG peak frequencies
#         self.last_omega_c = None
#         self.last_omega_y = None

#     def run(self):
#         try:
#             voltages = self.generate_voltage_list(self.voltage_start, self.voltage_end, self.voltage_step)
#             total_steps = len(voltages)*3
#             step_count = 0

#             for voltage in voltages:
#                 if not self.is_running:
#                     break

#                 self.voltage_control_panel.device_controller.set_voltage(voltage)
#                 print(f"Set voltage to {voltage} V")

#                 # 3 readout modes
#                 for readout_type in ["Cavity Readout Only", "YIG Readout Only", "Normal Operation"]:
#                     if not self.is_running:
#                         break

#                     # For multiple feedback attempts in "YIG Readout Only," skip restore to normal:
#                     skip_restore = (readout_type=="YIG Readout Only")
#                     self.switch_config_panel.set_configuration(readout_type, skip_restore=skip_restore)
#                     time.sleep(0.1)

#                     freqs, power_dbm = self.vna_control_panel.get_current_trace()

#                     new_omega_c = self.validate_peak(self.vna_control_panel.omega_C, self.last_omega_c, "cavity")
#                     new_omega_y = self.validate_peak(self.vna_control_panel.omega_Y, self.last_omega_y, "yig")

#                     if new_omega_c is not None:
#                         self.last_omega_c = new_omega_c
#                     if new_omega_y is not None:
#                         self.last_omega_y = new_omega_y

#                     omega_C = self.last_omega_c
#                     omega_Y = self.last_omega_y
#                     kappa_C = self.vna_control_panel.kappa_C
#                     kappa_Y = self.vna_control_panel.kappa_Y
#                     Delta = None
#                     K = None
#                     if omega_C is not None and omega_Y is not None:
#                         Delta = omega_C - omega_Y
#                     if kappa_C is not None and kappa_Y is not None:
#                         K = kappa_C - kappa_Y

#                     print(f"Readout={readout_type}: omegaC={omega_C}, omegaY={omega_Y}, Delta={Delta}")

#                     # YIG feedback
#                     if readout_type=="YIG Readout Only" and ENABLE_HASH_MAP and Delta is not None:
#                         best_current = self.current_value
#                         best_Delta = Delta
#                         attempt=0
#                         print("Starting YIG feedback with proportional control...")

#                         while attempt<max_adjustments and self.is_running:
#                             print(f" Attempt {attempt+1}: Delta={Delta}, bestDelta={best_Delta}")
#                             new_current = self.current_value + K_p*Delta
#                             # clamp new_current to hardware limits
#                             new_current = np.clip(new_current, CURRENT_MIN, CURRENT_MAX)
#                             print(f"** Setting current to {new_current:.7f} A (clamped) **")

#                             with self.current_lock:
#                                 try:
#                                     self.current_source.set_current(new_current)
#                                 except ValueError as e:
#                                     print(f"ERROR: current {new_current} is invalid for device: {e}")
#                                     # skip or clamp further
#                                     new_current = np.clip(new_current, CURRENT_MIN, CURRENT_MAX)
#                                     self.current_source.set_current(new_current)
#                                 self.current_value = new_current

#                             # optionally read-back:
#                             readback = self.current_source.get_current()
#                             print(f"Readback current: {readback} A")

#                             time.sleep(0.2)

#                             # Re-measure YIG readout
#                             self.switch_config_panel.set_configuration("YIG Readout Only", skip_restore=True)
#                             time.sleep(0.1)
#                             freqs_yig, power_yig = self.vna_control_panel.get_current_trace()
#                             new_omega_y = self.validate_peak(self.vna_control_panel.omega_Y, self.last_omega_y, "yig")
#                             if new_omega_y is not None:
#                                 self.last_omega_y = new_omega_y

#                             # Re-measure cavity to get updated Delta
#                             self.switch_config_panel.set_configuration("Cavity Readout Only", skip_restore=True)
#                             time.sleep(0.1)
#                             freqs_cav, power_cav = self.vna_control_panel.get_current_trace()
#                             new_omega_c = self.validate_peak(self.vna_control_panel.omega_C, self.last_omega_c, "cavity")
#                             if new_omega_c is not None:
#                                 self.last_omega_c = new_omega_c

#                             if self.last_omega_c is not None and self.last_omega_y is not None:
#                                 Delta = self.last_omega_c - self.last_omega_y
#                                 if abs(Delta)<abs(best_Delta):
#                                     best_Delta=Delta
#                                     best_current=new_current

#                             attempt+=1
#                             if Delta is not None and abs(Delta)<delta_tolerance:
#                                 print(f"Delta within tolerance: {Delta:.7f}. stopping.")
#                                 break

#                         print(f"Final best current after feedback: {best_current:.7f} A, bestDelta={best_Delta:.7g}")
#                         with self.current_lock:
#                             self.current_source.set_current(best_current)
#                             self.current_value = best_current
#                         time.sleep(0.2)

#                         # final measure 
#                         self.switch_config_panel.set_configuration("YIG Readout Only", skip_restore=True)
#                         time.sleep(0.1)
#                         _fy, _py = self.vna_control_panel.get_current_trace()
#                         # update final 
#                         final_omega_y = self.validate_peak(self.vna_control_panel.omega_Y, self.last_omega_y,"yig")
#                         if final_omega_y is not None:
#                             self.last_omega_y=final_omega_y

#                         self.switch_config_panel.set_configuration("Cavity Readout Only", skip_restore=True)
#                         time.sleep(0.1)
#                         _fc, _pc = self.vna_control_panel.get_current_trace()
#                         final_omega_c=self.validate_peak(self.vna_control_panel.omega_C,self.last_omega_c,"cavity")
#                         if final_omega_c is not None:
#                             self.last_omega_c=final_omega_c

#                         if self.last_omega_c is not None and self.last_omega_y is not None:
#                             Delta = self.last_omega_c - self.last_omega_y
#                         omega_C=self.last_omega_c
#                         omega_Y=self.last_omega_y

#                     self.save_data_to_db(
#                         freqs, power_dbm, voltage, readout_type,
#                         omega_C, omega_Y, kappa_C, kappa_Y, Delta, K
#                     )
#                     step_count+=1
#                     prog_percent=int((step_count/total_steps)*100)
#                     self.progress.emit(prog_percent)

#             self.finished.emit()
#         except Exception as e:
#             tb_str=traceback.format_exc()
#             print(tb_str)
#             self.failed.emit(tb_str)

#     def stop(self):
#         self.is_running=False

#     def generate_voltage_list(self, start,end,step):
#         return np.around(np.arange(start,end+step,step),decimals=8)

#     def validate_peak(self, new_peak, last_peak, mode="cavity"):
#         if new_peak is None:
#             return None
#         if last_peak is None:
#             return new_peak
#         if abs(new_peak - last_peak)>spurious_threshold:
#             print(f"WARNING: {mode} peak jumped from {last_peak:.5f} to {new_peak:.5f} GHz, ignoring as spurious.")
#             return last_peak
#         return new_peak

#     def save_data_to_db(self, freqs, power_dbm, voltage, readout_type,
#                         omega_C, omega_Y, kappa_C, kappa_Y, Delta, K):
#         from database import EPMeasurementModel

#         loop_phase=self.get_phase_shifter_value("loop_phase")
#         loop_att=config.LOOP_ATT
#         loopback_att=loop_att+config.LOOP_ATT_BACK_OFFSET
#         yig_fb_phase=self.get_phase_shifter_value("yig_phase")
#         yig_fb_att=self.get_attenuator_value("yig_att")
#         cavity_fb_phase=self.get_phase_shifter_value("cavity_phase")
#         cavity_fb_att=self.get_attenuator_value("cavity_att")

#         rtype=readout_type
#         if rtype=="Cavity Readout Only":
#             rtype="cavity"
#         elif rtype=="YIG Readout Only":
#             rtype="yig"
#         elif rtype=="Normal Operation":
#             rtype="normal"

#         measurements=[]
#         from database import EPMeasurementModel
#         for f, p in zip(freqs,power_dbm):
#             measurements.append(EPMeasurementModel(
#                 experiment_id=self.experiment_id,
#                 frequency_hz=f,
#                 power_dBm=p,
#                 set_voltage=voltage,
#                 readout_type=rtype,
#                 omega_C=omega_C,
#                 omega_Y=omega_Y,
#                 kappa_C=kappa_C,
#                 kappa_Y=kappa_Y,
#                 Delta=Delta,
#                 K=K,
#                 set_loop_phase_deg=loop_phase,
#                 set_loop_att=loop_att,
#                 set_loopback_att=loopback_att,
#                 set_yig_fb_phase_deg=yig_fb_phase,
#                 set_yig_fb_att=yig_fb_att,
#                 set_cavity_fb_phase_deg=cavity_fb_phase,
#                 set_cavity_fb_att=cavity_fb_att
#             ))

#         session=self.session_factory()
#         try:
#             session.add_all(measurements)
#             session.commit()
#         except Exception as e:
#             session.rollback()
#             raise e
#         finally:
#             session.close()

#     def get_phase_shifter_value(self,name):
#         for panel in self.phase_shifter_container.panels:
#             if panel.name==name:
#                 return panel.value
#         return None

#     def get_attenuator_value(self,name):
#         for panel in self.attenuator_container.panels:
#             if panel.name==name:
#                 return panel.value
#         return None



from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import uuid
import time
import traceback
import config
import threading  # Import threading for locks

delta_tolerance = .000001 * .75  # Tolerance for detuning in GHz - maybe increase this a bit? 
max_adjustments = 20  # Limit the number of adjustments per voltage step
current_adjustment_step = 1.0e-7  # Current adjustment step in Amperes

ENABLE_HASH_MAP = True  # Set to True to enable the adjustment logic


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

                    if readout_type == "YIG Readout Only" and Delta is not None and ENABLE_HASH_MAP:
                        # Initialize variables
                        best_current = self.current_value
                        best_Delta = Delta
                        adjustment_attempts = 0

                        # Define proportional gain
                        K_p = 0.5  # Adjust this value based on system response

                        print("Starting proportional control adjustments...")

                        while adjustment_attempts < max_adjustments:
                            print(f"Attempt {adjustment_attempts + 1}: Delta = {Delta}, Best Delta = {best_Delta}")

                            # Calculate new current using proportional control
                            new_current = self.current_value + K_p * Delta

                            # Set new current
                            print(f"Setting current to {new_current} A")
                            with self.current_lock:
                                self.current_source.voltage.set(new_current)
                                self.current_value = new_current  # Update current value

                            # Wait for current source to settle
                            time.sleep(0.1)

                            # Measure again
                            self.switch_config_panel.set_configuration("YIG Readout Only")
                            time.sleep(0.1)
                            freqs, power_dbm = self.vna_control_panel.get_current_trace()

                            # Recalculate parameters
                            omega_C = self.vna_control_panel.omega_C
                            omega_Y = self.vna_control_panel.omega_Y
                            Delta = omega_C - omega_Y if omega_C and omega_Y else None

                            # Update best values if the new Delta is better
                            if abs(Delta) < abs(best_Delta):
                                best_Delta = Delta
                                best_current = new_current

                            adjustment_attempts += 1

                            # Stop if within tolerance
                            if abs(Delta) < delta_tolerance:
                                print(f"Delta within tolerance: {Delta}. Stopping adjustments.")
                                break

                        # Set current to best found value
                        print(f"Setting current to best value: {best_current} A")
                        with self.current_lock:
                            self.current_source.voltage.set(best_current)
                            self.current_value = best_current  # Update current value

                        # Optionally, get final measurement at best current
                        time.sleep(0.1)
                        freqs, power_dbm = self.vna_control_panel.get_current_trace()
                        omega_C = self.vna_control_panel.omega_C
                        omega_Y = self.vna_control_panel.omega_Y
                        Delta = omega_C - omega_Y if omega_C and omega_Y else None


                    # If readout_type is "YIG Readout Only", adjust current if needed
                    # if readout_type == "YIG Readout Only" and Delta is not None and ENABLE_HASH_MAP:
                    #     # Initialize variables for adjustment
                    #     best_current = self.current_value
                    #     best_Delta = Delta
                    #     adjustment_attempts = 0

                    #     print("Adjusting current...")

                    #     print(
                    #         f"Are we going in the loop? {abs(Delta)} > {delta_tolerance} and {adjustment_attempts} < {max_adjustments}")

                    #     while abs(Delta) > delta_tolerance and adjustment_attempts < max_adjustments:
                    #         print(
                    #             f'Inside adjustment loop. Delta: {Delta}, tolerance: {delta_tolerance}, adjustment attempts: {adjustment_attempts}')

                    #         if Delta < -delta_tolerance:
                    #             # Detuning is negative; decrease current
                    #             print('Subtracting current_adjustment_step')
                    #             new_current = self.current_value - current_adjustment_step
                    #         elif Delta > delta_tolerance:
                    #             # Detuning is positive; increase current
                    #             print('Adding current_adjustment_step')
                    #             new_current = self.current_value + current_adjustment_step

                    #         # Set new current
                    #         print(f"Setting current to {new_current} A")
                    #         with self.current_lock:
                    #             self.current_source.current.set(new_current)
                    #             self.current_value = new_current  # Update current value

                    #         # Wait for current source to settle
                    #         time.sleep(0.1)

                    #         # Measure again in "YIG Readout Only" configuration
                    #         self.switch_config_panel.set_configuration("YIG Readout Only")
                    #         time.sleep(0.1)
                    #         freqs, power_dbm = self.vna_control_panel.get_current_trace()

                    #         # Recalculate parameters
                    #         omega_C = self.vna_control_panel.omega_C
                    #         omega_Y = self.vna_control_panel.omega_Y
                    #         Delta = omega_C - omega_Y if omega_C and omega_Y else None

                    #         # Check if this is the best Delta so far
                    #         if abs(Delta) < abs(best_Delta):
                    #             best_Delta = Delta
                    #             best_current = new_current

                    #         adjustment_attempts += 1

                    #     # After adjustment loop, set current to best_current
                    #     print(f"Setting current to best value: {best_current} A")
                    #     with self.current_lock:
                    #         self.current_source.current.set(best_current)
                    #         self.current_value = best_current  # Update current value

                    #     # Optionally, get final measurement at best current
                    #     time.sleep(0.1)
                    #     freqs, power_dbm = self.vna_control_panel.get_current_trace()
                    #     omega_C = self.vna_control_panel.omega_C
                    #     omega_Y = self.vna_control_panel.omega_Y
                    #     Delta = omega_C - omega_Y if omega_C and omega_Y else None

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