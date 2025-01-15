# components/vector_network_analyzer.py

import numpy as np
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import vna.configuration as configuration
from scipy import signal
import sys

import config
if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)
try:
    import sc5511a
    from sc5511a import SignalCore_SC5511A
except ImportError:
    print("Could not import SignalCore SC5511A driver")
    raise


class VectorNetworkAnalyzer:
    def __init__(self, center_freq: float, span_freq: float, att_into_qm_box: float, lo_serial_number: str,
                 base_dir: str, bandwidth: float = 90e6,
                 lo_power: int = 21, n_avg: int = 100, f_min: float = 2e6,
                 del_f: float = 1e6):
        self.center_freq = center_freq
        self.span_freq = span_freq
        self.att_into_qm_box = att_into_qm_box
        self.lo_serial_number = lo_serial_number
        self.base_dir = base_dir
        self.lo_power = lo_power
        self.n_avg = n_avg
        self.f_min = f_min
        self.del_f = del_f
        self.bandwidth = bandwidth

        # Initialize LO
        self.lo = SignalCore_SC5511A("SC1", lo_serial_number)
        self.lo.set_level(self.lo_power)
        self.lo.set_output(1)

        # QM Manager
        self.qmm = QuantumMachinesManager(configuration.qop_ip, log_level=0)
        self.qm = self.qmm.open_qm(configuration.config)

    def set_center_frequency(self, center_freq):
        self.center_freq = center_freq

    def set_span_frequency(self, span_freq):
        self.span_freq = span_freq

    def set_attenuation(self, att):
        self.att_into_qm_box = att

    def set_del_f(self, del_f):
        self.del_f = del_f

    def get_single_trace(self):
        freqs = np.arange(self.f_min, self.span_freq + self.f_min + self.del_f / 2, self.del_f)
        lo_freq = self.center_freq - self.span_freq / 2 - self.f_min
        full_freqs = np.add(lo_freq, freqs)

        self.lo.set_frequency(lo_freq)

        I, Q = self._run_qua_program(freqs, self.f_min, self.span_freq + self.f_min, self.del_f)

        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        amp = np.sqrt(I ** 2 + Q ** 2)
        power_dbm = 10 * np.log10(((amp ** 2) * 1000) / 50) + self.att_into_qm_box

        return full_freqs, power_dbm, phase

    def _run_qua_program(self, freqs, f_min: float, f_max: float, df: float) -> (np.ndarray, np.ndarray):
        with program() as resonator_spec:
            n = declare(int)
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()

            with for_(n, 0, n < self.n_avg, n + 1):
                with for_(f, f_min, f <= f_max, f + df):
                    update_frequency("resonator", f)
                    measure(
                        "long_readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                    )
                    wait(2500, "resonator")
                    save(I, I_st)
                    save(Q, Q_st)

            with stream_processing():
                I_st.buffer(len(freqs)).average().save("I")
                Q_st.buffer(len(freqs)).average().save("Q")

        job = self.qm.execute(resonator_spec)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        I = res_handles.get("I").fetch_all()
        Q = res_handles.get("Q").fetch_all()
        return I, Q

    def turn_off(self):
        self.lo.set_output(0)
        self.lo.close()
        self.qm.close()

    def close(self):
        self.turn_off()
