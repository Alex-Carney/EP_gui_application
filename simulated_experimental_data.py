"""
Simulate data for YIG, Cavity, and Normal (hybrid) modes
and save it in a CSV that matches the experimental DB columns.

This CSV can then be loaded or aggregated by the existing code
that expects columns from EPMeasurementModel.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import analysis.symbolic_module as sm  # user-provided symbolic module for the model

# Our DB columns from EPMeasurementModel
DB_COLUMNS = [
    "experiment_id",
    "frequency_hz",
    "power_dBm",
    "readout_type",
    "omega_C",
    "omega_Y",
    "kappa_C",
    "kappa_Y",
    "Delta",
    "K",
    "set_loop_phase_deg",
    "set_loop_att",
    "set_loopback_att",
    "set_yig_fb_phase_deg",
    "set_yig_fb_att",
    "set_cavity_fb_phase_deg",
    "set_cavity_fb_att",
    "set_voltage"  # We'll store kappa_y here
]

# Constants
J_COUPLING = 0.1
J_OFF = 0.0
CAVITY_READOUT = np.array([1, 0])
YIG_READOUT = np.array([0, 1])

# If your cavity damping is KAPPA_C,
# let's store that as the "gamma" for the cavity
KAPPA_C = 0.15
OMEGA_C = 6.0
OMEGA_Y = 6.0


def main():
    os.makedirs("plots_simulation", exist_ok=True)

    # Create an empty list of rows to store into a final CSV
    sim_data_rows = []

    symbols_dict = sm.setup_symbolic_equations()

    params = sm.ModelParams(
        J_val=J_COUPLING,
        g_val=0,
        cavity_freq=OMEGA_C,
        w_y=OMEGA_Y,
        gamma_vec=np.array([KAPPA_C, KAPPA_C]),
        drive_vector=CAVITY_READOUT,  # will change dynamically
        readout_vector=CAVITY_READOUT,
        phi_val=0,
    )

    # We'll define a range of kappa_y values that mimics "voltage"
    kappa_y_sweep = np.linspace(KAPPA_C, 1.0, 50)  # e.g. 50 steps
    # Frequencies (Hz) around cavity resonance, e.g. ±1 GHz
    lo_freqs_hz = np.linspace(params.cavity_freq - 1,
                              params.cavity_freq + 1,
                              2000) * 1e9  # convert to Hz

    # We'll define an experiment_id for the entire simulation
    experiment_id = "SIM_EXP_001"

    for kappa_y in kappa_y_sweep:
        # We'll store kappa_y as the "set_voltage" in the DB sense
        params.gamma_vec[1] = kappa_y

        # =============== YIG READOUT ===============
        params.readout_vector = YIG_READOUT
        params.drive_vector = YIG_READOUT
        params.J_val = J_OFF

        yig_response_func = sm.get_steady_state_response_transmission(symbols_dict, params)
        photon_numbers_yig = sm.compute_photon_numbers_transmission(yig_response_func, lo_freqs_hz)

        # Convert linear photon number to dBm
        # power (mW) ~ photon_numbers? Not strictly correct physically,
        # but let's treat photon_numbers as an arbitrary linear scale
        # we just do 10*log10(photon_numbers_yig + 1e-12)
        power_dBm_yig = 10.0 * np.log10(photon_numbers_yig + 1e-18)

        # Store each frequency point in DB row format
        for freq_hz, p_dbm in zip(lo_freqs_hz, power_dBm_yig):
            sim_data_rows.append({
                "experiment_id": experiment_id,
                "frequency_hz": freq_hz,
                "power_dBm": p_dbm,
                "readout_type": "yig",
                "omega_C": None,
                "omega_Y": None,
                "kappa_C": None,
                "kappa_Y": None,
                "Delta": None,
                "K": None,
                "set_loop_phase_deg": 0,
                "set_loop_att": 0,
                "set_loopback_att": 0,
                "set_yig_fb_phase_deg": 0,
                "set_yig_fb_att": 0,
                "set_cavity_fb_phase_deg": 0,
                "set_cavity_fb_att": 0,
                "set_voltage": kappa_y  # store kappa_y as "voltage"
            })

        # =============== CAVITY READOUT ===============
        params.readout_vector = CAVITY_READOUT
        params.drive_vector = CAVITY_READOUT
        params.J_val = J_OFF

        cav_response_func = sm.get_steady_state_response_transmission(symbols_dict, params)
        photon_numbers_cav = sm.compute_photon_numbers_transmission(cav_response_func, lo_freqs_hz)
        power_dBm_cav = 10.0 * np.log10(photon_numbers_cav + 1e-18)

        for freq_hz, p_dbm in zip(lo_freqs_hz, power_dBm_cav):
            sim_data_rows.append({
                "experiment_id": experiment_id,
                "frequency_hz": freq_hz,
                "power_dBm": p_dbm,
                "readout_type": "cavity",
                "omega_C": None,
                "omega_Y": None,
                "kappa_C": None,
                "kappa_Y": None,
                "Delta": None,
                "K": None,
                "set_loop_phase_deg": 0,
                "set_loop_att": 0,
                "set_loopback_att": 0,
                "set_yig_fb_phase_deg": 0,
                "set_yig_fb_att": 0,
                "set_cavity_fb_phase_deg": 0,
                "set_cavity_fb_att": 0,
                "set_voltage": kappa_y
            })

        # =============== HYBRID (normal) READOUT ===============
        params.readout_vector = CAVITY_READOUT
        params.drive_vector = CAVITY_READOUT
        params.J_val = J_COUPLING

        hybrid_response_func = sm.get_steady_state_response_transmission(symbols_dict, params)
        photon_numbers_hybrid = sm.compute_photon_numbers_transmission(hybrid_response_func, lo_freqs_hz)
        power_dBm_hybrid = 10.0 * np.log10(photon_numbers_hybrid + 1e-18)

        for freq_hz, p_dbm in zip(lo_freqs_hz, power_dBm_hybrid):
            sim_data_rows.append({
                "experiment_id": experiment_id,
                "frequency_hz": freq_hz,
                "power_dBm": p_dbm,
                "readout_type": "normal",
                "omega_C": None,
                "omega_Y": None,
                "kappa_C": None,
                "kappa_Y": None,
                "Delta": None,
                "K": None,
                "set_loop_phase_deg": 0,
                "set_loop_att": 0,
                "set_loopback_att": 0,
                "set_yig_fb_phase_deg": 0,
                "set_yig_fb_att": 0,
                "set_cavity_fb_phase_deg": 0,
                "set_cavity_fb_att": 0,
                "set_voltage": kappa_y
            })

    # Convert to a DataFrame
    sim_db_df = pd.DataFrame(sim_data_rows, columns=DB_COLUMNS)
    # Save as CSV
    sim_db_df.to_csv("simulation_db.csv", index=False)
    print(f"Simulation DB CSV created with {len(sim_db_df)} rows: simulation_db.csv")


if __name__ == '__main__':
    main()
