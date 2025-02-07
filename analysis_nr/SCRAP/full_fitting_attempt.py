#!/usr/bin/env python
"""
nr_detuning_fit.py

For every NR trace (indexed by detuning/current), use the optimal J (found previously)
to simulate a trace using the current’s cavity/YIG parameters. Then use the simulation’s
peak frequencies as initial guesses for a double Lorentzian fit on the experimental data.
An overlay plot is saved for each current.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sqlalchemy import create_engine

# Import your simulation routines and model definitions.
from symbolic_module import (
    setup_symbolic_equations,
    ModelParams,
    get_steady_state_response_transmission,
    compute_photon_numbers_transmission,
)

# Import your fitting function(s) from your existing code.
from nr_fitting import double_lorentzian_fit_NR, process_all_traces, compute_delta
# Import your data loader.
from nr_amperage_data_loader import NRAmperageDataLoader

# Configuration: frequency radius for simulation (in GHz)
FREQUENCY_RADIUS = 0.004

# Hardcoded optimal J (obtained from the max-detuning analysis)
optimal_J = 0.00848  # Adjust this value as needed

# Vertical offset for simulation trace (as used in your previous analysis)
VERTICAL_OFFSET = 10.5


# ----------------------------------------------------------------------------
# Helper function to simulate an NR trace given parameters.
def simulate_nr_trace(J_val, freq_axis, cavity_freq, w_y, gamma_vec, symbols_dict, vertical_offset=VERTICAL_OFFSET):
    params = ModelParams(
        J_val=J_val,
        g_val=0,
        cavity_freq=cavity_freq,
        w_y=w_y,
        gamma_vec=gamma_vec,
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([0, 1]),
        phi_val=np.deg2rad(180),
    )
    eqn = get_steady_state_response_transmission(symbols_dict, params)
    photon_numbers = compute_photon_numbers_transmission(eqn, freq_axis)
    sim_trace = np.log10(photon_numbers) - vertical_offset
    return sim_trace


# ----------------------------------------------------------------------------
# Function to load configuration from JSON.
def load_config(config_path, config_name="default"):
    with open(config_path, "r") as file:
        all_configs = json.load(file)
    if config_name not in all_configs:
        raise KeyError(f"Configuration '{config_name}' not found in {config_path}.")
    return all_configs[config_name]


# ----------------------------------------------------------------------------
def main():
    # Load configuration (adjust path and config name as needed)
    config_path = "nr_expr_config.json"
    config_name = "hailey_sunday"
    config = load_config(config_path, config_name)

    experiment_id = config["experiment_id"]

    # Frequency limits (Hz)
    colorplot_freq_min = config["frequency_limits"]["colorplot"]["min"]
    colorplot_freq_max = config["frequency_limits"]["colorplot"]["max"]
    cavity_freq_min = config["frequency_limits"]["cavity"]["min"]
    cavity_freq_max = config["frequency_limits"]["cavity"]["max"]
    yig_freq_min = config["frequency_limits"]["yig"]["min"]
    yig_freq_max = config["frequency_limits"]["yig"]["max"]

    # Current limits
    current_min = config["current_limits"]["min"]
    current_max = config["current_limits"]["max"]

    # Database path
    db_path = config["db_path"]

    # Create data loaders for each readout type.
    cavity_loader = NRAmperageDataLoader(db_path, experiment_id, "cavity",
                                         cavity_freq_min, cavity_freq_max, current_min, current_max)
    yig_loader = NRAmperageDataLoader(db_path, experiment_id, "yig",
                                      yig_freq_min, yig_freq_max, current_min, current_max)
    nr_loader = NRAmperageDataLoader(db_path, experiment_id, "nr",
                                     colorplot_freq_min, colorplot_freq_max, current_min, current_max)

    cavity_power, cavity_currents, cavity_freqs, cavity_settings = cavity_loader.load_data()
    yig_power, yig_currents, yig_freqs, yig_settings = yig_loader.load_data()
    nr_power, nr_currents, nr_freqs, nr_settings = nr_loader.load_data()

    if cavity_power is None or yig_power is None or nr_power is None:
        print("Error: One or more datasets are missing. Exiting.")
        return

    # Assume that the NR data comes as a pivot table so that nr_freqs is common for all currents.
    # (nr_freqs are in Hz; convert to GHz.)
    exp_freqs = nr_freqs / 1e9

    # Process cavity and YIG traces and compute detuning (Δ = ω_c - ω_y).
    cavity_results = process_all_traces(cavity_power, cavity_currents, cavity_freqs)
    yig_results = process_all_traces(yig_power, yig_currents, yig_freqs)
    cavity_df = pd.DataFrame(cavity_results)
    yig_df = pd.DataFrame(yig_results)
    delta_df = compute_delta(cavity_df, yig_df)
    print("Computed detuning for each current.")

    # Build maps from current to cavity and YIG parameters.
    cavity_map = cavity_df.set_index("current")[["omega", "kappa"]].to_dict(orient="index")
    yig_map = yig_df.set_index("current")[["omega", "kappa"]].to_dict(orient="index")
    delta_map = delta_df.set_index("current")["Delta"].to_dict()

    # Setup symbolic equations (independent of J).
    symbols_dict = setup_symbolic_equations()

    # Prepare output folder for overlay plots.
    output_folder = os.path.join("plots", f"{experiment_id}_detuning_fits")
    os.makedirs(output_folder, exist_ok=True)

    # For summary results.
    summary_results = []

    # Loop over each NR trace (by current) for which we have detuning and cavity/YIG parameters.
    for idx, cur in enumerate(nr_currents):
        if cur not in delta_map or cur not in cavity_map or cur not in yig_map:
            continue

        delta_val = delta_map[cur]
        # Retrieve experimental NR trace (in dBm) for this current.
        exp_trace_dbm = nr_power[idx, :]  # array of dBm values
        # (Assume that the experimental frequency axis is exp_freqs, in GHz.)

        # Get cavity and YIG parameters for this current.
        omega_c = cavity_map[cur]["omega"]  # GHz
        kappa_c = cavity_map[cur]["kappa"]
        omega_y = yig_map[cur]["omega"]  # GHz
        kappa_y = yig_map[cur]["kappa"]
        gamma_vec = np.array([kappa_c, kappa_y])

        # Define simulation frequency axis centered on the current’s mean frequency.
        mean_freq_current = (omega_c + omega_y) / 2.0  # GHz
        sim_freqs = np.linspace(mean_freq_current - FREQUENCY_RADIUS,
                                mean_freq_current + FREQUENCY_RADIUS, 1000)

        # Simulate NR trace using the optimal J.
        sim_trace = simulate_nr_trace(optimal_J, sim_freqs, omega_c, omega_y, gamma_vec, symbols_dict,
                                      vertical_offset=VERTICAL_OFFSET)

        # --- Use the simulation trace peaks as initial guesses for the double Lorentzian fit.
        sim_peaks_idx, _ = find_peaks(sim_trace, prominence=0.01)
        if len(sim_peaks_idx) < 2:
            print(f"Warning: Fewer than 2 peaks found in simulation trace for current = {cur} A. Skipping.")
            continue
        elif len(sim_peaks_idx) > 2:
            # Choose the two highest peaks.
            peak_vals = sim_trace[sim_peaks_idx]
            best_two_indices = np.argsort(peak_vals)[-2:]
            sim_peaks_idx = np.array(sim_peaks_idx)[best_two_indices]
        sim_peaks_freq = np.sort(sim_freqs[sim_peaks_idx])
        # (These are our initial guesses, in GHz.)

        # --- Prepare experimental trace for fitting.
        # Convert experimental data from dBm to linear scale.
        exp_trace_linear = 10 ** (exp_trace_dbm / 10)

        # Perform double Lorentzian fit on the experimental data using simulation peaks as initial guesses.
        fit_result = double_lorentzian_fit_NR(exp_freqs, exp_trace_linear,
                                              sim_peaks_freq[0], sim_peaks_freq[1])
        if fit_result is None:
            print(f"Double Lorentzian fit failed for current = {cur} A.")
            continue

        # Extract the fitted peak centers (in GHz) and sort.
        fitted_peak1 = fit_result.params["lz1_center"].value
        fitted_peak2 = fit_result.params["lz2_center"].value
        fitted_peaks = np.sort(np.array([fitted_peak1, fitted_peak2]))

        # Evaluate the fitted curve for plotting.
        x_fit = np.linspace(exp_freqs.min(), exp_freqs.max(), 1000)
        fit_curve_linear = fit_result.eval(x=x_fit)
        fit_curve_dbm = 10 * np.log10(fit_curve_linear)

        # For markers: evaluate the fitted curve at the fitted peak frequencies.
        fitted_y_linear = fit_result.eval(x=fitted_peaks)
        fitted_y_dbm = 10 * np.log10(fitted_y_linear)

        # --- Plot overlay:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot experimental trace (dBm).
        ax.plot(exp_freqs, exp_trace_dbm, label="Experimental Data", color="blue")
        # Plot fitted curve (converted to dBm).
        ax.plot(x_fit, fit_curve_dbm, label="Double Lorentzian Fit", color="green", linestyle="--")
        # Plot simulation trace (native units: log10(photon number)) versus sim_freqs.
        ax.plot(sim_freqs, sim_trace, label=f"Simulated Trace (J = {optimal_J:.6f})", color="red", linestyle=":")
        ax.set_xlabel("Frequency (GHz)", fontsize=14)
        ax.set_ylabel("Power (dBm) / log10(Photon Number)", fontsize=14)
        ax.set_title(f"Current = {cur} A, Δ = {delta_val} GHz", fontsize=14)
        ax.grid(True)

        # Markers:
        # a) Simulation initial guesses (magenta stars) on simulation trace.
        sim_marker_y = [sim_trace[np.argmin(np.abs(sim_freqs - f))] for f in sim_peaks_freq]
        ax.plot(sim_peaks_freq, sim_marker_y, 'm*', markersize=12, label="Sim Initial Guesses")
        # b) Fitted peaks (cyan stars) on the fitted curve.
        ax.plot(fitted_peaks, fitted_y_dbm, 'c*', markersize=12, label="Fitted Peaks")

        ax.legend()
        plt.tight_layout()

        # Save the overlay plot.
        plot_filename = f"nr_trace_current_{cur}_Delta_{delta_val}.png"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved overlay plot for current {cur} A (Δ = {delta_val} GHz) to {plot_path}")

        # Append summary info.
        summary_results.append({
            "current": cur,
            "Delta": delta_val,
            "sim_peak1": sim_peaks_freq[0],
            "sim_peak2": sim_peaks_freq[1],
            "fitted_peak1": fitted_peaks[0],
            "fitted_peak2": fitted_peaks[1]
        })

    # Optionally, save the summary results as CSV.
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(output_folder, f"nr_detuning_summary_{experiment_id}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV to {summary_csv_path}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
