#!/usr/bin/env python3
"""
Pure Theory Simulation & Sensitivity Analysis for NREP (or similar EPs)
-----------------------------------------------------------------------
This module simulates the NREP response using a Monte Carlo approach
and produces a baseline theory plot (with shaded uncertainty regions)
based on constant (average) parameters.

It also includes a sensitivity analysis routine that varies the uncertainty
in one parameter at a time (using raw uncertainty values in GHz) while setting
all other parameter uncertainties to zero. For each parameter under test, a single
plot is generated (one axis) that overlays both the lower and upper branch uncertainty
shading and the average simulation curves.
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import concurrent.futures
import matplotlib.cm as cm

import nr_simulation as nr_sim
# This module must provide:
#   - simulate_trace(J, omega_c, omega_y, kappa_c, kappa_y, freqs)
#   - fast_simulate_trace(fast_func, J, omega_c, omega_y, kappa_c, kappa_y, freqs)
#   - setup_fast_simulation(...)

# Global simulation switches and parameters
USE_PARALLEL = False
NUM_SIMULATION_SHOTS = 100


def run_single_theory_shot_fast(
        fast_func,
        J_val, J_val_unc,
        omega_c, omega_c_unc,
        omega_y, omega_y_unc,
        kappa_c, kappa_c_unc,
        kappa_y, kappa_y_unc,
        nr_freqs
):
    """
    Perform one Monte Carlo shot:
      1. Draw random parameters from normal distributions.
      2. Simulate the NR trace using the fast simulation function.
      3. Find peaks in the simulated trace and return them as a two‐element list [lower, upper].
    """
    J_val_sim = np.random.normal(J_val, J_val_unc)
    omega_c_sim = np.random.normal(omega_c, omega_c_unc)
    omega_y_sim = np.random.normal(omega_y, omega_y_unc)
    kappa_c_sim = np.random.normal(kappa_c, kappa_c_unc)
    kappa_y_sim = np.random.normal(kappa_y, kappa_y_unc)

    sim_trace = nr_sim.fast_simulate_trace(
        fast_func,
        J_val_sim, omega_c_sim, omega_y_sim, kappa_c_sim, kappa_y_sim,
        nr_freqs / 1e9  # pass frequencies in GHz
    )

    sim_peaks_idx, _ = find_peaks(sim_trace, prominence=0.00001)
    peak_freqs = nr_freqs[sim_peaks_idx] / 1e9  # convert to GHz
    peak_freqs.sort()

    if len(peak_freqs) == 2:
        return list(peak_freqs)  # [lower, upper]
    elif len(peak_freqs) == 1:
        # Duplicate single peak if only one is found
        return [peak_freqs[0], peak_freqs[0]]
    else:
        print("WARNING: No peaks found in this simulation shot.")
        return [np.nan, np.nan]


def simulate_theory_ep(detuning_array, nr_freqs, theory_params, num_shots=NUM_SIMULATION_SHOTS,
                       use_parallel=USE_PARALLEL):
    """
    For each detuning value (Δ = ω_c - ω_y) in detuning_array, run a theory simulation.
    For each detuning:
      - Compute ω_y = ω_c - Δ.
      - Use the average parameters to simulate the “average” trace and extract the peak frequencies.
      - Run num_shots Monte Carlo simulations (using run_single_theory_shot_fast)
        to obtain uncertainty bounds.

    Returns:
      detuning_array: 1D array of detuning values (GHz)
      avg_lower_peaks, avg_upper_peaks: average simulation peak frequencies (GHz)
      lower_min, lower_max, upper_min, upper_max: MC min/max arrays (GHz)
    """
    avg_lower_peaks = []
    avg_upper_peaks = []
    lower_min_arr = []
    lower_max_arr = []
    upper_min_arr = []
    upper_max_arr = []

    fast_func = nr_sim.setup_fast_simulation(drive=(1, 0), readout=(0, 1))

    for det in detuning_array:
        # For each detuning, assume ω_c is fixed and set ω_y = ω_c - det.
        omega_c = theory_params["omega_c"]
        omega_y = omega_c - det  # so that detuning Δ = ω_c - ω_y

        # Compute the average simulation trace (using nominal parameter values)
        sim_trace_avg = nr_sim.simulate_trace(
            theory_params["J_val"], omega_c, omega_y,
            theory_params["kappa_c"], theory_params["kappa_y"],
            nr_freqs / 1e9
        )
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)
        avg_peaks = nr_freqs[sim_peaks_idx_avg] / 1e9  # convert to GHz
        avg_peaks.sort()
        if len(avg_peaks) == 2:
            avg_lower, avg_upper = avg_peaks[0], avg_peaks[1]
        elif len(avg_peaks) == 1:
            avg_lower, avg_upper = avg_peaks[0], avg_peaks[0]
        else:
            avg_lower, avg_upper = np.nan, np.nan

        avg_lower_peaks.append(avg_lower)
        avg_upper_peaks.append(avg_upper)

        # Run Monte Carlo simulation to obtain uncertainty bounds.
        all_peak_freqs = []
        if use_parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for shot in range(num_shots):
                    futures.append(
                        executor.submit(
                            run_single_theory_shot_fast,
                            fast_func,
                            theory_params["J_val"], theory_params["J_val_unc"],
                            omega_c, theory_params["omega_c_unc"],
                            omega_y, theory_params["omega_y_unc"],
                            theory_params["kappa_c"], theory_params["kappa_c_unc"],
                            theory_params["kappa_y"], theory_params["kappa_y_unc"],
                            nr_freqs
                        )
                    )
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_peak_freqs.append(result)
        else:
            for shot in range(num_shots):
                result = run_single_theory_shot_fast(
                    fast_func,
                    theory_params["J_val"], theory_params["J_val_unc"],
                    omega_c, theory_params["omega_c_unc"],
                    omega_y, theory_params["omega_y_unc"],
                    theory_params["kappa_c"], theory_params["kappa_c_unc"],
                    theory_params["kappa_y"], theory_params["kappa_y_unc"],
                    nr_freqs
                )
                all_peak_freqs.append(result)
        all_peak_freqs = np.array(all_peak_freqs)  # shape: (num_shots, 2)

        lower_min = np.nanmin(all_peak_freqs[:, 0])
        lower_max = np.nanmax(all_peak_freqs[:, 0])
        upper_min = np.nanmin(all_peak_freqs[:, 1])
        upper_max = np.nanmax(all_peak_freqs[:, 1])

        lower_min_arr.append(lower_min)
        lower_max_arr.append(lower_max)
        upper_min_arr.append(upper_min)
        upper_max_arr.append(upper_max)

    return (np.array(detuning_array),
            np.array(avg_lower_peaks), np.array(avg_upper_peaks),
            np.array(lower_min_arr), np.array(lower_max_arr),
            np.array(upper_min_arr), np.array(upper_max_arr))


def plot_theory_ep_results(detuning_array, avg_lower, avg_upper,
                           lower_min, lower_max, upper_min, upper_max,
                           output_path):
    """
    Create a plot showing the theoretical EP results:
      - Shaded regions (fill_between) for the lower and upper branch uncertainties.
      - Solid lines for the average simulation peaks.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Shading for the lower branch.
    ax.fill_between(detuning_array, lower_min, lower_max,
                    color="blue", alpha=0.5, label="Lower branch bounds")
    # Shading for the upper branch.
    ax.fill_between(detuning_array, upper_min, upper_max,
                    color="red", alpha=0.5, label="Upper branch bounds")

    # Plot average simulation peaks.
    ax.plot(detuning_array, avg_lower, color="blue", lw=2, linestyle="--", label="Average lower peak")
    ax.plot(detuning_array, avg_upper, color="red", lw=2, linestyle="-.", label="Average upper peak")

    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=14)
    ax.set_title("Theoretical NREP with Uncertainties", fontsize=14)
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Saved theory EP plot to", output_path)


def sensitivity_analysis_for_param(param_key, unc_values, detuning_array, nr_freqs, base_params, output_folder):
    """
    Perform a sensitivity analysis for one parameter.

    For the parameter specified by param_key (e.g. "J_val", "omega_c", "kappa_c", "kappa_y",
    or "omega_y_unc"), vary its uncertainty using a list of raw uncertainty values (in GHz).
    For each raw uncertainty value, all other uncertainties are set to 0. Then, simulate the theory EP
    response and store the MC uncertainty bounds.

    This function then creates a single plot (one axis) overlaying the fill_between shading
    and average simulation curves for both the lower and upper branches.
    """
    # For parameters other than omega_y_unc, the corresponding uncertainty key is param_key+"_unc".
    # For "omega_y_unc" we simply use that key.
    if param_key == "omega_y_unc":
        test_unc_key = "omega_y_unc"
    else:
        test_unc_key = param_key + "_unc"

    results = {}  # Will store simulation results for each raw uncertainty value.
    for unc_val in unc_values:
        mod_params = copy.deepcopy(base_params)
        # Zero out all uncertainties.
        for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "omega_y_unc"]:
            mod_params[key] = 0.0
        # For the parameter under test, set its uncertainty to the raw value.
        mod_params[test_unc_key] = unc_val

        (det_arr, avg_lower, avg_upper,
         lower_min, lower_max, upper_min, upper_max) = simulate_theory_ep(
            detuning_array, nr_freqs, mod_params,
            num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL
        )
        results[unc_val] = (det_arr, avg_lower, avg_upper, lower_min, lower_max, upper_min, upper_max)

    # Create a single plot for this parameter’s sensitivity.
    # Here we use the plasma colormap and limit its range to avoid pale yellow.
    color_values = cm.jet(np.linspace(0, 1, len(unc_values)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, unc_val in enumerate(sorted(results.keys())):
        det_arr, avg_lower, avg_upper, lower_min, lower_max, upper_min, upper_max = results[unc_val]
        color = color_values[i]
        # Plot shading for both branches.
        ax.fill_between(det_arr, lower_min, lower_max, color=color, alpha=0.5)
        ax.fill_between(det_arr, upper_min, upper_max, color=color, alpha=0.5)
        # Plot the average simulation curves.
        ax.plot(det_arr, avg_lower, color=color, lw=2, linestyle="--")
        ax.plot(det_arr, avg_upper, color=color, lw=2, linestyle="-.")
        # Label once per uncertainty value.
        ax.plot([], [], color=color, lw=2, label=f"uncertainty = {1000 * unc_val:.1f} MHz")

    ax.set_title(f"Sensitivity of {test_unc_key} on NREP Simulation", fontsize=16)
    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"sensitivity_{test_unc_key}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved sensitivity plot for {test_unc_key} to {out_path}")
    return results


def sensitivity_main():
    """
    Run the sensitivity analysis. For each parameter of interest, vary its uncertainty
    (using raw uncertainty values in GHz) while setting all other uncertainties to zero.
    Then produce a sensitivity plot showing how the MC (shaded) uncertainty bounds vary.
    """
    # Define base theory parameters (all in GHz)
    base_params = {
        "J_val": 0.05,         # Nominal coupling J (GHz)
        "J_val_unc": 0.005,      # (Will be varied in sensitivity analysis)
        "omega_c": 6.0,        # Cavity frequency (GHz)
        "omega_c_unc": 0.01,     # (Will be varied)
        "kappa_c": 0.01,      # Cavity linewidth (GHz)
        "kappa_c_unc": 0.0001,   # (Will be varied)
        "kappa_y": 0.01,      # YIG linewidth (GHz)
        "kappa_y_unc": 0.0001,   # (Will be varied)
        "omega_y_unc": 0.01      # Uncertainty in YIG frequency (GHz; will be varied)
    }
    # Note: The simulation uses omega_y = omega_c - detuning.

    # Create a frequency axis for simulation (in Hz)
    nr_freqs = np.linspace(5.8e9, 6.15e9, 1000)

    # Define a range of detuning values (Δ = ω_c - ω_y) in GHz.
    detuning_array = np.linspace(0.075, 0.125, 100)

    # Define the raw uncertainty values (in GHz) to test. (e.g. 0.1 MHz to 1 MHz)
    unc_values = [0.1 / 1000, 0.25 / 1000, 0.5 / 1000, 1 / 1000]

    # Define which parameters to test.
    # (For parameters other than omega_y_unc, the sensitivity function will use param+"_unc".)
    params_to_test = ["J_val", "omega_c", "kappa_c", "kappa_y", "omega_y_unc"]

    # Create an output folder for the sensitivity plots.
    output_folder = "sensitivity_plots"
    os.makedirs(output_folder, exist_ok=True)

    # Loop over each parameter and run the sensitivity analysis.
    for param in params_to_test:
        sensitivity_analysis_for_param(param, unc_values, detuning_array, nr_freqs, base_params, output_folder)


if __name__ == "__main__":
    sensitivity_main()
