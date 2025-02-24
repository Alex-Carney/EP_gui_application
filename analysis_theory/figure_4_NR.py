#!/usr/bin/env python3
"""
Pure Theory Simulation & Sensitivity Analysis for NREP (or similar EPs)
-----------------------------------------------------------------------
This module simulates the NREP response using a Monte Carlo approach
and produces a baseline theory plot (with shaded uncertainty regions)
based on constant (average) parameters.

It also includes a sensitivity analysis routine. In this modified version,
we use a single uncertainty value (1 MHz) for each parameter group and produce
a single plot with three vertically stacked subplots (sharing the same X axis)
for:
  - J uncertainty (only J is varied),
  - Omega uncertainty (combined f_c and f_y, in different colors),
  - Kappa uncertainty (combined κ_c and κ_y, in different colors).
"""

# Global font size settings for plots and opacity
LEGEND_FONT_SIZE = 21
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 25
OPACITY = 0.25  # Adjust this to change the opacity of fill regions

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


# make matplotlib use bolded arial for text
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"

# Global simulation switches and parameters
USE_PARALLEL = False
NUM_SIMULATION_SHOTS = 100

SEED = 2345
np.random.seed(SEED)


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
    For each detuning value (Δ = f_c - f_y) in detuning_array, run a theory simulation.
    For each detuning:
      - Compute f_y = f_c - Δ.
      - Use the average parameters to simulate the “average” trace and extract the peak frequencies.
      - Run num_shots Monte Carlo simulations (using run_single_theory_shot_fast)
        to obtain uncertainty bounds.

    Returns:
      detuning_array: 1D array of detuning values [Ghz]
      avg_lower_peaks, avg_upper_peaks: average simulation peak frequencies [Ghz]
      lower_min, lower_max, upper_min, upper_max: MC min/max arrays [Ghz]
    """
    avg_lower_peaks = []
    avg_upper_peaks = []
    lower_min_arr = []
    lower_max_arr = []
    upper_min_arr = []
    upper_max_arr = []

    fast_func = nr_sim.setup_fast_simulation(drive=(1, 0), readout=(0, 1))

    for det in detuning_array:
        # For each detuning, assume f_c is fixed and set f_y = f_c - det.
        omega_c = theory_params["omega_c"]
        omega_y = omega_c - det  # so that detuning Δ = f_c - f_y

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


# Note: The original plot function is not used in the combined plot.
def plot_theory_ep_results(detuning_array, avg_lower, avg_upper,
                           lower_min, lower_max, upper_min, upper_max,
                           output_path, J_val):
    """
    Create a plot showing the theoretical EP results:
      - Shaded regions (fill_between) for the lower and upper branch uncertainties.
      - Solid lines for the average simulation peaks.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Shading for the lower branch.
    ax.fill_between(detuning_array, lower_min, lower_max,
                    color="blue", alpha=OPACITY, label="J uncertainty")
    # Shading for the upper branch.
    ax.fill_between(detuning_array, upper_min, upper_max,
                    color="blue", alpha=OPACITY)

    # ax.axvline(x=2 * J_val, color="red", linestyle="--", label="EP Line")

    ax.set_xlabel("Δf [Ghz]", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    ax.legend(loc="lower left", fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print("Saved theory EP plot to", output_path)


def sensitivity_main():
    """
    Modified sensitivity analysis:
      - Uses a single uncertainty value of 1 MHz (0.001 GHz).
      - Produces a single plot with three vertically stacked subplots:
          1. J uncertainty (only J is varied),
          2. Omega uncertainty (combined f_c and f_y, in different colors),
          3. Kappa uncertainty (combined κ_c and κ_y, in different colors).
      All subplots share the same X axis (Detuning Δ in GHz) and have no titles.
    """
    # Define base theory parameters (all in GHz)
    base_params = {
        "J_val": 0.05,      # Nominal coupling J [Ghz]
        "J_val_unc": 0.005, # (Will be overridden)
        "omega_c": 6.0,     # Cavity frequency [Ghz]
        "omega_c_unc": 0.01,# (Will be overridden)
        "kappa_c": 0.01,    # Cavity linewidth [Ghz]
        "kappa_c_unc": 0.0001, # (Will be overridden)
        "kappa_y": 0.01,    # YIG linewidth [Ghz]
        "kappa_y_unc": 0.0001, # (Will be overridden)
        "omega_y_unc": 0.01 # (Will be overridden)
    }
    # Note: The simulation uses omega_y = omega_c - detuning.

    # Create a frequency axis for simulation (in Hz)
    nr_freqs = np.linspace(5.8e9, 6.15e9, 1000)

    # Define a range of detuning values (Δ = f_c - f_y) in GHz.
    detuning_array = np.linspace(0.075, 0.125, 100)

    # Use a single uncertainty value of 1 MHz (1 MHz = 0.001 GHz)
    unc_value = 1 / 1000  # 0.001 GHz

    # Create an output folder for the sensitivity plots.
    output_folder = "fig4_NR"
    os.makedirs(output_folder, exist_ok=True)

    J_val = base_params["J_val"]

    # --- 1. Sensitivity for J ---
    mod_params_J = copy.deepcopy(base_params)
    # Zero out all uncertainties.
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "omega_y_unc"]:
        mod_params_J[key] = 0.0
    mod_params_J["J_val_unc"] = unc_value

    (det_arr_J, avg_lower_J, avg_upper_J,
     lower_min_J, lower_max_J, upper_min_J, upper_max_J) = simulate_theory_ep(
        detuning_array, nr_freqs, mod_params_J,
        num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL
    )

    # --- 2. Sensitivity for Omega (combined f_c and f_y) ---
    mod_params_omega_c = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "omega_y_unc"]:
        mod_params_omega_c[key] = 0.0
    mod_params_omega_c["omega_c_unc"] = unc_value
    results_omega_c = simulate_theory_ep(detuning_array, nr_freqs, mod_params_omega_c,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    mod_params_omega_y = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "omega_y_unc"]:
        mod_params_omega_y[key] = 0.0
    mod_params_omega_y["omega_y_unc"] = unc_value
    results_omega_y = simulate_theory_ep(detuning_array, nr_freqs, mod_params_omega_y,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    (det_arr_omega, avg_lower_c, avg_upper_c, lower_min_c, lower_max_c, upper_min_c, upper_max_c) = results_omega_c
    (det_arr_omega, avg_lower_y, avg_upper_y, lower_min_y, lower_max_y, upper_min_y, upper_max_y) = results_omega_y

    # --- 3. Sensitivity for Kappa (combined κ_c and κ_y) ---
    mod_params_kappa_c = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "omega_y_unc"]:
        mod_params_kappa_c[key] = 0.0
    mod_params_kappa_c["kappa_c_unc"] = unc_value
    results_kappa_c = simulate_theory_ep(detuning_array, nr_freqs, mod_params_kappa_c,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    mod_params_kappa_y = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "omega_y_unc"]:
        mod_params_kappa_y[key] = 0.0
    mod_params_kappa_y["kappa_y_unc"] = unc_value
    results_kappa_y = simulate_theory_ep(detuning_array, nr_freqs, mod_params_kappa_y,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    (det_arr_kappa, avg_lower_kc, avg_upper_kc, lower_min_kc, lower_max_kc, upper_min_kc, upper_max_kc) = results_kappa_c
    (det_arr_kappa, avg_lower_ky, avg_upper_ky, lower_min_ky, lower_max_ky, upper_min_ky, upper_max_ky) = results_kappa_y

    # Create a single figure with 3 vertically stacked subplots sharing the same X-axis
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 12))

    # Subplot 1: J uncertainty
    axs[0].fill_between(det_arr_J, lower_min_J, lower_max_J,
                        color="blue", alpha=OPACITY, label="J uncertainty")
    axs[0].fill_between(det_arr_J, upper_min_J, upper_max_J,
                        color="blue", alpha=OPACITY)
    axs[0].axvline(x=2 * J_val, color="red", linestyle="--", label="EP Line")
    axs[0].set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[0].tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axs[0].legend(loc="lower left", fontsize=LEGEND_FONT_SIZE)
    axs[0].grid(False)

    # Subplot 2: Omega uncertainty
    axs[1].fill_between(det_arr_omega, lower_min_c, lower_max_c,
                        color="blue", alpha=OPACITY, label="$f_c$ uncertainty")
    axs[1].fill_between(det_arr_omega, upper_min_c, upper_max_c,
                        color="blue", alpha=OPACITY)
    axs[1].fill_between(det_arr_omega, lower_min_y, lower_max_y,
                        color="red", alpha=OPACITY, label="$f_y$ uncertainty")
    axs[1].fill_between(det_arr_omega, upper_min_y, upper_max_y,
                        color="red", alpha=OPACITY)
    axs[1].axvline(x=2 * J_val, color="red", linestyle="--", label="EP Line")
    axs[1].set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[1].tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axs[1].legend(loc="lower left", fontsize=LEGEND_FONT_SIZE)
    axs[1].grid(False)

    # Subplot 3: Kappa uncertainty
    axs[2].fill_between(det_arr_kappa, lower_min_kc, lower_max_kc,
                        color="blue", alpha=OPACITY, label="$κ_c$ uncertainty")
    axs[2].fill_between(det_arr_kappa, upper_min_kc, upper_max_kc,
                        color="blue", alpha=OPACITY)
    axs[2].fill_between(det_arr_kappa, lower_min_ky, lower_max_ky,
                        color="red", alpha=OPACITY, label="$κ_y$ uncertainty")
    axs[2].fill_between(det_arr_kappa, upper_min_ky, upper_max_ky,
                        color="red", alpha=OPACITY)
    axs[2].axvline(x=2 * J_val, color="red", linestyle="--", label="EP Line")
    axs[2].set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[2].set_xlabel("Δf [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[2].tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axs[2].legend(loc="lower left", fontsize=LEGEND_FONT_SIZE)
    axs[2].grid(False)

    plt.tight_layout()
    combined_output_path = os.path.join(output_folder, "sensitivity_combined.png")
    plt.savefig(combined_output_path, dpi=400)
    plt.close(fig)
    print("Saved combined sensitivity plot to", combined_output_path)


if __name__ == "__main__":
    sensitivity_main()
