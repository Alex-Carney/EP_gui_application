#!/usr/bin/env python3
"""
Pure Theory Simulation & Sensitivity Analysis for PTEP (or similar EPs)
-----------------------------------------------------------------------
This module simulates the PTEP response using a Monte Carlo approach
and produces a baseline theory plot (with shaded uncertainty regions)
based on constant (average) parameters.

It also includes a sensitivity analysis routine. In this modified version,
we use a single uncertainty value (1 MHz) for each parameter group and produce
a single plot with three vertically stacked subplots (sharing the same X axis)
for:
  - J uncertainty (only J is varied),
  - Omega uncertainty (combined ω₍c₎ and ω₍y₎, in different colors),
  - Kappa uncertainty (combined κ₍c₎ and κ₍y₎, in different colors).
"""

# Global font size settings for plots and opacity
LEGEND_FONT_SIZE = 21
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 25
OPACITY = 0.25  # Change this value to adjust opacity of fill regions

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import concurrent.futures
import matplotlib.cm as cm

import nr_simulation as pt_sim
# This module must provide:
#   - simulate_trace(J, omega_c, omega_y, kappa_c, kappa_y, freqs)
#   - fast_simulate_trace(fast_func, J, omega_c, omega_y, kappa_c, kappa_y, freqs)
#   - setup_fast_simulation(...)

# Global simulation switches and parameters
plt.rcParams["font.family"] = "sans-serif"


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
      2. Simulate the PTEP trace using the fast simulation function.
      3. Find peaks in the simulated trace and return them as a two‐element list [lower, upper].
    """
    J_val_sim = np.random.normal(J_val, J_val_unc)
    omega_c_sim = np.random.normal(omega_c, omega_c_unc)
    omega_y_sim = np.random.normal(omega_y, omega_y_unc)
    kappa_c_sim = np.random.normal(kappa_c, kappa_c_unc)
    kappa_y_sim = np.random.normal(kappa_y, kappa_y_unc)

    sim_trace = pt_sim.fast_simulate_trace(
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
        return [peak_freqs[0], peak_freqs[0]]
    else:
        print("WARNING: No peaks found in this simulation shot.")
        return [np.nan, np.nan]


###############################################################################
# For all sensitivities (J, Omega, and Kappa), the independent variable is the k‐array.
def simulate_theory_ep(k_array, nr_freqs, theory_params, num_shots=NUM_SIMULATION_SHOTS,
                       use_parallel=USE_PARALLEL):
    """
    For each value in k_array, run a theory simulation.
    For each k:
      - Compute kappa_c from theory_params and kappa_y = kappa_c - k.
      - Use the average parameters to simulate the “average” trace and extract the peak frequencies.
      - Run num_shots Monte Carlo simulations to obtain uncertainty bounds.

    Returns:
      k_array: 1D array of k values (MHz)
      avg_lower_peaks, avg_upper_peaks: average simulation peak frequencies [Ghz]
      lower_min, lower_max, upper_min, upper_max: MC min/max arrays [Ghz]
    """
    avg_lower_peaks = []
    avg_upper_peaks = []
    lower_min_arr = []
    lower_max_arr = []
    upper_min_arr = []
    upper_max_arr = []

    fast_func = pt_sim.setup_fast_simulation(drive=(1, 0), readout=(1, 0))

    for k in k_array:
        kappa_c = theory_params["kappa_c"]
        kappa_y = kappa_c - k

        # Compute the average simulation trace using nominal parameters
        sim_trace_avg = pt_sim.simulate_trace(
            theory_params["J_val"], theory_params["omega_c"], theory_params["omega_y"],
            kappa_c, kappa_y, nr_freqs / 1e9
        )
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)
        avg_peaks = nr_freqs[sim_peaks_idx_avg] / 1e9
        avg_peaks.sort()
        if len(avg_peaks) == 2:
            avg_lower, avg_upper = avg_peaks[0], avg_peaks[1]
        elif len(avg_peaks) == 1:
            avg_lower, avg_upper = avg_peaks[0], avg_peaks[0]
        else:
            avg_lower, avg_upper = np.nan, np.nan

        avg_lower_peaks.append(avg_lower)
        avg_upper_peaks.append(avg_upper)

        # Run Monte Carlo simulation for uncertainty bounds
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
                            theory_params["omega_c"], theory_params["omega_c_unc"],
                            theory_params["omega_y"], theory_params["omega_y_unc"],
                            kappa_c, theory_params["kappa_c_unc"],
                            kappa_y, theory_params["kappa_y_unc"],
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
                    theory_params["omega_c"], theory_params["omega_c_unc"],
                    theory_params["omega_y"], theory_params["omega_y_unc"],
                    kappa_c, theory_params["kappa_c_unc"],
                    kappa_y, theory_params["kappa_y_unc"],
                    nr_freqs
                )
                all_peak_freqs.append(result)
        all_peak_freqs = np.array(all_peak_freqs)

        lower_min = np.nanmin(all_peak_freqs[:, 0])
        lower_max = np.nanmax(all_peak_freqs[:, 0])
        upper_min = np.nanmin(all_peak_freqs[:, 1])
        upper_max = np.nanmax(all_peak_freqs[:, 1])

        lower_min_arr.append(lower_min)
        lower_max_arr.append(lower_max)
        upper_min_arr.append(upper_min)
        upper_max_arr.append(upper_max)

    return (np.array(k_array),
            np.array(avg_lower_peaks), np.array(avg_upper_peaks),
            np.array(lower_min_arr), np.array(lower_max_arr),
            np.array(upper_min_arr), np.array(upper_max_arr))


###############################################################################
# Plotting function (unused in the combined plot)
def plot_theory_ep_results(det_arr, avg_lower, avg_upper,
                           lower_min, lower_max, upper_min, upper_max,
                           output_path):
    """
    Create a plot showing the theoretical PTEP results with two branches:
      - Shaded regions for the lower and upper branch uncertainties.
      - Dashed lines for the average simulation peaks.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.fill_between(det_arr, lower_min, lower_max,
                    color="blue", alpha=OPACITY, label="J uncertainty")
    ax.fill_between(det_arr, upper_min, upper_max,
                    color="blue", alpha=OPACITY)

    ax.set_xlabel("Δκ (MHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print("Saved theory PTEP plot to", output_path)


###############################################################################
# Main sensitivity routine with a single combined plot (three subplots)
def sensitivity_main():
    """
    Modified sensitivity analysis for PTEP:
      - Uses a single uncertainty value of 1 MHz (0.001 GHz).
      - Produces a single plot with three vertically stacked subplots:
          1. J uncertainty (only J is varied),
          2. Omega uncertainty (combined ω₍c₎ and ω₍y₎, in different colors),
          3. Kappa uncertainty (combined κ₍c₎ and κ₍y₎, in different colors).
      All subplots share the same X axis (K in MHz) and have no titles.
    """
    # Define base theory parameters (all in GHz)
    base_params = {
        "J_val": 0.1,           # Nominal coupling J [Ghz]
        "J_val_unc": 0.005,     # (Will be overridden)
        "omega_c": 6.0,         # Cavity frequency [Ghz]
        "omega_c_unc": 0.01,    # (Will be overridden)
        "kappa_c": 0.01,        # Cavity linewidth [Ghz]
        "kappa_c_unc": 0.0001,  # (Will be overridden)
        # For PTEP, kappa_y is used directly for all sensitivities
        "kappa_y": 0.01,        # YIG linewidth [Ghz]
        "kappa_y_unc": 0.0001,  # (Will be overridden)
        "omega_y": 6.0,         # YIG frequency [Ghz]
        "omega_y_unc": 0.01     # (Will be overridden)
    }
    # Create frequency axis for simulation (in Hz)
    pt_freqs = np.linspace(5.8e9, 6.15e9, 1000)
    # Define array of k values (in MHz) for all sensitivities
    k_array = np.linspace(0.01, 0.25, 100)
    # Use a single uncertainty value of 1 MHz (0.001 GHz)
    unc_value = 1 / 1000  # 0.001 GHz
    # Create output folder for sensitivity plots.
    output_folder = "fig4_PTEP"
    os.makedirs(output_folder, exist_ok=True)

    # --- 1. Sensitivity for J ---
    mod_params_J = copy.deepcopy(base_params)
    # Zero out all uncertainties.
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "omega_y_unc", "kappa_y_unc"]:
        mod_params_J[key] = 0.0
    mod_params_J["J_val_unc"] = unc_value

    (det_arr_J, avg_lower_J, avg_upper_J,
     lower_min_J, lower_max_J, upper_min_J, upper_max_J) = simulate_theory_ep(
        k_array, pt_freqs, mod_params_J,
        num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL
    )

    # --- 2. Sensitivity for Omega (combined ω₍c₎ and ω₍y₎) ---
    mod_params_omega_c = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "omega_y_unc", "kappa_y_unc"]:
        mod_params_omega_c[key] = 0.0
    mod_params_omega_c["omega_c_unc"] = unc_value
    results_omega_c = simulate_theory_ep(k_array, pt_freqs, mod_params_omega_c,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    mod_params_omega_y = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "omega_y_unc", "kappa_y_unc"]:
        mod_params_omega_y[key] = 0.0
    mod_params_omega_y["omega_y_unc"] = unc_value
    results_omega_y = simulate_theory_ep(k_array, pt_freqs, mod_params_omega_y,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    (det_arr_omega, avg_lower_c, avg_upper_c, lower_min_c, lower_max_c, upper_min_c, upper_max_c) = results_omega_c
    (det_arr_omega, avg_lower_y, avg_upper_y, lower_min_y, lower_max_y, upper_min_y, upper_max_y) = results_omega_y

    # --- 3. Sensitivity for Kappa (combined κ₍c₎ and κ₍y₎) ---
    mod_params_kappa_c = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "omega_y_unc", "kappa_y_unc"]:
        mod_params_kappa_c[key] = 0.0
    mod_params_kappa_c["kappa_c_unc"] = unc_value
    results_kappa_c = simulate_theory_ep(k_array, pt_freqs, mod_params_kappa_c,
                                         num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL)

    mod_params_kappa_y = copy.deepcopy(base_params)
    for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "omega_y_unc", "kappa_y_unc"]:
        mod_params_kappa_y[key] = 0.0
    mod_params_kappa_y["kappa_y_unc"] = unc_value
    results_kappa_y = simulate_theory_ep(k_array, pt_freqs, mod_params_kappa_y,
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
    # (Optional: Plot average simulation peaks if desired)
    # axs[0].plot(det_arr_J, avg_lower_J, color="blue", lw=2, linestyle="--")
    # axs[0].plot(det_arr_J, avg_upper_J, color="blue", lw=2, linestyle="-.")
    axs[0].set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[0].tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axs[0].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    axs[0].grid(False)

    # Subplot 2: Omega uncertainty
    axs[1].fill_between(det_arr_omega, lower_min_c, lower_max_c,
                        color="blue", alpha=OPACITY, label="$f_c$ uncertainty")
    axs[1].fill_between(det_arr_omega, upper_min_c, upper_max_c,
                        color="blue", alpha=OPACITY)
    # axs[1].plot(det_arr_omega, avg_lower_c, color="blue", lw=2, linestyle="--")
    # axs[1].plot(det_arr_omega, avg_upper_c, color="blue", lw=2, linestyle="-.")
    axs[1].fill_between(det_arr_omega, lower_min_y, lower_max_y,
                        color="red", alpha=OPACITY, label="$f_y$ uncertainty")
    axs[1].fill_between(det_arr_omega, upper_min_y, upper_max_y,
                        color="red", alpha=OPACITY)
    # axs[1].plot(det_arr_omega, avg_lower_y, color="red", lw=2, linestyle="--")
    # axs[1].plot(det_arr_omega, avg_upper_y, color="red", lw=2, linestyle="-.")
    axs[1].set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[1].tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axs[1].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    axs[1].grid(False)

    # Subplot 3: Kappa uncertainty
    # Changed from green/magenta to blue (for κ₍c₎) and red (for κ₍y₎)
    axs[2].fill_between(det_arr_kappa, lower_min_kc, lower_max_kc,
                        color="blue", alpha=OPACITY, label="$κ_c$ uncertainty")
    axs[2].fill_between(det_arr_kappa, upper_min_kc, upper_max_kc,
                        color="blue", alpha=OPACITY)
    # axs[2].plot(det_arr_kappa, avg_lower_kc, color="blue", lw=2, linestyle="--")
    # axs[2].plot(det_arr_kappa, avg_upper_kc, color="blue", lw=2, linestyle="-.")
    axs[2].fill_between(det_arr_kappa, lower_min_ky, lower_max_ky,
                        color="red", alpha=OPACITY, label="$κ_y$ uncertainty")
    axs[2].fill_between(det_arr_kappa, upper_min_ky, upper_max_ky,
                        color="red", alpha=OPACITY)
    # axs[2].plot(det_arr_kappa, avg_lower_ky, color="red", lw=2, linestyle="--")
    # axs[2].plot(det_arr_kappa, avg_upper_ky, color="red", lw=2, linestyle="-.")
    axs[2].set_ylabel("Frequency [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[2].set_xlabel("Δκ [Ghz]", fontsize=LABEL_FONT_SIZE)
    axs[2].tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axs[2].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    axs[2].grid(False)

    plt.tight_layout()
    combined_output_path = os.path.join(output_folder, "sensitivity_combined.png")
    plt.savefig(combined_output_path, dpi=400)
    plt.close(fig)
    print("Saved combined sensitivity plot to", combined_output_path)


if __name__ == "__main__":
    sensitivity_main()
