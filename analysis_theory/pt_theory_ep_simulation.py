#!/usr/bin/env python3
"""
Pure Theory Simulation & Sensitivity Analysis for PTEP
-------------------------------------------------------
This module simulates the PTEP response using a Monte Carlo approach and produces
a baseline theory plot (with shaded uncertainty regions) based on constant (average) parameters.

It also includes a sensitivity analysis routine that varies the uncertainty in one parameter
at a time (using raw uncertainty values in GHz) while setting all other uncertainties to zero.
For each parameter under test, a single plot is generated that overlays both the lower and upper
branch uncertainty shadings and the average simulation curves.

Differences for the PTEP:
    - The phase is fixed at 0.
    - Each simulated trace is computed as a function of frequency.
    - However, the overall theory simulation is parameterized by the nominal kappa detuning,
      K = κ₍c₎ – κ₍y₎ (i.e. for each nominal K, we set κ₍y₎ = κ₍c₎ – K). This nominal K
      is then used as the x-axis in the sensitivity (color) plot, while the trace peak positions
      (extracted in GHz) are plotted on the y-axis.
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import concurrent.futures
import matplotlib.cm as cm

import ptep_simulation as ptep
# This module must provide:
#   - simulate_trace(J, omega_c, w_y, kappa_c, kappa_y, freqs)
#   - fast_simulate_trace(fast_func, J, omega_c, w_y, kappa_c, kappa_y, freqs)
#   - setup_fast_simulation(...)

# Global simulation switches and parameters
USE_PARALLEL = False
NUM_SIMULATION_SHOTS = 100


def run_single_theory_shot_fast(
        fast_func,
        J_val, J_val_unc,
        omega_c, omega_c_unc,
        w_y, w_y_unc,
        kappa_c, kappa_c_unc,
        kappa_y, kappa_y_unc,
        freqs
):
    """
    Perform one Monte Carlo shot:
      1. Draw random parameters from normal distributions.
      2. Simulate the PTEP trace (as a function of frequency) using the fast simulation function.
      3. Find peaks in the simulated trace and return them as a two‐element list [lower, upper] (in GHz).
    """
    J_val_sim = np.random.normal(J_val, J_val_unc)
    omega_c_sim = np.random.normal(omega_c, omega_c_unc)
    w_y_sim = np.random.normal(w_y, w_y_unc)
    kappa_c_sim = np.random.normal(kappa_c, kappa_c_unc)
    kappa_y_sim = np.random.normal(kappa_y, kappa_y_unc)

    sim_trace = ptep.fast_simulate_trace(
        fast_func,
        J_val_sim, omega_c_sim, w_y_sim, kappa_c_sim, kappa_y_sim,
        freqs  # frequency sweep array in Hz
    )

    sim_peaks_idx, _ = find_peaks(sim_trace, prominence=0.00001)
    peak_freqs = freqs[sim_peaks_idx] / 1e9  # convert Hz to GHz
    peak_freqs.sort()

    if len(peak_freqs) == 2:
        return list(peak_freqs)  # [lower, upper]
    elif len(peak_freqs) == 1:
        return [peak_freqs[0], peak_freqs[0]]
    else:
        print("WARNING: No peaks found in this simulation shot.")
        return [np.nan, np.nan]


def simulate_theory_ep(kappa_detuning_array, freqs, theory_params, num_shots=NUM_SIMULATION_SHOTS,
                       use_parallel=USE_PARALLEL):
    """
    For each nominal kappa detuning value (K = κ₍c₎ – κ₍y₎) in kappa_detuning_array, run a theory simulation.
    For each K:
      - Set κ₍y₎ = κ₍c₎ – K (with κ₍c₎ fixed from theory_params).
      - Use the average parameters to simulate the trace (as a function of frequency) and extract the peak frequencies.
      - Run num_shots Monte Carlo simulations (using run_single_theory_shot_fast)
        to obtain uncertainty bounds.

    Returns:
      kappa_detuning_array : 1D array of nominal kappa detuning values (GHz)
      avg_lower_peaks, avg_upper_peaks : average simulation peak frequencies (GHz)
      lower_min, lower_max, upper_min, upper_max : MC min/max arrays (GHz)
    """
    avg_lower_peaks = []
    avg_upper_peaks = []
    lower_min_arr = []
    lower_max_arr = []
    upper_min_arr = []
    upper_max_arr = []

    fast_func = ptep.setup_fast_simulation(drive=(1, 0), readout=(0, 1))

    for K in kappa_detuning_array:
        # For each nominal kappa detuning K, set κ₍y₎ = κ₍c₎ – K.
        kappa_c = theory_params["kappa_c"]
        kappa_y = kappa_c - K

        omega_c = theory_params["omega_c"]
        w_y = theory_params["w_y"]

        # Compute the average simulation trace (using nominal parameter values)
        sim_trace_avg = ptep.simulate_trace(
            theory_params["J_val"], omega_c, w_y, kappa_c, kappa_y,
            freqs  # frequency sweep array in Hz
        )
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)
        avg_peaks = (freqs[sim_peaks_idx_avg] / 1e9)  # in GHz
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
                            w_y, theory_params["w_y_unc"],
                            kappa_c, theory_params["kappa_c_unc"],
                            kappa_y, theory_params["kappa_y_unc"],
                            freqs
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
                    w_y, theory_params["w_y_unc"],
                    kappa_c, theory_params["kappa_c_unc"],
                    kappa_y, theory_params["kappa_y_unc"],
                    freqs
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

    return (np.array(kappa_detuning_array),
            np.array(avg_lower_peaks), np.array(avg_upper_peaks),
            np.array(lower_min_arr), np.array(lower_max_arr),
            np.array(upper_min_arr), np.array(upper_max_arr))


def plot_theory_ep_results(kappa_detuning_array, avg_lower, avg_upper,
                           lower_min, lower_max, upper_min, upper_max,
                           output_path):
    """
    Create a plot showing the theoretical PTEP results:
      - Shaded regions (fill_between) for the lower and upper branch uncertainties.
      - Solid lines for the average simulation peaks.

    The x-axis is the nominal kappa detuning (GHz), and the y-axis is the peak frequency (GHz).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(kappa_detuning_array, lower_min, lower_max,
                    color="blue", alpha=0.5, label="Lower branch bounds")
    ax.fill_between(kappa_detuning_array, upper_min, upper_max,
                    color="red", alpha=0.5, label="Upper branch bounds")

    ax.plot(kappa_detuning_array, avg_lower, color="blue", lw=2, linestyle="--", label="Average lower peak")
    ax.plot(kappa_detuning_array, avg_upper, color="red", lw=2, linestyle="-.", label="Average upper peak")

    ax.set_xlabel("Kappa Detuning (GHz)", fontsize=14)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=14)
    ax.set_title("Theoretical PTEP with Uncertainties", fontsize=14)
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Saved theory PTEP plot to", output_path)


def sensitivity_analysis_for_param(param_key, unc_values, kappa_detuning_array, freqs, base_params, output_folder):
    """
    Perform a sensitivity analysis for one parameter.

    For the parameter specified by param_key (e.g. "J_val", "omega_c", "kappa_c", "kappa_y",
    or "w_y_unc"), vary its uncertainty using a list of raw uncertainty values (in GHz).
    For each raw uncertainty value, all other uncertainties are set to 0. Then, simulate the theory PTEP
    response and store the MC uncertainty bounds.

    This function then creates a single plot overlaying the fill_between shading and average simulation curves
    for both the lower and upper branches.
    """
    if param_key == "w_y_unc":
        test_unc_key = "w_y_unc"
    else:
        test_unc_key = param_key + "_unc"

    results = {}
    for unc_val in unc_values:
        mod_params = copy.deepcopy(base_params)
        for key in ["J_val_unc", "omega_c_unc", "kappa_c_unc", "kappa_y_unc", "w_y_unc"]:
            mod_params[key] = 0.0
        mod_params[test_unc_key] = unc_val

        (kappa_det_arr, avg_lower, avg_upper,
         lower_min, lower_max, upper_min, upper_max) = simulate_theory_ep(
            kappa_detuning_array, freqs, mod_params,
            num_shots=NUM_SIMULATION_SHOTS, use_parallel=USE_PARALLEL
        )
        results[unc_val] = (kappa_det_arr, avg_lower, avg_upper, lower_min, lower_max, upper_min, upper_max)

    color_values = cm.jet(np.linspace(0, 1, len(unc_values)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, unc_val in enumerate(sorted(results.keys())):
        det_arr, avg_lower, avg_upper, lower_min, lower_max, upper_min, upper_max = results[unc_val]
        color = color_values[i]
        ax.fill_between(det_arr, lower_min, lower_max, color=color, alpha=0.5)
        ax.fill_between(det_arr, upper_min, upper_max, color=color, alpha=0.5)
        ax.plot(det_arr, avg_lower, color=color, lw=2, linestyle="--")
        ax.plot(det_arr, avg_upper, color=color, lw=2, linestyle="-.")
        ax.plot([], [], color=color, lw=2, label=f"uncertainty = {1000 * unc_val:.1f} MHz")

    ax.set_title(f"Sensitivity of {test_unc_key} on PTEP Simulation", fontsize=16)
    ax.set_xlabel("Kappa Detuning (GHz)", fontsize=14)
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
    base_params = {
        "J_val": 0.05,         # Nominal coupling J (GHz)
        "J_val_unc": 0.005,
        "omega_c": 6.0,        # Cavity frequency (GHz)
        "omega_c_unc": 0.01,
        "kappa_c": 0.01,       # Nominal cavity linewidth (GHz)
        "kappa_c_unc": 0.0001,
        "kappa_y": 0.008,      # Nominal YIG linewidth (GHz); nominal K = κ₍c₎ – κ₍y₎.
        "kappa_y_unc": 0.0001,
        "w_y": 6.0,            # YIG frequency (GHz)
        "w_y_unc": 0.01
    }
    # Frequency sweep array for simulation (in Hz)
    freqs = np.linspace(5.8e9, 6.15e9, 1000)
    # Nominal kappa detuning values (in GHz) for the theory simulation.
    kappa_detuning_array = np.linspace(0.001, 0.005, 100)
    # Raw uncertainty values to test (in GHz)
    unc_values = [0.1/1000, 0.25/1000, 0.5/1000, 1/1000]
    params_to_test = ["J_val", "omega_c", "kappa_c", "kappa_y", "w_y_unc"]

    output_folder = "sensitivity_plots_ptep"
    os.makedirs(output_folder, exist_ok=True)

    for param in params_to_test:
        sensitivity_analysis_for_param(param, unc_values, kappa_detuning_array, freqs, base_params, output_folder)


if __name__ == "__main__":
    sensitivity_main()
