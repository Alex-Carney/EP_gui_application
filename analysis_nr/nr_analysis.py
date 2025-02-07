import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import concurrent.futures

import nr_simulation as nr_sim
from nr_amperage_data_loader import NRAmperageDataLoader
from nr_fitting import (process_all_traces, compute_delta, fit_NR_trace, iterative_NR_fit,
                        fit_trace, theory_supported_NR_fit)
from nr_plotting import (plot_raw_colorplot, plot_individual_trace, debug_plot_individual_NR_traces,
                         plot_delta_colorplot, plot_nr_peaks_only_vs_detuning)

# ------------------ GLOBAL SWITCH TO ENABLE/DISABLE PARALLELISM ------------------
USE_PARALLEL = False

NUM_SIMULATION_SHOTS = 5

# Configuration flags
CAV_YIG_DEBUG = False
NR_DEBUG = True
PLOTS_FOLDER = "plots"


def load_config(config_path, config_name="default"):
    """
    Load the configuration from a JSON file and select the configuration
    with the given name.
    """
    with open(config_path, "r") as file:
        all_configs = json.load(file)
    if config_name not in all_configs:
        raise KeyError(f"Configuration '{config_name}' not found in {config_path}.")
    return all_configs[config_name]


def run_single_theory_shot(
        J_val, J_val_unc,
        omega_c, omega_c_unc,
        omega_y, omega_y_unc,
        kappa_c, kappa_c_unc,
        kappa_y, kappa_y_unc,
        nr_freqs
):
    """
    Perform one Monte Carlo shot:
    1. Draw random parameters
    2. Simulate the NR trace
    3. Find peaks
    4. Sort them and return as a length-2 list [lower, upper].
    """
    J_val_sim = np.random.normal(J_val, J_val_unc)
    omega_c_sim = np.random.normal(omega_c, omega_c_unc)
    omega_y_sim = np.random.normal(omega_y, omega_y_unc)
    kappa_c_sim = np.random.normal(kappa_c, kappa_c_unc)
    kappa_y_sim = np.random.normal(kappa_y, kappa_y_unc)

    sim_trace = nr_sim.simulate_trace(
        J_val_sim, omega_c_sim, omega_y_sim, kappa_c_sim, kappa_y_sim,
        nr_freqs / 1e9
    )

    sim_peaks_idx, _ = find_peaks(sim_trace, prominence=0.00001)
    peak_freqs = nr_freqs[sim_peaks_idx] / 1e9
    peak_freqs.sort()

    if len(peak_freqs) == 2:
        return list(peak_freqs)   # [lower, upper]
    elif len(peak_freqs) == 1:
        # Return a double of the single peak
        return [peak_freqs[0], peak_freqs[0]]
    else:
        # No peaks found
        print("WARNING: No peaks found in the simulation.")
        return [np.nan, np.nan]


def main():
    # ------------------ CHANGE CONFIGURATION HERE ------------------
    config_path = "nr_expr_config.json"
    config_name = "hailey_sunday"  # Hardcoded configuration name

    # ------------------ EXTRACT DATA FROM CONFIG ------------------
    config = load_config(config_path, config_name)
    experiment_id = config["experiment_id"]
    colorplot_freq_min = config["frequency_limits"]["colorplot"]["min"]
    colorplot_freq_max = config["frequency_limits"]["colorplot"]["max"]
    cavity_freq_min = config["frequency_limits"]["cavity"]["min"]
    cavity_freq_max = config["frequency_limits"]["cavity"]["max"]
    yig_freq_min = config["frequency_limits"]["yig"]["min"]
    yig_freq_max = config["frequency_limits"]["yig"]["max"]
    current_min = config["current_limits"]["min"]
    current_max = config["current_limits"]["max"]
    db_path = config["db_path"]

    # ------------------ DATABASE LOADING ------------------
    cavity_loader = NRAmperageDataLoader(db_path, experiment_id, "cavity",
                                         cavity_freq_min, cavity_freq_max,
                                         current_min, current_max)
    yig_loader = NRAmperageDataLoader(db_path, experiment_id, "yig",
                                      yig_freq_min, yig_freq_max,
                                      current_min, current_max)
    nr_loader = NRAmperageDataLoader(db_path, experiment_id, "nr",
                                     colorplot_freq_min, colorplot_freq_max,
                                     current_min, current_max)
    cavity_power, cavity_currents, cavity_freqs, cavity_settings = cavity_loader.load_data()
    yig_power, yig_currents, yig_freqs, yig_settings = yig_loader.load_data()
    nr_power, nr_currents, nr_freqs, nr_settings = nr_loader.load_data()

    if cavity_power is None or yig_power is None or nr_power is None:
        print("Error: One or more datasets are missing. Exiting.")
        return

    # ------------------ RAW COLORPLOT GENERATION ------------------
    raw_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP_raw")
    plot_raw_colorplot(nr_power, nr_currents, nr_freqs, experiment_id, nr_settings,
                       raw_folder, readout_type="nr")

    # ------------------ CAVITY/YIG TRACES PLOTTING (OPTIONAL) ------------------
    if CAV_YIG_DEBUG:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        for rt, power, currents, freqs in [("cavity", cavity_power, cavity_currents, cavity_freqs),
                                           ("yig", yig_power, yig_currents, yig_freqs)]:
            for i, cur in enumerate(currents):
                trace = power[i, :]
                fit_data = fit_trace(cur, freqs, trace)
                plot_individual_trace(cur, freqs, trace, rt, debug_folder, fit_data)

    # ------------------ CAVITY/YIG TRACES ANALYSIS ------------------
    cavity_results = process_all_traces(cavity_power, cavity_currents, cavity_freqs)
    yig_results = process_all_traces(yig_power, yig_currents, yig_freqs)
    cavity_df = pd.DataFrame(cavity_results)
    yig_df = pd.DataFrame(yig_results)
    delta_df = compute_delta(cavity_df, yig_df)
    print("Computed detuning for each current.")

    # Build maps (cavity, YIG, delta) so that we can simulate NR traces
    cavity_map = cavity_df.set_index("current")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    yig_map = yig_df.set_index("current")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    delta_map = delta_df.set_index("current")["Delta"].to_dict()

    # nr_fit_dict will hold both the experimental fit data and the theory bounds
    nr_fit_dict = {}  # Key: current, Value: dict

    # ------------------ NR THEORY SIMULATIONS AND FITTING ------------------
    for i, cur in enumerate(nr_currents):
        if cur not in delta_map:
            continue

        trace = nr_power[i, :]
        # Only simulate if we have cavity and YIG data for this current
        if cur not in cavity_map or cur not in yig_map:
            continue

        omega_c, kappa_c, omega_c_unc, kappa_c_unc = (cavity_map[cur][key] for key in
                                                      ["omega", "kappa", "omega_unc", "kappa_unc"])
        omega_y, kappa_y, omega_y_unc, kappa_y_unc = (yig_map[cur][key] for key in
                                                      ["omega", "kappa", "omega_unc", "kappa_unc"])

        # Get simulation parameter J from config
        J_val = config["optimal_J"]
        J_val_unc = config["optimal_J_unc"]

        # Perform multiple simulation shots (Monte Carlo)
        all_peak_freqs = []
        start_time = time.time()
        print(f"Starting {NUM_SIMULATION_SHOTS} shots "
              f"(parallel={USE_PARALLEL}) for current={cur:.6f} A")

        if USE_PARALLEL:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for shot in range(NUM_SIMULATION_SHOTS):
                    futures.append(
                        executor.submit(
                            run_single_theory_shot,
                            J_val, J_val_unc,
                            omega_c, omega_c_unc,
                            omega_y, omega_y_unc,
                            kappa_c, kappa_c_unc,
                            kappa_y, kappa_y_unc,
                            nr_freqs
                        )
                    )
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_peak_freqs.append(result)
        else:
            # Serial fallback
            for shot in range(NUM_SIMULATION_SHOTS):
                result = run_single_theory_shot(
                    J_val, J_val_unc,
                    omega_c, omega_c_unc,
                    omega_y, omega_y_unc,
                    kappa_c, kappa_c_unc,
                    kappa_y, kappa_y_unc,
                    nr_freqs
                )
                all_peak_freqs.append(result)

        elapsed = time.time() - start_time
        print(f"Simulated {NUM_SIMULATION_SHOTS} shots in {elapsed:.2f} seconds.")
        all_peak_freqs = np.array(all_peak_freqs)  # shape: (NUM_SIMULATION_SHOTS, 2)

        # Determine the min/max for the lower and upper peak across all shots
        # These define the theory's predicted frequency range
        lowest_peak_range = (np.nanmin(all_peak_freqs[:, 0]), np.nanmax(all_peak_freqs[:, 0]))
        highest_peak_range = (np.nanmin(all_peak_freqs[:, 1]), np.nanmax(all_peak_freqs[:, 1]))

        # Simulate the "average" trace for an initial guess
        sim_trace_avg = nr_sim.simulate_trace(J_val, omega_c, omega_y, kappa_c, kappa_y, nr_freqs / 1e9)
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)

        # Use the theory-supported NR fit to find the experimental peaks
        fit_data = theory_supported_NR_fit(cur, nr_freqs, trace, sim_trace_avg, sim_peaks_idx_avg)

        # Store everything in nr_fit_dict
        nr_fit_dict[cur] = {
            "trace": trace,
            "sim_trace": sim_trace_avg,
            "sim_peaks_idx": sim_peaks_idx_avg,
            "fit_data": fit_data,
            # The theory bounds from Monte Carlo
            "theory_lower_min": lowest_peak_range[0],
            "theory_lower_max": lowest_peak_range[1],
            "theory_upper_min": highest_peak_range[0],
            "theory_upper_max": highest_peak_range[1],
        }

        # ------------------ PLOT INDIVIDUAL TRACE (DEBUG) ------------------
        if NR_DEBUG:
            debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
            print(f"Plotting NR trace for current {cur:.6f} A")
            order_prefix = f"{i+1:03d}_"
            plot_individual_trace(
                cur,
                nr_freqs,
                trace,
                "nr",
                debug_folder,
                fit_data,
                detuning_val=delta_map[cur],
                order_prefix=order_prefix,
                simulated_trace=sim_trace_avg,
                simulated_trace_peak_idxs=sim_peaks_idx_avg,
                simulated_vertical_offset=config["simulated_vertical_offset"],
                peak1_lower_bound=lowest_peak_range[0],
                peak1_upper_bound=lowest_peak_range[1],
                peak2_lower_bound=highest_peak_range[0],
                peak2_upper_bound=highest_peak_range[1],
            )

    # ------------------ NR DETUNING COLORPLOT ------------------
    output_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP")
    plot_delta_colorplot(nr_power, nr_currents, nr_freqs, delta_df, experiment_id, nr_settings, output_folder)

    # ------------------ NR DETUNING PEAK LOCATION PLOT ------------------
    overlay_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_nr_peaks")
    os.makedirs(overlay_folder, exist_ok=True)
    print("Plotting NR peak locations vs. Detuning overlay...")

    # --- 1) Collect the *experimental* peak data for errorbar plotting ---
    detuning_list = []
    peak_list = []
    peak_unc_list = []

    # --- 2) Collect the *theory* min/max data for shading ---
    theory_detuning_list = []
    theory_lower_min_list = []
    theory_lower_max_list = []
    theory_upper_min_list = []
    theory_upper_max_list = []

    for cur in nr_fit_dict.keys():
        fit_data = nr_fit_dict[cur]["fit_data"]
        if fit_data is None:
            continue

        # This current's detuning
        delta_val = delta_map[cur]

        # Collect experimental fit data
        if fit_data.get("fit_type") == "single":
            detuning_list.append(delta_val)
            peak_list.append(fit_data.get("omega"))
            peak_unc_list.append(fit_data.get("omega_unc", 0.0))
        elif fit_data.get("fit_type") == "double":
            detuning_list.extend([delta_val, delta_val])
            peak_list.extend([fit_data.get("peak1"), fit_data.get("peak2")])
            peak_unc_list.extend([fit_data.get("peak1_unc", 0.0),
                                  fit_data.get("peak2_unc", 0.0)])

        # Collect theory bounding data
        theory_detuning_list.append(delta_val)
        theory_lower_min_list.append(nr_fit_dict[cur]["theory_lower_min"])
        theory_lower_max_list.append(nr_fit_dict[cur]["theory_lower_max"])
        theory_upper_min_list.append(nr_fit_dict[cur]["theory_upper_min"])
        theory_upper_max_list.append(nr_fit_dict[cur]["theory_upper_max"])

    # If no experimental peaks were extracted, skip plotting
    if len(peak_list) == 0:
        print("No NR peaks extracted for the overlay plot.")
        return

    # Convert to arrays
    detuning_array = np.array(detuning_list)
    peak_array = np.array(peak_list)
    peak_unc_array = np.array(peak_unc_list)

    theory_detuning_array = np.array(theory_detuning_list)
    theory_lower_min_array = np.array(theory_lower_min_list)
    theory_lower_max_array = np.array(theory_lower_max_list)
    theory_upper_min_array = np.array(theory_upper_min_list)
    theory_upper_max_array = np.array(theory_upper_max_list)

    # Sort by detuning so fill_between() works left-to-right
    sort_idx = np.argsort(theory_detuning_array)
    theory_detuning_array = theory_detuning_array[sort_idx]
    theory_lower_min_array = theory_lower_min_array[sort_idx]
    theory_lower_max_array = theory_lower_max_array[sort_idx]
    theory_upper_min_array = theory_upper_min_array[sort_idx]
    theory_upper_max_array = theory_upper_max_array[sort_idx]

    # Now create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the theoretical shading first (behind the data)
    # Lower branch
    ax.fill_between(
        theory_detuning_array,
        theory_lower_min_array,
        theory_lower_max_array,
        color="blue",
        alpha=0.2,
        label="Theory Lower Branch"
    )
    # Upper branch
    ax.fill_between(
        theory_detuning_array,
        theory_upper_min_array,
        theory_upper_max_array,
        color="green",
        alpha=0.2,
        label="Theory Upper Branch"
    )

    # Plot the experimental data on top
    ax.errorbar(detuning_array, peak_array,
                yerr=peak_unc_array,
                fmt="o", ecolor="red",
                capsize=4, label="NR Hybridized Peaks (Data)")

    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=14)
    ax.set_title("NR Peak Locations vs. Detuning", fontsize=14)

    # Tidy up plot ranges, in case some shading is out of range
    y_min = min(peak_array.min(), theory_lower_min_array.min(), theory_upper_min_array.min())
    y_max = max(peak_array.max(), theory_lower_max_array.max(), theory_upper_max_array.max())
    ax.set_ylim(y_min, y_max)

    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    overlay_plot_path = os.path.join(overlay_folder, f"nr_peaks_overlay_exp_{experiment_id}.png")
    plt.savefig(overlay_plot_path, dpi=300)
    plt.close(fig)
    print("Saved NR peaks overlay plot to", overlay_plot_path)


if __name__ == "__main__":
    main()
