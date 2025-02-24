import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pt_config_loader as cl
import pt_simulation as pt_sim
import pt_plotting as pt_plot
import pt_voltage_data_loader as pt_data
import pt_fitting as pt_fit

# ------------------ COMMONLY CHANGED INPUTS ------------------
CONFIG_NAME = "cab_nr2"
CAV_YIG_DEBUG = False
NR_DEBUG = False

# ------------------ TOP LEVEL CONFIGURATION ------------------
NUM_SIMULATION_SHOTS = 1250
PLOTS_FOLDER = "plots"
SEED = 12345
CONFIG_PATH = "pt_expr_config.json"
np.random.seed(SEED)


def main():
    # ------------------ EXTRACT DATA FROM CONFIG ------------------
    config: cl.PTExperimentConfig = cl.load_config(CONFIG_PATH, CONFIG_NAME)
    experiment_id = config.experiment_id
    db_path = config.db_path
    readout_type = config.readout_type
    colorplot_freq_min = config.colorplot_freq_min
    colorplot_freq_max = config.colorplot_freq_max
    cavity_freq_min = config.cavity_freq_min
    cavity_freq_max = config.cavity_freq_max
    yig_freq_min = config.yig_freq_min
    yig_freq_max = config.yig_freq_max
    voltage_min = config.voltage_min
    voltage_max = config.voltage_max

    # ------------------ DATABASE LOADING ------------------
    cavity_loader = pt_data.PTVoltageDataLoader(db_path, experiment_id, "cavity",
                                                cavity_freq_min, cavity_freq_max,
                                                voltage_min, voltage_max, independent_var="set_voltage")
    yig_loader = pt_data.PTVoltageDataLoader(db_path, experiment_id, "yig",
                                             yig_freq_min, yig_freq_max,
                                             voltage_min, voltage_max, independent_var="set_voltage")
    pt_loader = pt_data.PTVoltageDataLoader(db_path, experiment_id, readout_type,
                                            colorplot_freq_min, colorplot_freq_max,
                                            voltage_min, voltage_max, independent_var="set_voltage")
    cavity_power, cavity_voltages, cavity_freqs, cavity_settings = cavity_loader.load_data()
    yig_power, yig_voltages, yig_freqs, yig_settings = yig_loader.load_data()
    pt_power, nr_voltages, pt_freqs, nr_settings = pt_loader.load_data()

    if cavity_power is None or yig_power is None or pt_power is None:
        print("Error: One or more datasets are missing. Exiting.")
        return

    # ------------------ RAW COLORPLOT GENERATION ------------------
    raw_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP_raw")
    pt_plot.plot_raw_colorplot(pt_power, nr_voltages, pt_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type=readout_type)
    pt_plot.plot_raw_colorplot(yig_power, yig_voltages, yig_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type='yig')
    pt_plot.plot_raw_colorplot(cavity_power, cavity_voltages, cavity_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type='cavity')

    # ------------------ CAVITY/YIG TRACES PLOTTING (OPTIONAL) ------------------
    if CAV_YIG_DEBUG:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        for rt, power, voltages, freqs in [("cavity", cavity_power, cavity_voltages, cavity_freqs),
                                           ("yig", yig_power, yig_voltages, yig_freqs)]:
            for i, v in enumerate(voltages):
                trace = power[i, :]
                fit_data = pt_fit.fit_trace(v, freqs, trace, peak_selection_option="amplitude")
                pt_plot.plot_individual_trace(v, freqs, trace, rt, debug_folder, fit_data)

    # ------------------ CAVITY/YIG TRACES ANALYSIS - MEASURING DETUNING ------------------
    cavity_results = pt_fit.process_all_traces(cavity_power, cavity_voltages, cavity_freqs,
                                               peak_selection_option="amplitude")
    yig_results = pt_fit.process_all_traces(yig_power, yig_voltages, yig_freqs, apply_pre_smoothing=True,
                                            peak_selection_option="amplitude")
    cavity_df = pd.DataFrame(cavity_results)
    yig_df = pd.DataFrame(yig_results)
    K_df = pt_fit.compute_K(cavity_df, yig_df)

    # Build maps (cavity, YIG, delta) so that we can simulate NR traces
    cavity_map = cavity_df.set_index("voltage")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    yig_map = yig_df.set_index("voltage")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    K_map = K_df.set_index("voltage")["K"].to_dict()

    # nr_fit_dict will hold both the experimental fit data and the theory bounds
    nr_fit_dict = {}  # Key: current, Value: dict

    # ------------------ NR THEORY SIMULATIONS AND FITTING ------------------
    # (Assuming nr_currents is defined in your nr_loader output.)
    for i, v in enumerate(nr_voltages):
        if v not in K_map:
            continue

        trace = pt_power[i, :]
        # Only simulate if we have cavity and YIG data for this current
        if v not in cavity_map or v not in yig_map:
            continue

        omega_c, kappa_c, omega_c_unc, kappa_c_unc = (cavity_map[v][key] for key in
                                                      ["omega", "kappa", "omega_unc", "kappa_unc"])
        omega_y, kappa_y, omega_y_unc, kappa_y_unc = (yig_map[v][key] for key in
                                                      ["omega", "kappa", "omega_unc", "kappa_unc"])

        # Perform multiple simulation shots (Monte Carlo)
        all_peak_freqs = []
        drive = (1, 0)
        # NOTE: The readout configuration didn't actually change in the experiment...oops
        readout = (1, 0) if readout_type == "normal" else (0, 1)
        fast_func = pt_sim.setup_fast_simulation(drive=drive, readout=readout)
        for shot in range(NUM_SIMULATION_SHOTS):
            result = pt_sim.run_single_theory_shot_fast(
                fast_func,
                config.optimal_J, config.optimal_J_unc,
                omega_c, omega_c_unc,
                omega_y, omega_y_unc,
                kappa_c, kappa_c_unc,
                kappa_y, kappa_y_unc,
                pt_freqs,
            )
            all_peak_freqs.append(result)

        all_peak_freqs = np.array(all_peak_freqs)  # shape: (NUM_SIMULATION_SHOTS, 2)
        # Simulate the "average" trace for an initial guess
        sim_trace_avg = pt_sim.simulate_trace(config.optimal_J, omega_c, omega_y, kappa_c, kappa_y, pt_freqs / 1e9,
                                              readout=[1, 0] if readout_type == "normal" else [0, 1])
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)

        # Determine the min/max for the lower and upper peak across all shots
        lowest_peak_range = (np.nanmin(all_peak_freqs[:, 0]), np.nanmax(all_peak_freqs[:, 0]))
        highest_peak_range = (np.nanmin(all_peak_freqs[:, 1]), np.nanmax(all_peak_freqs[:, 1]))

        # Use the theory-supported NR fit to find the experimental peaks
        fit_data = pt_fit.theory_supported_PT_fit(v, pt_freqs, trace, sim_trace_avg, sim_peaks_idx_avg,
                                                  config.amplitude_threshold_overfitting)

        # Store everything in nr_fit_dict
        nr_fit_dict[v] = {
            "trace": trace,
            "sim_trace": sim_trace_avg,
            "sim_peaks_idx": sim_peaks_idx_avg,
            "fit_data": fit_data,
            "theory_lower_min": lowest_peak_range[0],
            "theory_lower_max": lowest_peak_range[1],
            "theory_upper_min": highest_peak_range[0],
            "theory_upper_max": highest_peak_range[1],
        }

        # ------------------ PLOT INDIVIDUAL TRACE (DEBUG) ------------------
        if NR_DEBUG:
            debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
            order_prefix = f"{i + 1:03d}_"
            pt_plot.plot_individual_trace(
                v,
                pt_freqs,
                trace,
                readout_type,
                debug_folder,
                fit_data,
                K_val=K_map[v],
                order_prefix=order_prefix,
                simulated_trace=sim_trace_avg,
                simulated_trace_peak_idxs=sim_peaks_idx_avg,
                simulated_vertical_offset=config.simulated_vertical_offset,
                peak1_lower_bound=lowest_peak_range[0],
                peak1_upper_bound=lowest_peak_range[1],
                peak2_lower_bound=highest_peak_range[0],
                peak2_upper_bound=highest_peak_range[1],
            )

    # ------------------ NR DETUNING COLORPLOT ------------------
    output_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP")
    pt_plot.plot_K_colorplot(pt_power, nr_voltages, pt_freqs, K_df, experiment_id, nr_settings, output_folder)

    # ------------------ NR DETUNING PEAK LOCATION PLOT ------------------
    overlay_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_nr_peaks")
    os.makedirs(overlay_folder, exist_ok=True)
    print("Plotting NR peak locations vs. Detuning overlay...")

    # --- 1) Collect the *experimental* peak data for errorbar plotting ---
    K_list = []
    peak_list = []
    peak_unc_list = []
    linewidth_list = []
    splitting_ratio_list = []
    K_double_list = []

    # --- 2) Collect the *theory* min/max data for shading ---
    (theory_K_list, theory_lower_min_list, theory_lower_max_list,
     theory_upper_min_list, theory_upper_max_list) = ([] for _ in range(5))

    for v in nr_fit_dict.keys():
        fit_data = nr_fit_dict[v]["fit_data"]
        if fit_data is None:
            continue
        K_val = K_map[v]
        if fit_data.get("fit_type") == "single":
            K_list.append(K_val)
            peak_list.append(fit_data.get("omega"))
            peak_unc_list.append(fit_data.get("omega_unc", 0.0))
            linewidth_list.append(fit_data.get("linewidth", 0.0))
        elif fit_data.get("fit_type") == "double":
            K_list.extend([K_val, K_val])
            peak_list.extend([fit_data.get("peak1"), fit_data.get("peak2")])
            peak_unc_list.extend([fit_data.get("peak1_unc", 0.0),
                                  fit_data.get("peak2_unc", 0.0)])
            linewidth_list.extend([fit_data.get("peak1_linewidth", 0.0),
                                   fit_data.get("peak2_linewidth", 0.0)])
            splitting = abs(fit_data.get("peak1") - fit_data.get("peak2"))
            avg_linewidth = (fit_data.get("peak1_linewidth") + fit_data.get("peak2_linewidth")) / 2
            if avg_linewidth > 0:
                ratio = splitting / avg_linewidth
            else:
                ratio = 0
            splitting_ratio_list.append(ratio)
            K_double_list.append(K_val)

        theory_K_list.append(K_val)
        theory_lower_min_list.append(nr_fit_dict[v]["theory_lower_min"])
        theory_lower_max_list.append(nr_fit_dict[v]["theory_lower_max"])
        theory_upper_min_list.append(nr_fit_dict[v]["theory_upper_min"])
        theory_upper_max_list.append(nr_fit_dict[v]["theory_upper_max"])

    if len(peak_list) == 0:
        print("No NR peaks extracted for the overlay plot.")
        return

    K_array = np.array(K_list)
    peak_array = np.array(peak_list)
    peak_unc_array = np.array(peak_unc_list)
    linewidth_array = np.array(linewidth_list)
    splitting_ratio_array = np.array(splitting_ratio_list)
    detuning_double_array = np.array(K_double_list)

    theory_K_array = np.array(theory_K_list)
    theory_lower_min_array = np.array(theory_lower_min_list)
    theory_lower_max_array = np.array(theory_lower_max_list)
    theory_upper_min_array = np.array(theory_upper_min_list)
    theory_upper_max_array = np.array(theory_upper_max_list)

    sort_idx = np.argsort(theory_K_array)
    theory_K_array = theory_K_array[sort_idx]
    theory_lower_min_array = theory_lower_min_array[sort_idx]
    theory_lower_max_array = theory_lower_max_array[sort_idx]
    theory_upper_min_array = theory_upper_min_array[sort_idx]
    theory_upper_max_array = theory_upper_max_array[sort_idx]

    # Get the kappa_C from the data
    cavity_kappa_values = cavity_df["kappa"].values
    yig_kappa_values = yig_df["kappa"].values
    average_kappa = np.mean(cavity_kappa_values)

    # Get the average Delta from the data
    cavity_freq_values = cavity_df["omega"].values
    yig_freq_values = yig_df["omega"].values

    cavity_freq = np.mean(cavity_freq_values)
    yig_freq = np.mean(yig_freq_values)

    delta_values = cavity_freq_values - yig_freq_values

    print("\n Average kappa_C:", average_kappa)
    # print the average and stderr of delta_values
    print("Average delta:", np.mean(delta_values))
    print("Standard deviation of delta:", np.std(delta_values))
    print("Cavity freq average: ", cavity_freq)
    print("Yig freq average: ", yig_freq)

    print("max yig kappa: ", np.max(yig_kappa_values))
    print("min yig kappa: ", np.min(yig_kappa_values))
    print("average yig kappa: ", np.mean(yig_kappa_values))

    pt_plot.plot_final_peak_plot(
        theory_K_array=theory_K_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        optimal_J=config.optimal_J,
        K_array=K_array,
        peak_array=peak_array,
        experiment_id=experiment_id,
        overlay_folder=overlay_folder,
        theory_upper_min_array=theory_upper_min_array,
        theory_upper_max_array=theory_upper_max_array,
        peak_unc_array=peak_unc_array,
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end,
        errorbar_color="red",
        kappa_cavity=average_kappa,
    )

    # Plot the final plot again, but use Linewidths for the errorbar values instead
    pt_plot.plot_final_peak_plot(
        theory_K_array=theory_K_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        optimal_J=config.optimal_J,
        K_array=K_array,
        peak_array=peak_array,
        experiment_id=experiment_id,
        overlay_folder=overlay_folder,
        theory_upper_min_array=theory_upper_min_array,
        theory_upper_max_array=theory_upper_max_array,
        peak_unc_array=linewidth_array,  # Use linewidths for the errorbar values
        errorbar_color="cyan",
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end,
        filename_prepend="fwhm_"
    )

    # ------------------ LINEWIDTH PLOTS ------------------
    pt_plot.plot_linewidth_vs_K(K_array, linewidth_array, config.optimal_J, linewidth_unc_array=None,
                                experiment_id=experiment_id, output_folder=overlay_folder, errorbar_color="blue",
                                overlap_region_start=config.overlap_region_start,
                                overlap_region_end=config.overlap_region_end)

    pt_plot.plot_sum_linewidth_vs_K(K_array, linewidth_array, config.optimal_J, linewidth_unc_array=None,
                                    experiment_id=experiment_id, output_folder=overlay_folder,
                                    errorbar_color="blue",
                                    overlap_region_start=config.overlap_region_start,
                                    overlap_region_end=config.overlap_region_end)

    # Now call the new plotting function:
    pt_plot.plot_splitting_ratio_vs_K(
        detuning_double_array,
        splitting_ratio_array,
        optimal_J=config.optimal_J,
        experiment_id=experiment_id,
        output_folder=overlay_folder,
        errorbar_color="magenta",
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end
    )

    # ------------------ FIGURE 3 PLOT (DETUNING COLORPLOT WITH SENSITIVITY) ------------------
    fig3_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_FIG3")

    # Take all of the peaks found with detuning less than the EP threshold that are single peaks,
    # and average their frequency.
    ep_threshold = 2 * config.optimal_J  # EP threshold defined as 2 * optimal_J
    single_peak_frequencies = []

    # Loop through each current in nr_fit_dict
    for v, data in nr_fit_dict.items():
        fit_data = data.get("fit_data")
        if fit_data is None:
            continue
        # Only consider single Lorentzian fits
        if fit_data.get("fit_type") == "single":
            K_val = K_map.get(v)
            if K_val is None:
                continue
            # Check if the detuning is below the EP threshold
            if K_val < ep_threshold:
                # Extract the single peak frequency (named "omega")
                omega_val = fit_data.get("omega")
                if omega_val is not None:
                    single_peak_frequencies.append(omega_val)

    if single_peak_frequencies:
        avg_single_peak_frequency = np.mean(single_peak_frequencies)
        print("Average frequency for single peaks with detuning less than EP threshold:",
              avg_single_peak_frequency)

        # Plot the figure 3 plot
        pt_plot.plot_fig3_PT(pt_power, nr_voltages, pt_freqs, K_df,
                             experiment_id, nr_settings, avg_single_peak_frequency, fig3_folder,
                             optimal_J=config.optimal_J, frequency_radius=.0006)

    else:
        print("WARNING: No single peaks with detuning less than EP threshold were found.")


if __name__ == "__main__":
    main()
