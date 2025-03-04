import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import config_loader as cl
import nr_simulation as nr_sim
import nr_plotting as nr_plot
import nr_amperage_data_loader as nr_data
import nr_fitting as nr_fit

# ------------------ COMMONLY CHANGED INPUTS ------------------
CONFIG_NAME = "hailey_friday_normal"
CAV_YIG_DEBUG = False
NR_DEBUG = False

# ------------------ TOP LEVEL CONFIGURATION ------------------
NUM_SIMULATION_SHOTS = 1250
PLOTS_FOLDER = "plots"
SEED = 12345
CONFIG_PATH = "nr_expr_config.json"
np.random.seed(SEED)


def main():
    # ------------------ EXTRACT DATA FROM CONFIG ------------------
    config: cl.ExperimentConfig = cl.load_config(CONFIG_PATH, CONFIG_NAME)
    experiment_id = config.experiment_id
    db_path = config.db_path
    readout_type = config.readout_type
    colorplot_freq_min = config.colorplot_freq_min
    colorplot_freq_max = config.colorplot_freq_max
    cavity_freq_min = config.cavity_freq_min
    cavity_freq_max = config.cavity_freq_max
    yig_freq_min = config.yig_freq_min
    yig_freq_max = config.yig_freq_max
    current_min = config.current_min
    current_max = config.current_max

    # ------------------ DATABASE LOADING ------------------
    cavity_loader = nr_data.NRAmperageDataLoader(db_path, experiment_id, "cavity",
                                                 cavity_freq_min, cavity_freq_max,
                                                 current_min, current_max)
    yig_loader = nr_data.NRAmperageDataLoader(db_path, experiment_id, "yig",
                                              yig_freq_min, yig_freq_max,
                                              current_min, current_max)
    nr_loader = nr_data.NRAmperageDataLoader(db_path, experiment_id, readout_type,
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
    nr_plot.plot_raw_colorplot(nr_power, nr_currents, nr_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type=readout_type)
    nr_plot.plot_raw_colorplot(yig_power, yig_currents, yig_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type='yig')
    nr_plot.plot_raw_colorplot(cavity_power, cavity_currents, cavity_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type='cavity')

    # ------------------ CAVITY/YIG TRACES PLOTTING (OPTIONAL) ------------------
    if CAV_YIG_DEBUG:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        for rt, power, currents, freqs in [("cavity", cavity_power, cavity_currents, cavity_freqs),
                                           ("yig", yig_power, yig_currents, yig_freqs)]:
            for i, cur in enumerate(currents):
                trace = power[i, :]
                fit_data = nr_fit.fit_trace(cur, freqs, trace)
                nr_plot.plot_individual_trace(cur, freqs, trace, rt, debug_folder, fit_data)

    # ------------------ CAVITY/YIG TRACES ANALYSIS - MEASURING DETUNING ------------------
    cavity_results = nr_fit.process_all_traces(cavity_power, cavity_currents, cavity_freqs)
    yig_results = nr_fit.process_all_traces(yig_power, yig_currents, yig_freqs)
    cavity_df = pd.DataFrame(cavity_results)
    yig_df = pd.DataFrame(yig_results)
    delta_df = nr_fit.compute_delta(cavity_df, yig_df)

    # Build maps (cavity, YIG, delta) so that we can simulate NR traces
    cavity_map = cavity_df.set_index("current")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    yig_map = yig_df.set_index("current")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    delta_map = delta_df.set_index("current")["Delta"].to_dict()

    # nr_fit_dict will hold both the experimental fit data and the theory bounds
    nr_fit_dict = {}  # Key: current, Value: dict

    # ------------------ NR THEORY SIMULATIONS AND FITTING ------------------
    # (Assuming nr_currents is defined in your nr_loader output.)
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

        # Perform multiple simulation shots (Monte Carlo)
        all_peak_freqs = []
        start_time = time.time()
        drive = (1, 0)
        # NOTE: The readout configuration didn't actually change in the experiment...oops
        readout = (0, 1) if readout_type == "nr" else (0, 1)
        fast_func = nr_sim.setup_fast_simulation(drive=drive, readout=readout)
        for shot in range(NUM_SIMULATION_SHOTS):
            result = nr_sim.run_single_theory_shot_fast(
                fast_func,
                config.optimal_J, config.optimal_J_unc,
                omega_c, omega_c_unc,
                omega_y, omega_y_unc,
                kappa_c, kappa_c_unc,
                kappa_y, kappa_y_unc,
                nr_freqs,
            )
            all_peak_freqs.append(result)

        all_peak_freqs = np.array(all_peak_freqs)  # shape: (NUM_SIMULATION_SHOTS, 2)
        # Simulate the "average" trace for an initial guess
        sim_trace_avg = nr_sim.simulate_trace(config.optimal_J, omega_c, omega_y, kappa_c, kappa_y, nr_freqs / 1e9,
                                              readout=[0, 1] if readout_type == "nr" else [0, 1])
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)

        # Determine the min/max for the lower and upper peak across all shots
        lowest_peak_range = (np.nanmin(all_peak_freqs[:, 0]), np.nanmax(all_peak_freqs[:, 0]))
        highest_peak_range = (np.nanmin(all_peak_freqs[:, 1]), np.nanmax(all_peak_freqs[:, 1]))

        # Use the theory-supported NR fit to find the experimental peaks
        fit_data = nr_fit.theory_supported_NR_fit(cur, nr_freqs, trace, sim_trace_avg, sim_peaks_idx_avg,
                                                  config.amplitude_threshold_overfitting)

        # Store everything in nr_fit_dict
        nr_fit_dict[cur] = {
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
            nr_plot.plot_individual_trace(
                cur,
                nr_freqs,
                trace,
                readout_type,
                debug_folder,
                fit_data,
                detuning_val=delta_map[cur],
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
    nr_plot.plot_delta_colorplot(nr_power, nr_currents, nr_freqs, delta_df, experiment_id, nr_settings, output_folder)
    nr_plot.plot_delta_colorplot(yig_power, nr_currents, yig_freqs, delta_df, experiment_id, nr_settings,
                                 output_folder, filename_prepend='YIG_')

    # ------------------ NR DETUNING PEAK LOCATION PLOT ------------------
    overlay_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_nr_peaks")
    os.makedirs(overlay_folder, exist_ok=True)
    print("Plotting NR peak locations vs. Detuning overlay...")

    # --- 1) Collect the *experimental* peak data for errorbar plotting ---
    detuning_list = []
    peak_list = []
    peak_unc_list = []
    linewidth_list = []
    splitting_ratio_list = []
    detuning_double_list = []

    # --- 2) Collect the *theory* min/max data for shading ---
    (theory_detuning_list, theory_lower_min_list, theory_lower_max_list,
     theory_upper_min_list, theory_upper_max_list) = ([] for _ in range(5))

    for cur in nr_fit_dict.keys():
        fit_data = nr_fit_dict[cur]["fit_data"]
        if fit_data is None:
            continue
        delta_val = delta_map[cur]
        if fit_data.get("fit_type") == "single":
            detuning_list.append(delta_val)
            peak_list.append(fit_data.get("omega"))
            peak_unc_list.append(fit_data.get("omega_unc", 0.0))
            linewidth_list.append(fit_data.get("linewidth", 0.0))
        elif fit_data.get("fit_type") == "double":
            detuning_list.extend([delta_val, delta_val])
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
            detuning_double_list.append(delta_val)

        theory_detuning_list.append(delta_val)
        theory_lower_min_list.append(nr_fit_dict[cur]["theory_lower_min"])
        theory_lower_max_list.append(nr_fit_dict[cur]["theory_lower_max"])
        theory_upper_min_list.append(nr_fit_dict[cur]["theory_upper_min"])
        theory_upper_max_list.append(nr_fit_dict[cur]["theory_upper_max"])

    if len(peak_list) == 0:
        print("No NR peaks extracted for the overlay plot.")
        return

    detuning_array = np.array(detuning_list)
    peak_array = np.array(peak_list)
    peak_unc_array = np.array(peak_unc_list)
    linewidth_array = np.array(linewidth_list)
    splitting_ratio_array = np.array(splitting_ratio_list)
    detuning_double_array = np.array(detuning_double_list)

    theory_detuning_array = np.array(theory_detuning_list)
    theory_lower_min_array = np.array(theory_lower_min_list)
    theory_lower_max_array = np.array(theory_lower_max_list)
    theory_upper_min_array = np.array(theory_upper_min_list)
    theory_upper_max_array = np.array(theory_upper_max_list)

    sort_idx = np.argsort(theory_detuning_array)
    theory_detuning_array = theory_detuning_array[sort_idx]
    theory_lower_min_array = theory_lower_min_array[sort_idx]
    theory_lower_max_array = theory_lower_max_array[sort_idx]
    theory_upper_min_array = theory_upper_min_array[sort_idx]
    theory_upper_max_array = theory_upper_max_array[sort_idx]

    # Extract kappa values from the DataFrames
    cavity_kappa_values = cavity_df["kappa"].values
    yig_kappa_values = yig_df["kappa"].values

    cavity_freq_values = cavity_df["omega"].values
    avg_cavity_freq = np.mean(cavity_freq_values)

    # Calculate the average kappa for cavity and YIG
    avg_cavity_kappa = np.mean(cavity_kappa_values)
    print('avg cavity freq is ', avg_cavity_freq)

    xaxis_fig3 = np.array([.0160, .0165, .0170, .0175])
    print("The Fig 3 x axis is, therefore we convert it to YIG freqs then divide it by 28e9")
    print((avg_cavity_freq - xaxis_fig3) / 28)
    print("Or we can just divide the detuning value, and call that Delta B")
    print(xaxis_fig3 / 28)

    avg_yig_kappa = np.mean(yig_kappa_values)

    avg_kappa = (avg_cavity_kappa + avg_yig_kappa) / 2
    avg_cavity_f = cavity_freq_values.mean()

    nr_plot.plot_final_peak_plot(
        theory_detuning_array=theory_detuning_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        optimal_J=config.optimal_J,
        kappa_val=avg_kappa,
        detuning_array=detuning_array,
        peak_array=peak_array,
        experiment_id=experiment_id,
        overlay_folder=overlay_folder,
        theory_upper_min_array=theory_upper_min_array,
        theory_upper_max_array=theory_upper_max_array,
        peak_unc_array=peak_unc_array,
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end,
        errorbar_color="red",
    )

    # Plot the final plot again, but use Linewidths for the errorbar values instead
    nr_plot.plot_final_peak_plot(
        theory_detuning_array=theory_detuning_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        kappa_val=avg_kappa,
        optimal_J=config.optimal_J,
        detuning_array=detuning_array,
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

    nr_plot.plot_fig2(
        theory_detuning_array=theory_detuning_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        optimal_J=config.optimal_J,
        kappa_val=avg_kappa,
        detuning_array=detuning_array,
        peak_array=peak_array,
        experiment_id=experiment_id,
        overlay_folder=overlay_folder,
        theory_upper_min_array=theory_upper_min_array,
        theory_upper_max_array=theory_upper_max_array,
        peak_unc_array=peak_unc_array,
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end,
        overlap_region_max_freq=config.overlap_region_max_freq,
        overlap_region_min_freq=config.overlap_region_min_freq,
        errorbar_color="red",
        cavity_freq=avg_cavity_f,
        yig_freqs=yig_freqs,
        lo_freqs=nr_freqs
    )

    # ------------------ LINEWIDTH PLOTS ------------------
    nr_plot.plot_linewidth_vs_detuning(detuning_array, linewidth_array, config.optimal_J, linewidth_unc_array=None,
                                       experiment_id=experiment_id, output_folder=overlay_folder, errorbar_color="blue",
                                       overlap_region_start=config.overlap_region_start,
                                       overlap_region_end=config.overlap_region_end)

    nr_plot.plot_sum_linewidth_vs_detuning(detuning_array, linewidth_array, config.optimal_J, linewidth_unc_array=None,
                                           experiment_id=experiment_id, output_folder=overlay_folder,
                                           errorbar_color="blue",
                                           overlap_region_start=config.overlap_region_start,
                                           overlap_region_end=config.overlap_region_end)

    # Now call the new plotting function:
    nr_plot.plot_splitting_ratio_vs_detuning(
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
    for cur, data in nr_fit_dict.items():
        fit_data = data.get("fit_data")
        if fit_data is None:
            continue
        # Only consider single Lorentzian fits
        if fit_data.get("fit_type") == "single":
            delta_val = delta_map.get(cur)
            if delta_val is None:
                continue
            # Check if the detuning is below the EP threshold
            if delta_val < ep_threshold:
                # Extract the single peak frequency (named "omega")
                omega_val = fit_data.get("omega")
                if omega_val is not None:
                    single_peak_frequencies.append(omega_val)

    if single_peak_frequencies:
        avg_single_peak_frequency = np.mean(single_peak_frequencies)
        print("Average frequency for single peaks with detuning less than EP threshold:",
              avg_single_peak_frequency)

        # Plot the figure 3 plot
        nr_plot.plot_fig3(nr_power, nr_currents, nr_freqs, delta_df,
                          experiment_id, nr_settings, avg_kappa, avg_single_peak_frequency, fig3_folder,
                          optimal_J=config.optimal_J, frequency_radius=.002, yig_power_grid=yig_power,
                          yig_freqs=yig_freqs)

        # Plot the figure 3 plot
        nr_plot.plot_research_figure(nr_power, nr_currents, nr_freqs, Delta_df=delta_df,
                                     experiment_id=experiment_id, settings=nr_settings,
                                     avg_single_peak_frequency=avg_single_peak_frequency,
                                     fig3_folder=fig3_folder, smoothing_window=50, optimal_J=config.optimal_J,
                                     yig_power_grid=yig_power, yig_freqs=yig_freqs, kappa_val=avg_kappa,
                                     yig_trace_freq=5.990425)

    else:
        print("WARNING: No single peaks with detuning less than EP threshold were found.")


if __name__ == "__main__":
    main()
