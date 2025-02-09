# plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from nr_fitting import iterative_NR_fit


def plot_individual_trace(current_value, frequencies, power_dbm, readout_type, base_folder, fit_data,
                          detuning_val=None, order_prefix="", simulated_trace=None, simulated_trace_peak_idxs=None,
                          simulated_vertical_offset=None,
                          peak1_lower_bound=None,
                          peak1_upper_bound=None,
                          peak2_lower_bound=None,
                          peak2_upper_bound=None):
    """
    Plot an individual trace. For NR traces, if detuning_val is provided,
    include it in the title and filename. The order_prefix (if provided) is
    prepended to the file name.

    Saves the plot in <base_folder>/debug/<readout_type>
    """
    # Use the debug subfolder for individual trace plots
    folder = ensure_debug_folder(base_folder, readout_type)

    freqs_ghz = frequencies / 1e9
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(freqs_ghz, power_dbm, "b", label="Data")

    if fit_data is not None and fit_data.get("fit_result") is not None:
        fit_result = fit_data["fit_result"]
        x_fit = fit_data.get("x_fit", np.linspace(freqs_ghz.min(), freqs_ghz.max(), 200))
        best_fit_linear = fit_result.eval(x=x_fit)
        best_fit_db = 10 * np.log10(best_fit_linear)
        ax.plot(x_fit, best_fit_db, "m--", label="Lorentzian Fit")

        if fit_data.get("fit_type") == "single":
            center = fit_result.params["lz_center"].value
            peak_val_db = 10 * np.log10(fit_result.eval(x=np.array([center])))
            ax.plot(center, peak_val_db, "r*", markersize=10, label="Peak")
            ax.annotate(f"Peak: {center} GHz", (center, peak_val_db),
                        textcoords="offset points", xytext=(0, 20), ha="center", color="red")
        elif fit_data.get("fit_type") == "double":
            # Plot peak markers for each Lorentzian
            try:
                center1 = fit_result.params["lz1_center"].value
                peak_val_db1 = 10 * np.log10(fit_result.eval(x=np.array([center1])))
                ax.plot(center1, peak_val_db1, "r*", markersize=10, label="Peak 1")
                ax.annotate(f"Peak 1: {center1} GHz", (center1, peak_val_db1),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="red")
            except Exception as e:
                print("Error in double fit (peak1):", e)
            try:
                center2 = fit_result.params["lz2_center"].value
                peak_val_db2 = 10 * np.log10(fit_result.eval(x=np.array([center2])))
                ax.plot(center2, peak_val_db2, "y*", markersize=10, label="Peak 2")
                ax.annotate(f"Peak 2: {center2} GHz", (center2, peak_val_db2),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="orange")
            except Exception as e:
                print("Error in double fit (peak2):", e)

            # --- NEW FUNCTIONALITY ---
            # Plot the individual Lorentzian components if available.
            try:
                # This assumes that the fit_result was built with components named "lz1" and "lz2"
                comps = fit_result.eval_components(x=x_fit)
                if "lz1_" in comps:
                    comp1_linear = comps["lz1_"]
                    comp1_db = 10 * np.log10(comp1_linear)
                    ax.plot(x_fit, comp1_db, "c:", label="Lorentzian 1")
                if "lz2_" in comps:
                    comp2_linear = comps["lz2_"]
                    comp2_db = 10 * np.log10(comp2_linear)
                    ax.plot(x_fit, comp2_db, "k:", label="Lorentzian 2")
            except Exception as e:
                print("Error plotting individual Lorentzian components:", e)
            # --- END NEW FUNCTIONALITY ---

    title = f"{readout_type.capitalize()} Trace at {current_value} A"
    file_suffix = f"{current_value}"
    if readout_type.lower() in ["nr", "normal"] and detuning_val is not None:
        title += f", Detuning = {detuning_val} GHz"
        file_suffix += f"_Delta_{detuning_val}"

    if simulated_trace is not None:
        ax.plot(freqs_ghz, simulated_trace - simulated_vertical_offset, "g--", label="Theory")
        if simulated_trace_peak_idxs is not None:
            for idx in simulated_trace_peak_idxs:
                ax.plot(freqs_ghz[idx], simulated_trace[idx] - simulated_vertical_offset, "g*", markersize=10,
                        label="Theory Peak")
                ax.annotate(f"Simulated Peak: {freqs_ghz[idx]:.3f} GHz", (freqs_ghz[idx], simulated_trace[idx]),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="green")

    # Add vertical lines for peak bounds, if they exist
    if peak1_lower_bound is not None:
        ax.axvline(peak1_lower_bound, color="red", linestyle="--", label="Peak 1 Lower Bound")
    if peak1_upper_bound is not None:
        ax.axvline(peak1_upper_bound, color="red", linestyle="--", label="Peak 1 Upper Bound")
    if peak2_lower_bound is not None:
        ax.axvline(peak2_lower_bound, color="orange", linestyle="--", label="Peak 2 Lower Bound")
    if peak2_upper_bound is not None:
        ax.axvline(peak2_upper_bound, color="orange", linestyle="--", label="Peak 2 Upper Bound")

    ax.set_xlabel("Frequency (GHz)", fontsize=14)
    ax.set_ylabel("Power (dBm)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()

    plot_filename = f"{order_prefix}{readout_type}_trace_current_{file_suffix}.png"
    plot_path = os.path.join(folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved individual {readout_type} trace plot for current = {current_value} A to {plot_path}")


def ensure_debug_folder(base_folder, readout_type):
    """
    Returns the full path for saving individual trace plots and creates it if necessary.
    All individual trace plots are saved in: <base_folder>/debug/<readout_type>
    """
    folder = os.path.join(base_folder, "debug", readout_type.lower())
    os.makedirs(folder, exist_ok=True)
    return folder


def debug_plot_individual_NR_traces(power_grid, currents, frequencies, base_folder, detuning_data):
    """
    For NR traces, sort by descending |Delta| and use iterative_NR_fit to carry over
    previous peak guesses. The order index is prepended to the filename.

    Saves individual trace plots in <base_folder>/debug/nr
    """
    readout_type = "nr"
    # Use the debug folder for individual trace plots
    folder = ensure_debug_folder(base_folder, readout_type)

    # Assume detuning_data is a DataFrame with columns "current" and "Delta"
    delta_map = detuning_data.set_index("current")["Delta"].to_dict()
    nr_info = []
    for i, cur in enumerate(currents):
        if cur in delta_map:
            nr_info.append((i, cur, delta_map[cur]))

    # Sort by descending absolute Delta (largest |Delta| first)
    nr_info.sort(key=lambda tup: -abs(tup[2]))

    prev_guesses = None
    for order, (i, cur, delta_val) in enumerate(nr_info, start=1):
        trace = power_grid[i, :]
        fit_data = iterative_NR_fit(cur, frequencies, trace, initial_guesses=prev_guesses)
        if fit_data is not None:
            if fit_data.get("fit_type") == "double":
                prev_guesses = [fit_data.get("peak1"), fit_data.get("peak2")]
            elif fit_data.get("fit_type") == "single":
                prev_guesses = [fit_data.get("omega"), fit_data.get("omega")]
        order_prefix = f"{order:03d}_"
        plot_individual_trace(cur, frequencies, trace, readout_type, base_folder, fit_data,
                              detuning_val=delta_val, order_prefix=order_prefix)


def plot_nr_peaks_only_vs_detuning(nr_power, nr_currents, nr_freqs, delta_df, base_folder, experiment_id):
    """
    Overlay plot of NR peaks versus detuning.

    Saves the plot directly in base_folder (e.g. plots/experiment_id/)
    """
    from nr_fitting import fit_NR_trace
    # Use base_folder directly for colorplots
    os.makedirs(base_folder, exist_ok=True)

    delta_map = delta_df.set_index("current")["Delta"]
    detuning_list = []
    peak_list = []
    peak_unc_list = []
    for i, cur in enumerate(nr_currents):
        if cur not in delta_map:
            continue
        delta_val = delta_map[cur]
        fit_data = fit_NR_trace(cur, nr_freqs, nr_power[i, :])
        if fit_data is None:
            continue
        if fit_data.get("fit_type") == "single":
            detuning_list.append(delta_val)
            peak_list.append(fit_data.get("omega"))
            peak_unc_list.append(fit_data.get("omega_unc", 0.0))
        elif fit_data.get("fit_type") == "double":
            detuning_list.extend([delta_val, delta_val])
            peak_list.extend([fit_data.get("peak1"), fit_data.get("peak2")])
            peak_unc_list.extend([fit_data.get("peak1_unc", 0.0), fit_data.get("peak2_unc", 0.0)])

    if len(peak_list) == 0:
        print("No NR peaks extracted for the overlay plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(detuning_list, peak_list, yerr=peak_unc_list, fmt="o",
                ecolor="red", capsize=4, label="NR Hybridized Peaks")
    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=14)
    ax.set_title("NR Peak Locations vs. Detuning", fontsize=14)
    ax.set_ylim(min(peak_list), max(peak_list))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    plot_filename = f"nr_peaks_overlay_exp_{experiment_id}.png"
    plot_path = os.path.join(base_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved NR peaks overlay plot to {plot_path}")


def plot_delta_colorplot(nr_power_grid, currents, frequencies, delta_df, experiment_id, settings, base_folder):
    """
    Plot a colorplot of NR power vs. detuning.

    Saves the plot directly in base_folder (e.g. plots/experiment_id/)
    """
    os.makedirs(base_folder, exist_ok=True)

    delta_map = delta_df.set_index("current")["Delta"]
    Delta_values = np.array([delta_map.get(c, np.nan) for c in currents])
    valid_mask = ~np.isnan(Delta_values)
    nr_power_grid = nr_power_grid[valid_mask, :]
    Delta_values = Delta_values[valid_mask]
    sort_idx = np.argsort(Delta_values)
    Delta_sorted = Delta_values[sort_idx]
    power_grid_sorted = nr_power_grid[sort_idx, :]

    n_rows = len(Delta_sorted)
    delta_edges = np.zeros(n_rows + 1)
    if n_rows > 1:
        delta_edges[1:-1] = (Delta_sorted[:-1] + Delta_sorted[1:]) / 2
        delta_edges[0] = Delta_sorted[0] - (Delta_sorted[1] - Delta_sorted[0]) / 2
        delta_edges[-1] = Delta_sorted[-1] + (Delta_sorted[-1] - Delta_sorted[-2]) / 2
    else:
        delta_edges[0] = Delta_sorted[0] - 0.001
        delta_edges[1] = Delta_sorted[0] + 0.001

    freqs_ghz = frequencies / 1e9
    n_cols = len(freqs_ghz)
    freq_edges = np.zeros(n_cols + 1)
    if n_cols > 1:
        freq_edges[1:-1] = (freqs_ghz[:-1] + freqs_ghz[1:]) / 2
        freq_edges[0] = freqs_ghz[0] - (freqs_ghz[1] - freqs_ghz[0]) / 2
        freq_edges[-1] = freqs_ghz[-1] + (freqs_ghz[-1] - freqs_ghz[-2]) / 2
    else:
        freq_edges[0] = freqs_ghz[0] - 0.01
        freq_edges[1] = freqs_ghz[0] + 0.01

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(delta_edges, freq_edges, power_grid_sorted.T, shading="auto", cmap="inferno")
    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Frequency (GHz)", fontsize=14)

    title = f"Experiment {experiment_id} NR Hybridized Data\n"
    title += f"Loop Att: {settings.get('set_loop_att', 'N/A')} dB, "
    title += f"YIG FB Phase: {settings.get('set_yig_fb_phase_deg', 'N/A')}°"
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(c, ax=ax, label="Power (dBm)")
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()

    plot_filename = f"NR_colorplot_detuning_exp_{experiment_id}.png"
    plot_path = os.path.join(base_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved NR color plot (detuning) to {plot_path}")


def plot_raw_colorplot(power_grid, currents, frequencies, experiment_id, settings, base_folder, readout_type="nr"):
    """
    Plot a raw data colorplot (current as X-axis).

    Saves the plot directly in base_folder (e.g. plots/experiment_id/)
    """
    os.makedirs(base_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(currents, frequencies / 1e9, power_grid.T, shading="auto", cmap="inferno")
    ax.set_xlabel("Current (A)", fontsize=14)
    ax.set_ylabel("Frequency (GHz)", fontsize=14)

    title = f"Experiment {experiment_id} {readout_type.upper()} Raw Data\n"
    if settings:
        title += f"Loop Att: {settings.get('set_loop_att', 'N/A')} dB"
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(c, ax=ax, label="Power (dBm)")
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()

    plot_filename = f"{readout_type}_raw_colorplot_exp_{experiment_id}.png"
    plot_path = os.path.join(base_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved raw {readout_type.upper()} color plot (current as X-axis) to {plot_path}")


def plot_final_peak_plot(theory_detuning_array, theory_lower_min_array, theory_lower_max_array,
                         optimal_J, detuning_array,
                         peak_array, peak_unc_array, experiment_id, overlay_folder,
                         theory_upper_min_array, theory_upper_max_array):
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

    # Add a vertical line at 2*J
    ax.axvline(x=2 * optimal_J, color="red", linestyle="--", label="Δ = 2J")

    # --------------------- Start Add Shading for Zone III ---------------------
    # Add a shaded region that covers the largest 50 values of detuning (X points)
    # This is to highlight the region where the NR peaks are most likely to be found
    # First, determine the 50 largest detuning values from the experimental data.
    if len(detuning_array) >= 50:
        largest_50_detuning = np.sort(detuning_array)[-50:]
    else:
        largest_50_detuning = detuning_array

    # Get the minimum and maximum values from these 50 points.
    region_xmin = np.min(largest_50_detuning)
    region_xmax = np.max(largest_50_detuning)

    # Shade this region over the full y-axis.
    ax.axvspan(region_xmin, region_xmax, color="orange", alpha=0.2, label="J Calculation Region")
    # --------------------- End Add Shading for Zone III --------------------

    # Plot the experimental data on top
    ax.errorbar(detuning_array, peak_array,
                yerr=peak_unc_array,
                fmt="o", ecolor="red",
                capsize=4, label="NR Hybridized Peaks (Data)",
                markersize=2, color="black")

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
