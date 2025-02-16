# plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from nr_fitting import iterative_NR_fit
from scipy.signal import savgol_filter


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
            # --- SINGLE LORENTZIAN FIT ---
            center = fit_result.params["lz_center"].value
            # Evaluate the fit at the center to get the peak value (in linear units)
            peak_linear = fit_result.eval(x=np.array([center]))[0]
            # Compute the half–maximum (3 dB point) in linear units, then convert to dB
            half_max_linear = peak_linear / 2
            half_max_db = 10 * np.log10(half_max_linear)
            # Plot the peak marker at the actual peak value (for reference)
            ax.plot(center, 10 * np.log10(peak_linear), "r*", markersize=10, label="Peak")
            ax.annotate(f"Peak: {center:.3f} GHz", (center, 10 * np.log10(peak_linear)),
                        textcoords="offset points", xytext=(0, 20), ha="center", color="red")
            # Plot horizontal line representing the FWHM at the 3 dB level
            linewidth = fit_data.get("linewidth", None)
            if linewidth is not None and not np.isnan(linewidth):
                x_left = center - linewidth / 2
                x_right = center + linewidth / 2
                ax.hlines(y=half_max_db, xmin=x_left, xmax=x_right,
                          color="purple", linestyle="--", label="FWHM (3 dB)")
        elif fit_data.get("fit_type") == "double":
            # --- DOUBLE LORENTZIAN FIT ---
            # For each component, use its own evaluation to compute the 3 dB level.
            try:
                center1 = fit_result.params["lz1_center"].value
                comps = fit_result.eval_components(x=np.array([center1]))
                if "lz1_" in comps:
                    comp1_val = comps["lz1_"][0]  # get the evaluated value (linear)
                    half_max1 = comp1_val / 2
                    half_max1_db = 10 * np.log10(half_max1)
                else:
                    # Fallback if component not found
                    comp1_val = fit_result.eval(x=np.array([center1]))[0]
                    half_max1_db = 10 * np.log10(comp1_val / 2)
                peak_val_db1 = 10 * np.log10(fit_result.eval(x=np.array([center1]))[0])
                ax.plot(center1, peak_val_db1, "r*", markersize=10, label="Peak 1")
                ax.annotate(f"Peak 1: {center1:.3f} GHz", (center1, peak_val_db1),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="red")
                linewidth1 = fit_data.get("peak1_linewidth", None)
                if linewidth1 is not None and not np.isnan(linewidth1):
                    x_left1 = center1 - linewidth1 / 2
                    x_right1 = center1 + linewidth1 / 2
                    ax.hlines(y=half_max1_db, xmin=x_left1, xmax=x_right1,
                              color="purple", linestyle="--", label="FWHM 1 (3 dB)")
            except Exception as e:
                print("Error in double fit (peak1):", e)
            try:
                center2 = fit_result.params["lz2_center"].value
                comps = fit_result.eval_components(x=np.array([center2]))
                if "lz2_" in comps:
                    comp2_val = comps["lz2_"][0]
                    half_max2 = comp2_val / 2
                    half_max2_db = 10 * np.log10(half_max2)
                else:
                    comp2_val = fit_result.eval(x=np.array([center2]))[0]
                    half_max2_db = 10 * np.log10(comp2_val / 2)
                peak_val_db2 = 10 * np.log10(fit_result.eval(x=np.array([center2]))[0])
                ax.plot(center2, peak_val_db2, "y*", markersize=10, label="Peak 2")
                ax.annotate(f"Peak 2: {center2:.3f} GHz", (center2, peak_val_db2),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="orange")
                linewidth2 = fit_data.get("peak2_linewidth", None)
                if linewidth2 is not None and not np.isnan(linewidth2):
                    x_left2 = center2 - linewidth2 / 2
                    x_right2 = center2 + linewidth2 / 2
                    ax.hlines(y=half_max2_db, xmin=x_left2, xmax=x_right2,
                              color="brown", linestyle="--", label="FWHM 2 (3 dB)")
            except Exception as e:
                print("Error in double fit (peak2):", e)

            # --- Plot individual Lorentzian components if available ---
            try:
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
                ax.annotate(f"Simulated Peak: {freqs_ghz[idx]:.3f} GHz", (freqs_ghz[idx],
                                                                          simulated_trace[idx]),
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

    import os  # Ensure os is imported if not already
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
                         theory_upper_min_array, theory_upper_max_array,
                         overlap_region_start=None, overlap_region_end=None, errorbar_color="red", filename_prepend=""):
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
    if overlap_region_start is not None:
        ax.axvline(x=overlap_region_start, color="black", linestyle="--")
    if overlap_region_end is not None:
        ax.axvline(x=overlap_region_end, color="black", linestyle="--")

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
    # ax.axvspan(region_xmin, region_xmax, color="orange", alpha=0.2, label="J Calculation Region")
    ax.axvline(x=region_xmin, color="black", linestyle="--")
    # --------------------- End Add Shading for Zone III --------------------

    # Plot the experimental data on top
    ax.errorbar(detuning_array, peak_array,
                yerr=peak_unc_array,
                fmt="o", ecolor=errorbar_color,
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
    # Move the legend to the bottom left and make it opaque.
    ax.legend(loc="lower left", framealpha=1)
    plt.tight_layout()

    overlay_plot_path = os.path.join(overlay_folder, f"{filename_prepend}nr_peaks_overlay_exp_{experiment_id}.png")
    plt.savefig(overlay_plot_path, dpi=300)
    plt.close(fig)
    print("Saved NR peaks overlay plot to", overlay_plot_path)


def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='same')


def plot_fig3(nr_power_grid, currents, frequencies, delta_df, experiment_id, settings,
              avg_single_peak_frequency, fig3_folder, frequency_radius=0.002,
              smoothing_window=100, deriv_ylim=None, optimal_J=None):
    """
    Create Figure 3 consisting of three vertically stacked subplots:

      1) Top: A colorplot with detuning on the X-axis and frequency (in GHz) on the Y-axis.
         Overlaid on the colorplot are three horizontal dotted lines at the average single‐peak
         frequency (from your NR fits) and at frequencies ±<frequency_radius> from it.

      2) Middle: A 2D trace plot of power (in dBm, s_21) versus detuning.
         Three traces are extracted from the colorplot at the center frequency and at the two offsets.
         Both the original (solid) and a smoothed version (dashed) are plotted.

      3) Bottom: A 2D trace plot of the derivative (numerical d(Power)/d(Detuning)) versus detuning.
         Only the smoothed derivative traces (for center, center + <frequency_radius>, and center - <frequency_radius>)
         are plotted.

    The colors used for the three traces in the middle and bottom panels are identical to the
    horizontal dotted lines in the top panel.

    Parameters:
      nr_power_grid : 2D numpy array
          The NR power data. Rows correspond to different detuning values (ordered by current)
          and columns correspond to frequency points.
      currents : array-like
          The current values corresponding to the rows in nr_power_grid.
      frequencies : 1D numpy array
          The frequency axis (in Hz) corresponding to the columns of nr_power_grid.
      delta_df : pandas DataFrame
          Contains detuning information with at least a "current" column and a "Delta" column.
      experiment_id : str or int
          Identifier for the experiment (used in titles/filenames).
      settings : dict
          A dictionary of settings (for use in plot titles if desired).
      avg_single_peak_frequency : float
          The average frequency (in GHz) computed from the single-Lorentzian fits.
      fig3_folder : str
          Path to the folder where the figure will be saved.
      frequency_radius : float, optional
          The offset (in GHz) above and below the average frequency (default is 0.002, i.e. 2 MHz).
      smoothing_window : int, optional
          The window length for the Savitzky–Golay filter (default is 20).
      deriv_ylim : tuple or None, optional
          If provided, sets the y-limits for the derivative plot (bottom subplot).
    """
    # --- Prepare the colorplot data ---
    # Create a mapping from current to detuning from the DataFrame.
    delta_map = delta_df.set_index("current")["Delta"].to_dict()
    Delta_values = np.array([delta_map.get(c, np.nan) for c in currents])
    valid_mask = ~np.isnan(Delta_values)
    # Filter out any rows with no detuning data.
    nr_power_grid = nr_power_grid[valid_mask, :]
    Delta_values = Delta_values[valid_mask]
    # Sort the data by detuning.
    sort_idx = np.argsort(Delta_values)
    Delta_sorted = Delta_values[sort_idx]
    power_grid_sorted = nr_power_grid[sort_idx, :]

    # Compute the "edges" for the detuning axis (for pcolormesh).
    n_rows = len(Delta_sorted)
    delta_edges = np.zeros(n_rows + 1)
    if n_rows > 1:
        delta_edges[1:-1] = (Delta_sorted[:-1] + Delta_sorted[1:]) / 2
        delta_edges[0] = Delta_sorted[0] - (Delta_sorted[1] - Delta_sorted[0]) / 2
        delta_edges[-1] = Delta_sorted[-1] + (Delta_sorted[-1] - Delta_sorted[-2]) / 2
    else:
        delta_edges[0] = Delta_sorted[0] - 0.001
        delta_edges[1] = Delta_sorted[0] + 0.001

    # Convert the frequency axis to GHz.
    freqs_ghz = frequencies / 1e9
    n_cols = len(freqs_ghz)
    freq_edges = np.zeros(n_cols + 1)
    if n_cols > 1:
        freq_edges[1:-1] = (freqs_ghz[:-1] + freqs_ghz[1:]) / 2
        freq_edges[0] = freqs_ghz[0] - (freqs_ghz[1] - freqs_ghz[0]) / 2
        freq_edges[-1] = freqs_ghz[-1] + (freqs_ghz[-1] - freqs_ghz[-2]) / 2
    else:
        freq_edges[0] = freqs_ghz[0] - 0.001
        freq_edges[1] = freqs_ghz[0] + 0.001

    # --- Determine the target frequencies ---
    # Here, avg_single_peak_frequency is assumed to be in GHz.
    target_center = avg_single_peak_frequency
    target_plus = target_center + frequency_radius  # e.g., +2 MHz (0.002 GHz)
    target_minus = target_center - frequency_radius  # e.g., -2 MHz (0.002 GHz)

    # Find the index in the frequency axis closest to each target.
    idx_center = np.argmin(np.abs(freqs_ghz - target_center))
    idx_plus = np.argmin(np.abs(freqs_ghz - target_plus))
    idx_minus = np.argmin(np.abs(freqs_ghz - target_minus))

    # Define colors for consistency.
    color_center = 'red'
    color_plus = 'blue'
    color_minus = 'green'

    # --- Create the figure with 3 vertical subplots ---
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 15))

    # (1) Top subplot: Colorplot with horizontal dotted lines.
    ax0 = axes[0]
    c = ax0.pcolormesh(delta_edges, freq_edges, power_grid_sorted.T, shading="auto", cmap="inferno")
    cb = fig.colorbar(c, ax=ax0, label="Power (dBm)")
    # Draw horizontal dotted lines at the target frequencies.
    ax0.axhline(target_center, color=color_center, linestyle='--', linewidth=2, label='Center Frequency')
    ax0.axhline(target_plus, color=color_plus, linestyle='--', linewidth=2, label='Center + 2 MHz')
    ax0.axhline(target_minus, color=color_minus, linestyle='--', linewidth=2, label='Center - 2 MHz')
    ax0.set_ylabel("Frequency (GHz)", fontsize=14)
    ax0.set_title(f"NR Colorplot with Detuning (Exp {experiment_id})", fontsize=16)
    ax0.legend(loc='upper right')

    # (2) Middle subplot: 2D trace plot (Power vs. Detuning).
    ax1 = axes[1]
    # For a fixed frequency, the trace is given by the corresponding column of power_grid_sorted.
    trace_center = power_grid_sorted[:, idx_center]
    trace_plus = power_grid_sorted[:, idx_plus]
    trace_minus = power_grid_sorted[:, idx_minus]

    # Use scipy's savgol_filter for smoothing.
    # Ensure the window length is odd.
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    smoothed_center = savgol_filter(trace_center, smoothing_window, 2)
    smoothed_plus = savgol_filter(trace_plus, smoothing_window, 2)
    smoothed_minus = savgol_filter(trace_minus, smoothing_window, 2)

    # Plot the original (solid) and smoothed (dashed) traces.
    ax1.plot(Delta_sorted, trace_center, color=color_center, label='Center Frequency')
    ax1.plot(Delta_sorted, trace_plus, color=color_plus, label='Center + 2 MHz')
    ax1.plot(Delta_sorted, trace_minus, color=color_minus, label='Center - 2 MHz')
    ax1.plot(Delta_sorted, smoothed_center, color=color_center, linestyle='--',
             label='Center Frequency (Smoothed)')
    ax1.plot(Delta_sorted, smoothed_plus, color=color_plus, linestyle='--',
             label='Center + 2 MHz (Smoothed)')
    ax1.plot(Delta_sorted, smoothed_minus, color=color_minus, linestyle='--',
             label='Center - 2 MHz (Smoothed)')
    ax1.set_ylabel("Power (dBm)", fontsize=14)
    ax1.set_title("2D Trace Plot: Power vs. Detuning", fontsize=16)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # --- Finalize the figure ---
    fig.suptitle("Figure 3: NR Detuning and Sensitivity", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if optimal_J is not None:
        plt.axvline(2 * optimal_J, color="red", linestyle="--", label="Δ = 2J")

    # Save the figure.
    os.makedirs(fig3_folder, exist_ok=True)
    plot_filename = f"FIG3_exp_{experiment_id}.png"
    plot_path = os.path.join(fig3_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved FIG3 to {plot_path}")


def plot_linewidth_vs_detuning(detuning_array, linewidth_array, optimal_J, linewidth_unc_array=None,
                               experiment_id=None, output_folder=".", errorbar_color="cyan",
                               overlap_region_start=None, overlap_region_end=None):
    """
    Create a plot of linewidth vs. detuning.

    Parameters:
      detuning_array : numpy.ndarray
         Array of detuning values (in GHz) to be plotted on the X axis.
      linewidth_array : numpy.ndarray
         Array of linewidth values (in GHz) to be plotted on the Y axis.
         (When there are two linewidths at a given detuning, both values should be included.)
      linewidth_unc_array : numpy.ndarray, optional
         Array of linewidth uncertainties (in GHz) for errorbars. If None, no errorbars are drawn.
      experiment_id : str or int, optional
         Experiment identifier (used to name the output file).
      output_folder : str
         Folder in which to save the plot.
      errorbar_color : str
         Color for the error bars (and/or markers).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # If uncertainties are provided, use an errorbar plot; otherwise, simply scatter.
    if linewidth_unc_array is not None:
        ax.errorbar(detuning_array, linewidth_array, yerr=linewidth_unc_array,
                    fmt="o", ecolor=errorbar_color, capsize=4, markersize=4,
                    color="black", label="Linewidth")
    else:
        ax.plot(detuning_array, linewidth_array, "o", color=errorbar_color, label="Linewidth")

    # Add a vertical line at 2*J.
    ax.axvline(x=2 * optimal_J, color="red", linestyle="--", label="Δ = 2J")
    if overlap_region_start is not None:
        ax.axvline(x=overlap_region_start, color="black", linestyle="--")
    if overlap_region_end is not None:
        ax.axvline(x=overlap_region_end, color="black", linestyle="--")

    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Linewidth (GHz)", fontsize=14)
    ax.set_title("Linewidth vs. Detuning", fontsize=16)
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    if experiment_id is not None:
        plot_filename = f"linewidth_vs_detuning_exp_{experiment_id}.png"
    else:
        plot_filename = "linewidth_vs_detuning.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved linewidth vs. detuning plot to {plot_path}")


def plot_sum_linewidth_vs_detuning(detuning_array, linewidth_array, optimal_J, linewidth_unc_array=None,
                                   experiment_id=None, output_folder=".", errorbar_color="cyan",
                                   overlap_region_start=None, overlap_region_end=None):
    """
    Create a plot of average linewidth vs. detuning.

    For each unique detuning value, if there are two linewidth values, they are averaged;
    if there is only one linewidth value, that value is used. This ensures that exactly one
    point is plotted for each detuning.

    Parameters:
      detuning_array : numpy.ndarray
         Array of detuning values (in GHz) corresponding to each measurement.
         (When there are two peaks at a given detuning, this array contains duplicate values.)
      linewidth_array : numpy.ndarray
         Array of linewidth values (in GHz) corresponding to each measurement.
         (When there are two peaks at a given detuning, both values are included.)
      optimal_J : float
         The J parameter from which 2*J is computed and drawn as a vertical line.
      linewidth_unc_array : numpy.ndarray, optional
         Array of linewidth uncertainties (in GHz) for errorbars. If None, no errorbars are drawn.
      experiment_id : str or int, optional
         Experiment identifier (used to name the output file).
      output_folder : str
         Folder in which to save the plot.
      errorbar_color : str
         Color for the error bars (and/or markers).
      overlap_region_start : float, optional
         X-axis value at which to draw a vertical line for the start of an overlap region.
      overlap_region_end : float, optional
         X-axis value at which to draw a vertical line for the end of an overlap region.
    """
    # Group by unique detuning values and average the corresponding linewidths.
    # (Assumes that duplicate detuning values are exactly equal.)
    unique_detunings = np.unique(detuning_array)
    avg_linewidth = []
    if linewidth_unc_array is not None:
        avg_linewidth_unc = []
        for d in unique_detunings:
            mask = (detuning_array == d)
            avg_linewidth.append(np.sum(linewidth_array[mask]))
            avg_linewidth_unc.append(np.sum(linewidth_unc_array[mask]))
        avg_linewidth = np.array(avg_linewidth)
        avg_linewidth_unc = np.array(avg_linewidth_unc)
    else:
        for d in unique_detunings:
            mask = (detuning_array == d)
            avg_linewidth.append(np.sum(linewidth_array[mask]))
        avg_linewidth = np.array(avg_linewidth)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    if linewidth_unc_array is not None:
        ax.errorbar(unique_detunings, avg_linewidth, yerr=avg_linewidth_unc,
                    fmt="o", ecolor=errorbar_color, capsize=4, markersize=4,
                    color="black", label="Avg. Linewidth")
    else:
        ax.plot(unique_detunings, avg_linewidth, "o", color=errorbar_color, label="Avg. Linewidth")

    # Add vertical lines at 2*J and (if provided) for the overlap region.
    ax.axvline(x=2 * optimal_J, color="red", linestyle="--", label="Δ = 2J")
    if overlap_region_start is not None:
        ax.axvline(x=overlap_region_start, color="black", linestyle="--")
    if overlap_region_end is not None:
        ax.axvline(x=overlap_region_end, color="black", linestyle="--")

    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Total Linewidth (GHz)", fontsize=14)
    ax.set_title("Total (Sum) Linewidth vs. Detuning", fontsize=16)
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    # Save the plot to the output folder.
    if experiment_id is not None:
        plot_filename = f"avg_linewidth_vs_detuning_exp_{experiment_id}.png"
    else:
        plot_filename = "avg_linewidth_vs_detuning.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved average linewidth vs. detuning plot to {plot_path}")


def plot_splitting_ratio_vs_detuning(detuning_array, splitting_ratio_array, optimal_J,
                                     experiment_id=None, output_folder=".",
                                     errorbar_color="magenta",
                                     overlap_region_start=None, overlap_region_end=None):
    """
    Create a plot of the splitting-to-linewidth ratio vs. detuning.

    For each double–Lorentzian fit, the splitting is defined as the absolute difference
    between the two peak centers and the average linewidth is defined as the average of the
    two FWHM values. The ratio (splitting / average linewidth) is a measure of how well
    resolved the two peaks are.

    Parameters:
      detuning_array : numpy.ndarray
         Array of detuning values (in GHz) for each double–Lorentzian fit.
      splitting_ratio_array : numpy.ndarray
         Array of the splitting-to-linewidth ratio (dimensionless) for each double fit.
      optimal_J : float
         The optimal J parameter (in GHz) so that a vertical line is drawn at Δ = 2J.
      experiment_id : str or int, optional
         Identifier for the experiment (used to name the output file).
      output_folder : str
         Folder in which to save the plot.
      errorbar_color : str
         Color to use for the markers (and error bars if you decide to add uncertainties).
      overlap_region_start : float, optional
         If provided, draws a vertical line at this detuning value.
      overlap_region_end : float, optional
         If provided, draws a vertical line at this detuning value.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data as a scatter plot. (If you have uncertainties for the ratio,
    # you could extend this to an errorbar plot.)
    ax.plot(detuning_array, splitting_ratio_array, "o", color=errorbar_color, label="Splitting/Linewidth")

    # Draw a vertical line at Δ = 2J.
    ax.axvline(x=2 * optimal_J, color="red", linestyle="--", label="Δ = 2J")
    if overlap_region_start is not None:
        ax.axvline(x=overlap_region_start, color="black", linestyle="--")
    if overlap_region_end is not None:
        ax.axvline(x=overlap_region_end, color="black", linestyle="--")

    ax.set_xlabel("Detuning Δ (GHz)", fontsize=14)
    ax.set_ylabel("Splitting / Average Linewidth", fontsize=14)
    ax.set_title("Peak Resolving Ratio vs. Detuning", fontsize=16)
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    if experiment_id is not None:
        plot_filename = f"splitting_ratio_vs_detuning_exp_{experiment_id}.png"
    else:
        plot_filename = "splitting_ratio_vs_detuning.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved splitting ratio vs. detuning plot to {plot_path}")
