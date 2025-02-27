# plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from nr_fitting import iterative_NR_fit
from scipy.signal import savgol_filter

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"


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
    plt.savefig(plot_path, dpi=400)
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


def plot_delta_colorplot(nr_power_grid, currents, frequencies, delta_df, experiment_id, settings, base_folder,
                         filename_prepend=''):
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

    plot_filename = f"{filename_prepend}NR_colorplot_detuning_exp_{experiment_id}.png"
    plot_path = os.path.join(base_folder, plot_filename)
    plt.savefig(plot_path, dpi=400)
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
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"Saved raw {readout_type.upper()} color plot (current as X-axis) to {plot_path}")


def plot_final_peak_plot(theory_detuning_array, theory_lower_min_array, theory_lower_max_array,
                         optimal_J, kappa_val, detuning_array,
                         peak_array, peak_unc_array, experiment_id, overlay_folder,
                         theory_upper_min_array, theory_upper_max_array,
                         overlap_region_start=None, overlap_region_end=None, errorbar_color="red", filename_prepend=""):
    # Define font sizes
    label_fs = 22
    legend_fs = 12
    tick_fs = 20

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
    ax.axvline(x=2 * optimal_J, color="red", linestyle="--", label=r'$\Delta f = 2J$')
    # Add a vertical line at 2* sqrt(J**2 + kappa**2)
    ax.axvline(x=1 * np.sqrt(4 * optimal_J ** 2 + kappa_val ** 2), color="cyan", linestyle="--",
               label=r'$\Delta f = \sqrt{4J^2 + \kappa^2}$')

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
    ax.axvline(x=region_xmin, color="black", linestyle="--")
    # --------------------- End Add Shading for Zone III --------------------

    # Plot the experimental data on top
    ax.errorbar(detuning_array, peak_array,
                yerr=peak_unc_array,
                fmt="o", ecolor=errorbar_color,
                capsize=4, label="Hybridized Peaks (Data)",
                markersize=2, color="black")

    # Set axis labels with bold font weight
    ax.set_xlabel("Δf [GHz]", fontsize=label_fs, fontweight="bold")
    ax.set_ylabel("Peak Frequency [GHz]", fontsize=label_fs, fontweight="bold")

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Tidy up plot ranges, in case some shading is out of range
    y_min = min(peak_array.min(), theory_lower_min_array.min(), theory_upper_min_array.min())
    y_max = max(peak_array.max(), theory_lower_max_array.max(), theory_upper_max_array.max())
    ax.set_ylim(y_min, y_max)

    # Move the legend to the bottom left and make it opaque.
    ax.legend(loc="lower left", framealpha=1, fontsize=legend_fs)

    plt.tight_layout()

    overlay_plot_path = os.path.join(overlay_folder, f"{filename_prepend}nr_peaks_overlay_exp_{experiment_id}.png")
    plt.savefig(overlay_plot_path, dpi=400)
    plt.close(fig)
    print("Saved NR peaks overlay plot to", overlay_plot_path)


def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='same')


def plot_fig3(nr_power_grid, currents, frequencies, Delta_df, experiment_id, settings,
              avg_single_peak_frequency, fig3_folder, frequency_radius=0.002,
              smoothing_window=50, deriv_ylim=None, optimal_J=None, yig_power_grid=None, yig_freqs=None):
    """
    Create a two-panel figure with a colorplot on top and a 2D trace plot below.
    The figure is formatted with minimal whitespace, a shared colorbar (only spanning the top
    subplot), and legends with opaque borders. The naming conventions for the lines are updated:

      Top plot:
         - Red horizontal line:   f_EP = {target_center} GHz
         - Blue horizontal line:  f_UB = {target_plus} GHz
         - Green horizontal line: f_LB = {target_minus} GHz

      Bottom plot (only solid lines are labeled):
         - Red:   S_{21} (f = f_EP = {target_center} GHz)
         - Blue:  S_{21} (f = f_UB = {target_plus} GHz)
         - Green: S_{21} (f = f_LB = {target_minus} GHz)

    Additionally, if optimal_J is provided, a vertical line at Δ = -2 * optimal_J is drawn in neon green
    (hex: #39FF14) in both plots and labeled as "EP Line".

    The shared x-axis is labeled as '$\Delta \kappa$ [GHz]'. All text is enlarged for a publication-ready appearance.
    """

    # -------------------------
    # Prepare the colorplot data
    # -------------------------
    # Map from voltage to detuning (K) using the DataFrame.
    Delta_map = Delta_df.set_index("current")["Delta"].to_dict()
    Delta_values = np.array([Delta_map.get(c, np.nan) for c in currents])
    valid_mask = ~np.isnan(Delta_values)
    nr_power_grid = nr_power_grid[valid_mask, :]
    yig_power_grid = yig_power_grid[valid_mask, :] if yig_power_grid is not None else None
    Delta_values = Delta_values[valid_mask]

    # Sort by detuning.
    sort_idx = np.argsort(Delta_values)
    Delta_sorted = Delta_values[sort_idx]
    power_grid_sorted = nr_power_grid[sort_idx, :]
    yig_power_grid_sorted = yig_power_grid[sort_idx, :] if yig_power_grid is not None else None

    # Compute edges for pcolormesh along the detuning axis.
    n_rows = len(Delta_sorted)
    Delta_edges = np.zeros(n_rows + 1)
    if n_rows > 1:
        Delta_edges[1:-1] = (Delta_sorted[:-1] + Delta_sorted[1:]) / 2
        Delta_edges[0] = Delta_sorted[0] - (Delta_sorted[1] - Delta_sorted[0]) / 2
        Delta_edges[-1] = Delta_sorted[-1] + (Delta_sorted[-1] - Delta_sorted[-2]) / 2
    else:
        Delta_edges[0] = Delta_sorted[0] - 0.001
        Delta_edges[1] = Delta_sorted[0] + 0.001

    # Convert frequency axis to GHz.
    freqs_ghz = frequencies / 1e9
    yig_freqs_ghz = yig_freqs / 1e9 if yig_freqs is not None else None
    n_cols = len(freqs_ghz)
    freq_edges = np.zeros(n_cols + 1)
    if n_cols > 1:
        freq_edges[1:-1] = (freqs_ghz[:-1] + freqs_ghz[1:]) / 2
        freq_edges[0] = freqs_ghz[0] - (freqs_ghz[1] - freqs_ghz[0]) / 2
        freq_edges[-1] = freqs_ghz[-1] + (freqs_ghz[-1] - freqs_ghz[-2]) / 2
    else:
        freq_edges[0] = freqs_ghz[0] - 0.001
        freq_edges[1] = freqs_ghz[0] + 0.001

    # -------------------------
    # Determine target frequencies
    # -------------------------
    # avg_single_peak_frequency is assumed to be in GHz.
    target_center = avg_single_peak_frequency
    target_plus = target_center + frequency_radius  # e.g., +2 MHz = 0.002 GHz
    target_minus = target_center - frequency_radius  # e.g., -2 MHz = 0.002 GHz

    # Indices in the frequency axis closest to the target frequencies.
    idx_center = np.argmin(np.abs(freqs_ghz - target_center))
    idx_plus = np.argmin(np.abs(freqs_ghz - target_plus))
    idx_minus = np.argmin(np.abs(freqs_ghz - target_minus))

    idx_yig = np.argmin(np.abs(yig_freqs_ghz - 5.990)) if yig_freqs is not None else None

    # Define colors.
    color_center = 'red'
    color_plus = 'blue'
    color_minus = 'green'
    neon_green = "#39FF14"  # neon green for the vertical line

    # -------------------------
    # Create the figure with 2 vertical subplots.
    # -------------------------
    label_fs = 22  # Axis labels and colorbar label
    tick_fs = 20  # Tick labels
    legend_fs = 12  # Legend text

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 15))
    ax0, ax1 = axes

    # -- Top subplot: Colorplot --
    im = ax0.pcolormesh(Delta_edges, freq_edges, power_grid_sorted.T,
                        shading="auto", cmap="inferno")
    ax0.set_ylabel("Frequency (GHz)", fontsize=label_fs)
    ax0.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Draw horizontal lines with updated labels.
    ax0.axhline(target_center, color=color_center, linestyle='--',
                linewidth=2, label=f"$f_{{EP}} = {target_center:.4f}\\;$ GHz")
    ax0.axhline(target_plus, color=color_plus, linestyle='--',
                linewidth=2, label=f"$f_{{UB}} = {target_plus:.4f}\\;$ GHz")
    ax0.axhline(target_minus, color=color_minus, linestyle='--',
                linewidth=2, label=f"$f_{{LB}} = {target_minus:.4f}\\;$ GHz")

    # Add vertical line for EP Line if optimal_J is provided.
    if optimal_J is not None:
        ax0.axvline(2 * optimal_J, color=neon_green, linestyle="--",
                    linewidth=2, label="EP Line")

    # Legend with opaque frame.
    leg0 = ax0.legend(loc='best', fontsize=legend_fs, frameon=True)
    leg0.get_frame().set_alpha(1.0)

    # -- Bottom subplot: 2D Trace Plot --
    # Extract traces for the three frequency indices.
    trace_center = power_grid_sorted[:, idx_center]
    trace_plus = power_grid_sorted[:, idx_plus]
    trace_minus = power_grid_sorted[:, idx_minus]

    # Extract YIG trace
    if yig_power_grid_sorted is not None and idx_yig is not None:
        trace_yig = yig_power_grid_sorted[:, idx_yig]
        # Plot the YIG trace in gray with a label
        ax1.plot(Delta_sorted, trace_yig, color="gray", label="YIG Trace")

    # Smooth the traces.
    if smoothing_window % 2 == 0:
        smoothing_window += 1  # ensure odd window length
    smoothed_center = savgol_filter(trace_center, smoothing_window, 2)
    smoothed_plus = savgol_filter(trace_plus, smoothing_window, 2)
    smoothed_minus = savgol_filter(trace_minus, smoothing_window, 2)

    # Plot the solid (original) traces with labels.
    ax1.plot(Delta_sorted, trace_center, color=color_center,
             label=f"$S_{{21}} (f = f_{{EP}})$")
    ax1.plot(Delta_sorted, trace_plus, color=color_plus,
             label=f"$S_{{21}} (f = f_{{UB}})$")
    ax1.plot(Delta_sorted, trace_minus, color=color_minus,
             label=f"$S_{{21}} (f = f_{{LB}})$")

    # Plot the smoothed (dashed) traces without labels.
    ax1.plot(Delta_sorted, smoothed_center, color=color_center, linestyle='--')
    ax1.plot(Delta_sorted, smoothed_plus, color=color_plus, linestyle='--')
    ax1.plot(Delta_sorted, smoothed_minus, color=color_minus, linestyle='--')

    ax1.set_ylabel("Power (dBm)", fontsize=label_fs)
    ax1.set_xlabel(r'Δf [GHz]', fontsize=label_fs)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Add vertical line for EP Line if optimal_J is provided.
    if optimal_J is not None:
        ax1.axvline(2 * optimal_J, color=neon_green, linestyle="--",
                    linewidth=2, label="EP Line")

    leg1 = ax1.legend(loc='best', fontsize=legend_fs, frameon=True)
    leg1.get_frame().set_alpha(1.0)

    # -------------------------
    # Create a colorbar for the top subplot only.
    # -------------------------
    # Adjust the right margin to make room for the colorbar.
    plt.locator_params(axis='x', nbins=6)
    fig.subplots_adjust(right=0.85)
    # Get the position of the top axis.
    pos0 = ax0.get_position()
    # Create a colorbar that spans only the top half of ax0.
    cbar_x = pos0.x1 + 0.01
    cbar_y = pos0.y0 + pos0.height / 2  # start at the mid-point vertically
    cbar_width = 0.03
    cbar_height = pos0.height / 2
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Power (dBm)", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    # -------------------------
    # Final layout adjustments and save the figure.
    # -------------------------
    # from matplotlib.ticker import MaxNLocator
    # ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Limit to roughly 5 tick
    # ax0.xaxis.set_major_locator(MaxNLocator(nbins=4))  # Limit to roughly 5 tick

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(fig3_folder, exist_ok=True)
    plot_filename = f"OLD_FIG3_exp_{experiment_id}.png"
    plot_path = os.path.join(fig3_folder, plot_filename)
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"Saved FIG3 to {plot_path}")


def plot_research_figure(nr_power_grid, currents, frequencies, Delta_df, experiment_id, settings,
                         avg_single_peak_frequency, fig3_folder,
                         smoothing_window=50, optimal_J=None, yig_power_grid=None, yig_freqs=None,
                         yig_trace_freq=5.990):
    """
    Create a three-panel, publication-ready figure:

      Top panel:
         - NR colorplot (pcolormesh) of power vs. detuning with a horizontal solid red line
           (labeled "EP Readout").
         - An inset (lower right) showing the YIG colorplot with a horizontal solid blue line.
         - The top legend includes both "EP Readout" and "YIG Readout" (frequency values removed).
         - The inset now includes x- and y-axis labels with a white background for clarity.

      Middle panel:
         - Raw S₍₂₁₎ trace (solid line) vs. detuning.
         - Overplotted are the smoothed S₍₂₁₎ traces (dashed lines).
         - Both EP (red) and YIG (blue, if available) traces are plotted on a single y-axis.
         - The EP vertical line (neon-green) is added if optimal_J is provided.

      Bottom panel:
         - The derivative (absolute value) of the S₍₂₁₎ trace is computed using the
           Savitzky–Golay filter’s derivative option with a larger window to suppress noise.
         - The derivative labels include absolute value signs and are formatted as
           "|dS₍₂₁₎/dB| (f=f₍EP₎)" (and similarly for YIG).
         - The Y-axis label now includes the units (dB/Hz).

      For all panels:
         - A vertical neon-green line is drawn at Δ = 2 * optimal_J (if provided).
         - The x-axis tick locator is limited to reduce clutter.
    """

    # -------------------------
    # Prepare the colorplot data
    # -------------------------
    Delta_map = Delta_df.set_index("current")["Delta"].to_dict()
    Delta_values = np.array([Delta_map.get(c, np.nan) for c in currents])
    valid_mask = ~np.isnan(Delta_values)

    nr_power_grid = nr_power_grid[valid_mask, :]
    if yig_power_grid is not None:
        yig_power_grid = yig_power_grid[valid_mask, :]
    Delta_values = Delta_values[valid_mask]

    sort_idx = np.argsort(Delta_values)
    Delta_sorted = Delta_values[sort_idx]
    power_grid_sorted = nr_power_grid[sort_idx, :]
    if yig_power_grid is not None:
        yig_power_grid_sorted = yig_power_grid[sort_idx, :]

    # Compute edges for pcolormesh along the detuning axis.
    n_rows = len(Delta_sorted)
    Delta_edges = np.zeros(n_rows + 1)
    if n_rows > 1:
        Delta_edges[1:-1] = (Delta_sorted[:-1] + Delta_sorted[1:]) / 2
        Delta_edges[0] = Delta_sorted[0] - (Delta_sorted[1] - Delta_sorted[0]) / 2
        Delta_edges[-1] = Delta_sorted[-1] + (Delta_sorted[-1] - Delta_sorted[-2]) / 2
    else:
        Delta_edges[0] = Delta_sorted[0] - 0.001
        Delta_edges[1] = Delta_sorted[0] + 0.001

    # Convert main frequency axis to GHz.
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

    # For the inset (YIG data), compute frequency edges if available.
    if yig_freqs is not None:
        yig_freqs_ghz = yig_freqs / 1e9
        n_yig = len(yig_freqs_ghz)
        yig_freq_edges = np.zeros(n_yig + 1)
        if n_yig > 1:
            yig_freq_edges[1:-1] = (yig_freqs_ghz[:-1] + yig_freqs_ghz[1:]) / 2
            yig_freq_edges[0] = yig_freqs_ghz[0] - (yig_freqs_ghz[1] - yig_freqs_ghz[0]) / 2
            yig_freq_edges[-1] = yig_freqs_ghz[-1] + (yig_freqs_ghz[-1] - yig_freqs_ghz[-2]) / 2
        else:
            yig_freq_edges[0] = yig_freqs_ghz[0] - 0.001
            yig_freq_edges[1] = yig_freqs_ghz[0] + 0.001

    # -------------------------
    # Determine target frequencies
    # -------------------------
    target_center = avg_single_peak_frequency  # in GHz
    idx_center = np.argmin(np.abs(freqs_ghz - target_center))
    if yig_freqs is not None:
        idx_yig = np.argmin(np.abs(yig_freqs_ghz - yig_trace_freq))

    # Define colors.
    color_center = 'red'  # EP sensor
    color_inset = 'blue'  # YIG sensor
    neon_green = "#39FF14"

    # -------------------------
    # Create the figure with 3 vertical subplots.
    # -------------------------
    label_fs = 26  # Axis labels and colorbar
    tick_fs = 24  # Tick labels
    legend_fs = 20  # Legend text

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(13, 20))
    ax0, ax1, ax2 = axes

    # -- Top panel: NR Colorplot with Inset --
    im = ax0.pcolormesh(Delta_edges, freq_edges, power_grid_sorted.T,
                        shading="auto", cmap="inferno")
    ax0.set_ylabel("Frequency [GHz]", fontsize=label_fs, weight='bold')
    ax0.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Draw a solid horizontal line for EP sensor.
    ax0.axhline(target_center, color=color_center, linestyle='-', linewidth=2, label="EP Readout")
    # Add a dummy plot to include YIG in the legend.
    ax0.plot([], [], color=color_inset, linestyle='-', linewidth=2, label="YIG Readout")

    if optimal_J is not None:
        ax0.axvline(2 * optimal_J, color=neon_green, linestyle="--", linewidth=2, label="EP Line")

    leg0 = ax0.legend(loc='best', fontsize=legend_fs, frameon=True)
    leg0.get_frame().set_alpha(1.0)

    # Inset: YIG Colorplot in bottom-right of top panel.
    if (yig_power_grid is not None) and (yig_freqs is not None):
        # Increase the inset size and borderpad to provide extra space for tick labels.
        ax_inset = inset_axes(ax0, width="50%", height="50%", loc='lower right', borderpad=2)
        im_inset = ax_inset.pcolormesh(Delta_edges, yig_freq_edges, yig_power_grid_sorted.T,
                                       shading="auto", cmap="inferno", norm=im.norm)
        # Draw a solid horizontal blue line in the inset.
        ax_inset.axhline(yig_trace_freq, color=color_inset, linestyle='-', linewidth=2)
        # Remove axis labels, leave only tick labels.
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        # Adjust tick parameters: smaller font size and white color.
        ax_inset.tick_params(axis='both', which='both', labelsize=tick_fs - 8, colors="white")

    # -- Middle panel: S21 Trace and Smoothed Trace --
    trace_center = power_grid_sorted[:, idx_center]
    if (yig_power_grid is not None) and (yig_freqs is not None):
        trace_yig = yig_power_grid_sorted[:, idx_yig]

    # Standard single-axis plotting for the middle panel.
    ax1.plot(Delta_sorted, trace_center, color=color_center, label="Power (f=f$_{EP}$)", linewidth=2)
    ax1.plot(Delta_sorted, savgol_filter(trace_center, smoothing_window, 2),
             color=color_center, linestyle='--', linewidth=2, label="Power (f=f$_{EP}$) Smooth")
    if (yig_power_grid is not None) and (yig_freqs is not None):
        ax1.plot(Delta_sorted, trace_yig, color=color_inset, label="Power (f=f$_{YIG}$)", linewidth=2)
        ax1.plot(Delta_sorted, savgol_filter(trace_yig, smoothing_window, 2),
                 color=color_inset, linestyle='--', linewidth=2, label="Power (f=f$_{YIG}$) Smooth")
    ax1.set_ylabel("Power [dBm]", fontsize=label_fs, weight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=tick_fs)
    if optimal_J is not None:
        ax1.axvline(2 * optimal_J, color=neon_green, linestyle="--", linewidth=2, label="EP Line")
    leg1 = ax1.legend(loc='best', fontsize=legend_fs, frameon=True)
    leg1.get_frame().set_alpha(1.0)

    # -- Bottom panel: Derivative of S21 Trace --
    deriv_window = smoothing_window * 2 + 1  # Ensure an odd number
    delta = Delta_sorted[1] - Delta_sorted[0]  # Assume uniform spacing in detuning
    deriv_center = savgol_filter(trace_center, deriv_window, 3, deriv=1, delta=delta)
    if (yig_power_grid is not None) and (yig_freqs is not None):
        deriv_yig = savgol_filter(trace_yig, deriv_window, 3, deriv=1, delta=delta)

    ax2.plot(Delta_sorted, np.abs(deriv_center), color=color_center,
             label="|dP/d(Δf)| (f=f$_{EP}$)", linewidth=2)
    if (yig_power_grid is not None) and (yig_freqs is not None):
        ax2.plot(Delta_sorted, np.abs(deriv_yig), color=color_inset,
                 label="|dP/d(Δf)| (f=f$_{YIG}$)", linewidth=2)

    ax2.set_xlabel(r'Δf [GHz]', fontsize=label_fs, weight='bold')
    # Include slope units on the Y axis: (dB/Hz)
    ax2.set_ylabel("|dP/d(Δf)| [dBm/GHz]", fontsize=label_fs, weight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=tick_fs)
    if optimal_J is not None:
        ax2.axvline(2 * optimal_J, color=neon_green, linestyle="--", linewidth=2, label="EP Line")
    leg2 = ax2.legend(loc='best', fontsize=legend_fs, frameon=True)
    leg2.get_frame().set_alpha(1.0)

    # Limit number of x-axis ticks.
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # -------------------------
    # Create a colorbar for the top panel.
    # -------------------------
    fig.subplots_adjust(right=0.85)
    pos0 = ax0.get_position()
    cbar_x = pos0.x1 + 0.01
    cbar_y = pos0.y0 + pos0.height / 7  # Starting at roughly 1/7 of ax0 height from bottom
    cbar_width = 0.03
    cbar_height = pos0.height * 1.33
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Power [dBm]", fontsize=label_fs, weight='bold')
    cbar.ax.tick_params(labelsize=tick_fs)

    # -------------------------
    # Final layout adjustments and save the figure.
    # -------------------------
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=8))  # Adjust as needed
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(fig3_folder, exist_ok=True)
    plot_filename = f"FIG3_exp_{experiment_id}.png"
    plot_path = os.path.join(fig3_folder, plot_filename)
    plt.savefig(plot_path, dpi=400)
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
    ax.legend(loc="best")
    plt.tight_layout()

    if experiment_id is not None:
        plot_filename = f"linewidth_vs_detuning_exp_{experiment_id}.png"
    else:
        plot_filename = "linewidth_vs_detuning.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=400)
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
    ax.legend(loc="best")
    plt.tight_layout()

    # Save the plot to the output folder.
    if experiment_id is not None:
        plot_filename = f"avg_linewidth_vs_detuning_exp_{experiment_id}.png"
    else:
        plot_filename = "avg_linewidth_vs_detuning.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=400)
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
    ax.legend(loc="best")
    plt.tight_layout()

    if experiment_id is not None:
        plot_filename = f"splitting_ratio_vs_detuning_exp_{experiment_id}.png"
    else:
        plot_filename = "splitting_ratio_vs_detuning.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"Saved splitting ratio vs. detuning plot to {plot_path}")
