#!/usr/bin/env python3
"""
NR EP Analysis Script

For each current (set_amperage) the database contains three traces:
  • a cavity-only trace,
  • a YIG-only trace, and
  • a hybridized NR trace.

This script:
  1. Loads the data for the three readouts.
  2. Fits the cavity and YIG traces (using a single Lorentzian) to extract
     the mode center (ω, in GHz) and linewidth (κ, in GHz).
  3. Merges these fits to compute the detuning Δ = ω₍c₎ − ω₍y₎ (and its uncertainty)
     for each current.
  4. Uses the NR hybridized data and re-labels the X axis from current to Δ.
  5. Produces a color plot where X = Δ and Y = frequency.
  6. For each NR trace, uses a basic SciPy peak‐finding routine to get initial guesses.
     • If only one peak is found, a single Lorentzian fit is performed and then a
       double Lorentzian fit is also attempted (using that same guess for both peaks).
       The two fits are compared via their reduced chi‐square and the better one is used.
     • If two (or more) peaks are found, a double Lorentzian fit is performed
       (using the two most prominent peaks as the initial guesses).
       (No FWHM lines are plotted for the double Lorentzian fits.)
  7. An overlay scatter plot is produced showing the NR peak locations versus Δ.

No scattershot filtering is applied.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from lmfit.models import LorentzianModel
from sqlalchemy import create_engine

# -----------------------------------------------------------------------------
# Constants for plotting and file saving
# -----------------------------------------------------------------------------
PLOTS_FOLDER = "plots"
SAVE_DPI = 300
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12

CAV_YIG_DEBUG = True
NR_DEBUG = True


# -----------------------------------------------------------------------------
# Database and Data Access Functions
# -----------------------------------------------------------------------------
def get_engine(db_path):
    return create_engine(f"sqlite:///{db_path}")


def get_data_from_db(engine, experiment_id, readout_type, freq_min, freq_max, current_min, current_max,
                     independent_var="set_amperage"):
    settings_query = f"""
    SELECT DISTINCT set_loop_att, set_loopback_att,
                    set_yig_fb_phase_deg, set_yig_fb_att,
                    set_cavity_fb_phase_deg, set_cavity_fb_att
    FROM expr
    WHERE experiment_id = '{experiment_id}' AND readout_type = '{readout_type}'
    """
    settings_df = pd.read_sql_query(settings_query, engine)
    settings = settings_df.iloc[0].to_dict() if not settings_df.empty else {}

    data_query = f"""
    SELECT frequency_hz, {independent_var} as current, power_dBm
    FROM expr
    WHERE experiment_id = '{experiment_id}'
      AND readout_type = '{readout_type}'
      AND {independent_var} BETWEEN {current_min} AND {current_max}
      AND frequency_hz BETWEEN {freq_min} AND {freq_max}
    ORDER BY current, frequency_hz
    """
    data = pd.read_sql_query(data_query, engine)
    if data.empty:
        return None, None, None, None
    pivot_table = data.pivot_table(index="current", columns="frequency_hz",
                                   values="power_dBm", aggfunc="first")
    currents = pivot_table.index.values
    frequencies = pivot_table.columns.values
    power_grid = pivot_table.values
    return power_grid, currents, frequencies, settings


# -----------------------------------------------------------------------------
# Fitting Functions for Single Lorentzian (for cavity and YIG)
# -----------------------------------------------------------------------------
def single_lorentzian_fit(x, y, center_guess):
    model = LorentzianModel(prefix="lz_")
    pars = model.make_params()
    sigma_guess = 0.001
    amp_guess = y.max() * np.pi * sigma_guess
    pars["lz_center"].set(value=center_guess, min=x.min(), max=x.max())
    pars["lz_amplitude"].set(value=amp_guess, min=0)
    pars["lz_sigma"].set(value=sigma_guess, min=1e-6)
    try:
        result = model.fit(y, pars, x=x)
    except Exception as e:
        print("Single Lorentzian fit error:", e)
        return None
    return result


def fit_trace(current_value, frequencies, power_dbm):
    # For cavity and YIG traces
    freqs_ghz = frequencies / 1e9
    peaks, _ = find_peaks(power_dbm, prominence=0.1)
    if len(peaks) == 0:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": None}
    peak_idx = peaks[0]
    center_guess = freqs_ghz[peak_idx]
    power_linear = 10 ** (power_dbm / 10)
    widths, _, _, _ = peak_widths(power_linear, [peak_idx], rel_height=0.5)
    if len(widths) == 0:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": None}
    freq_step = (freqs_ghz[-1] - freqs_ghz[0]) / (len(freqs_ghz) - 1)
    fwhm_guess = widths[0] * freq_step
    fit_range = 5 * fwhm_guess
    mask = (freqs_ghz >= center_guess - fit_range) & (freqs_ghz <= center_guess + fit_range)
    if mask.sum() < 5:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": x_fit}
    x_fit = freqs_ghz[mask]
    y_fit = 10 ** (power_dbm[mask] / 10)
    fit_result = single_lorentzian_fit(x_fit, y_fit, center_guess)
    if fit_result is None:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": x_fit}
    center = fit_result.params["lz_center"].value
    center_unc = fit_result.params["lz_center"].stderr if fit_result.params["lz_center"].stderr is not None else np.nan
    sigma = fit_result.params["lz_sigma"].value
    sigma_unc = fit_result.params["lz_sigma"].stderr if fit_result.params["lz_sigma"].stderr is not None else np.nan
    fwhm = 2 * sigma
    fwhm_unc = 2 * sigma_unc if not np.isnan(sigma_unc) else np.nan
    return {"current": current_value, "omega": center, "omega_unc": center_unc,
            "kappa": fwhm, "kappa_unc": fwhm_unc, "fit_type": "single", "fit_result": fit_result, "x_fit": x_fit}


def process_all_traces(power_grid, currents, frequencies):
    results = []
    for i, current in enumerate(currents):
        power_dbm = power_grid[i, :]
        res = fit_trace(current, frequencies, power_dbm)
        results.append(res)
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Compute Detuning from Cavity and YIG Fits
# -----------------------------------------------------------------------------
def compute_delta(cavity_df, yig_df):
    cavity_df = cavity_df.rename(columns={"omega": "omega_c", "omega_unc": "omega_c_unc",
                                          "kappa": "kappa_c", "kappa_unc": "kappa_c_unc"})
    yig_df = yig_df.rename(columns={"omega": "omega_y", "omega_unc": "omega_y_unc",
                                    "kappa": "kappa_y", "kappa_unc": "kappa_y_unc"})
    merged = pd.merge(cavity_df, yig_df, on="current", how="inner")
    merged["Delta"] = merged["omega_c"] - merged["omega_y"]
    merged["Delta_unc"] = np.sqrt(merged["omega_c_unc"] ** 2 + merged["omega_y_unc"] ** 2)
    return merged


# -----------------------------------------------------------------------------
# New NR Fitting Function using basic SciPy peak finding and dual fitting
# -----------------------------------------------------------------------------
def double_lorentzian_fit_NR(x, y, guess1, guess2):
    from lmfit.models import LorentzianModel
    lz1 = LorentzianModel(prefix="lz1_")
    lz2 = LorentzianModel(prefix="lz2_")
    mod = lz1 + lz2
    sigma_guess = 0.001
    amp_guess = y.max() * np.pi * sigma_guess
    pars = mod.make_params()
    # Ensure the first guess is less than the second.
    if guess1 > guess2:
        guess1, guess2 = guess2, guess1
    pars["lz1_center"].set(value=guess1, min=x.min(), max=guess2)
    pars["lz2_center"].set(value=guess2, min=guess2, max=x.max())
    pars["lz1_amplitude"].set(value=amp_guess, min=0)
    pars["lz1_sigma"].set(value=sigma_guess, min=1e-6)
    pars["lz2_amplitude"].set(value=amp_guess, min=0)
    pars["lz2_sigma"].set(value=sigma_guess, min=1e-6)
    try:
        out = mod.fit(y, pars, x=x)
    except Exception as e:
        print("Double Lorentzian fit failed:", e)
        return None
    return out


def fit_NR_trace(current_value, frequencies, power_dbm):
    """
    For an NR trace, use basic SciPy.find_peaks with specified parameters.
      - If only one peak is found, perform a single Lorentzian fit.
        ALSO perform a double Lorentzian fit using the same guess for both peaks.
        Compare the fits (via redchi) and return the one with the lower value.
      - If two or more peaks are found, take the two most prominent peaks and perform a double Lorentzian fit.
    Returns a dictionary with:
      - "current": current value
      - "fit_type": "single" or "double"
      - For single fit: "omega" and "omega_unc"
      - For double fit: "peak1", "peak1_unc", "peak2", "peak2_unc"
      - "fit_result": the chosen lmfit result (or None)
      - "x_fit": the x-data (in GHz) used for fitting.
      - Also returns the initial guesses as "peak1_guess" and "peak2_guess".
    """
    freqs_ghz = frequencies / 1e9
    # Use basic peak-finding parameters:
    peaks_indices, props = find_peaks(power_dbm, height=-30, prominence=0.02, distance=15)
    if len(peaks_indices) == 0:
        return {"current": current_value, "fit_type": None, "fit_result": None, "x_fit": None}
    elif len(peaks_indices) == 1:
        # Compute single fit:
        center_guess = freqs_ghz[peaks_indices[0]]
        power_linear = 10 ** (power_dbm / 10)
        single_fit = single_lorentzian_fit(freqs_ghz, power_linear, center_guess)
        # ALSO attempt a double fit using the same guess for both peaks:
        double_fit = double_lorentzian_fit_NR(freqs_ghz, power_linear, center_guess, center_guess + 0.001)
        # Compare the reduced chi-square (redchi) if both fits succeeded.
        # If double_fit is None, use single_fit.
        if double_fit is None or single_fit is None:
            chosen = single_fit if single_fit is not None else double_fit
            chosen_type = "single" if single_fit is not None else "double"
        else:
            # Use redchi to compare fits:
            if hasattr(double_fit, "redchi") and hasattr(single_fit, "redchi"):
                if double_fit.redchi < single_fit.redchi:
                    chosen = double_fit
                    chosen_type = "double"
                else:
                    chosen = single_fit
                    chosen_type = "single"
            else:
                # Fall back to single if redchi is not available
                chosen = single_fit
                chosen_type = "single"
        if chosen_type == "single":
            center = chosen.params["lz_center"].value
            center_unc = chosen.params["lz_center"].stderr if chosen.params["lz_center"].stderr is not None else np.nan
            return {"current": current_value, "fit_type": "single", "omega": center, "omega_unc": center_unc,
                    "fit_result": chosen, "x_fit": freqs_ghz, "peak1_guess": center_guess, "peak2_guess": None}
        else:
            peak1 = chosen.params["lz1_center"].value
            peak1_unc = chosen.params["lz1_center"].stderr if chosen.params["lz1_center"].stderr is not None else np.nan
            peak2 = chosen.params["lz2_center"].value
            peak2_unc = chosen.params["lz2_center"].stderr if chosen.params["lz2_center"].stderr is not None else np.nan
            return {"current": current_value, "fit_type": "double",
                    "peak1": peak1, "peak1_unc": peak1_unc,
                    "peak2": peak2, "peak2_unc": peak2_unc,
                    "fit_result": chosen, "x_fit": freqs_ghz,
                    "peak1_guess": center_guess, "peak2_guess": center_guess + 0.001}
    else:
        # If two or more peaks are found, take the two with highest prominence.
        prominences = props["prominences"]
        sorted_idx = np.argsort(prominences)[::-1]
        best_two = peaks_indices[sorted_idx[:2]]
        best_two = np.sort(best_two)
        guess1 = freqs_ghz[best_two[0]]
        guess2 = freqs_ghz[best_two[1]]
        power_linear = 10 ** (power_dbm / 10)
        fit_result = double_lorentzian_fit_NR(freqs_ghz, power_linear, guess1, guess2)
        if fit_result is None:
            return {"current": current_value, "fit_type": "single", "omega": guess1, "omega_unc": np.nan,
                    "fit_result": None, "x_fit": freqs_ghz}
        peak1 = fit_result.params["lz1_center"].value
        peak1_unc = fit_result.params["lz1_center"].stderr if fit_result.params[
                                                                  "lz1_center"].stderr is not None else np.nan
        peak2 = fit_result.params["lz2_center"].value
        peak2_unc = fit_result.params["lz2_center"].stderr if fit_result.params[
                                                                  "lz2_center"].stderr is not None else np.nan
        return {"current": current_value, "fit_type": "double",
                "peak1": peak1, "peak1_unc": peak1_unc,
                "peak2": peak2, "peak2_unc": peak2_unc,
                "fit_result": fit_result, "x_fit": freqs_ghz,
                "peak1_guess": guess1, "peak2_guess": guess2}


# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def plot_individual_trace(current_value, frequencies, power_dbm, readout_type, folder, fit_data, detuning_val=None,
                          order_prefix=""):
    """
    Plot an individual trace. For NR traces, if detuning_val is provided,
    include it in the title and filename. The order_prefix (if provided) is
    prepended to the file name for proper alphabetical sorting.
    """
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
            ax.annotate("Peak: " + str(center) + " GHz", (center, peak_val_db),
                        textcoords="offset points", xytext=(0, 20), ha="center", color="red")
        elif fit_data.get("fit_type") == "double":
            try:
                center1 = fit_result.params["lz1_center"].value
                peak_val_db1 = 10 * np.log10(fit_result.eval(x=np.array([center1])))
                ax.plot(center1, peak_val_db1, "r*", markersize=10, label="Peak 1")
                ax.annotate("Peak 1: " + str(center1) + " GHz", (center1, peak_val_db1),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="red")
            except Exception as e:
                print("Error in double fit (peak1):", e)
            try:
                center2 = fit_result.params["lz2_center"].value
                peak_val_db2 = 10 * np.log10(fit_result.eval(x=np.array([center2])))
                ax.plot(center2, peak_val_db2, "y*", markersize=10, label="Peak 2")
                ax.annotate("Peak 2: " + str(center2) + " GHz", (center2, peak_val_db2),
                            textcoords="offset points", xytext=(0, 20), ha="center", color="orange")
            except Exception as e:
                print("Error in double fit (peak2):", e)
    title = readout_type.capitalize() + " Trace at " + str(current_value) + " A"
    file_suffix = f"{current_value}"
    if readout_type.lower() == "nr" and detuning_val is not None:
        title += ", Detuning = " + str(detuning_val) + " GHz"
        file_suffix += "_Delta_" + str(detuning_val)
    ax.set_xlabel("Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Power (dBm)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=LABEL_FONT_SIZE)
    ax.legend()
    plt.tight_layout()
    # Prepend order_prefix (if provided) to the file name so that Windows sorts the files in order.
    plot_filename = f"{order_prefix}_{readout_type}_trace_current_{file_suffix}.png"
    plot_path = os.path.join(folder, plot_filename)
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved individual", readout_type, "trace plot for current =", current_value, "A to", plot_path)


def debug_plot_individual_traces(power_grid, currents, frequencies, readout_type, folder, detuning_data=None):
    """
    For each trace, if detuning_data is provided (for NR), look up the corresponding detuning value.
    For NR traces, the traces are sorted by detuning (low to high) and an order index (zero-padded)
    is prepended to the filename so that Windows sorts the files in the desired order.
    """
    debug_folder = os.path.join(folder, "debug", readout_type)
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    if readout_type.lower() == "nr" and detuning_data is not None:
        # Build a list of (index, current, delta) for all NR traces with valid detuning
        sorted_list = []
        for i, cur in enumerate(currents):
            try:
                delta_val = float(detuning_data.loc[detuning_data["current"] == cur, "Delta"].iloc[0])
                sorted_list.append((i, cur, delta_val))
            except Exception as e:
                print("Could not extract detuning for current", cur, ":", e)
        # Sort the list by delta value (low to high)
        sorted_list.sort(key=lambda x: x[2])
        # Loop over the sorted list and assign a zero-padded order prefix for each file name.
        for order, (i, cur, delta_val) in enumerate(sorted_list, start=1):
            trace = power_grid[i, :]
            fit_data = fit_NR_trace(cur, frequencies, trace)
            # Use the extracted delta_val for the plot title as well.
            order_prefix = f"{order:03d}_"
            plot_individual_trace(cur, frequencies, trace, readout_type, debug_folder,
                                  fit_data, detuning_val=delta_val, order_prefix=order_prefix)
    else:
        # For non-NR traces (or if no detuning data is provided) use the original order.
        for i, current in enumerate(currents):
            trace = power_grid[i, :]
            if readout_type in ["cavity", "yig"]:
                fit_data = fit_trace(current, frequencies, trace)
                extra = None
            else:
                fit_data = fit_NR_trace(current, frequencies, trace)
                extra = None
                if detuning_data is not None:
                    try:
                        extra = float(detuning_data.loc[detuning_data["current"] == current, "Delta"].iloc[0])
                    except Exception as e:
                        print("Could not extract detuning for current", current, ":", e)
            plot_individual_trace(current, frequencies, trace, readout_type, debug_folder,
                                  fit_data, detuning_val=extra)



def plot_nr_peaks_only_vs_detuning(nr_power, nr_currents, nr_freqs, delta_df, output_folder, experiment_id):
    """
    For each NR trace, use fit_NR_trace to extract peak locations (one or two)
    and then plot these versus the detuning Δ (from delta_df). X = Δ (GHz) and Y = Peak Frequency (GHz)
    """
    delta_map = delta_df.set_index("current")["Delta"]
    detuning_list = []
    peak_list = []
    peak_unc_list = []
    for i, current in enumerate(nr_currents):
        if current not in delta_map.index:
            continue
        delta_val = delta_map.loc[current]
        fit_data = fit_NR_trace(current, nr_freqs, nr_power[i, :])
        if fit_data is None:
            continue
        if fit_data.get("fit_type") == "single":
            peak = fit_data.get("omega")
            unc = fit_data.get("omega_unc", 0.0)
            detuning_list.append(delta_val)
            peak_list.append(peak)
            peak_unc_list.append(unc)
        elif fit_data.get("fit_type") == "double":
            peak1 = fit_data.get("peak1")
            unc1 = fit_data.get("peak1_unc", 0.0)
            peak2 = fit_data.get("peak2")
            unc2 = fit_data.get("peak2_unc", 0.0)
            detuning_list.extend([delta_val, delta_val])
            peak_list.extend([peak1, peak2])
            peak_unc_list.extend([unc1, unc2])
    if len(peak_list) == 0:
        print("No NR peaks extracted for the overlay plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(detuning_list, peak_list, yerr=peak_unc_list, fmt="o", ecolor="red", capsize=4,
                label="NR Hybridized Peaks")
    ax.set_xlabel("Detuning Δ (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("NR Peak Locations vs. Detuning", fontsize=LABEL_FONT_SIZE)
    y_min = min(peak_list)
    y_max = max(peak_list)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_path = os.path.join(output_folder, f"nr_peaks_overlay_exp_{experiment_id}.png")
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved NR peaks overlay plot to", plot_path)


def plot_delta_colorplot(nr_power_grid, currents, frequencies, delta_df, experiment_id, settings, folder):
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
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(delta_edges, freq_edges, power_grid_sorted.T, shading="auto", cmap="inferno")
    ax.set_xlabel("Detuning Δ (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
    title = "Experiment " + experiment_id + " NR Hybridized Data\n"
    title += "Loop Att: " + str(settings.get("set_loop_att", "N/A")) + " dB, YIG FB Phase: " + str(
        settings.get("set_yig_fb_phase_deg", "N/A")) + "°"
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(c, ax=ax, label="Power (dBm)")
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    plt.tight_layout()
    plot_path = os.path.join(folder, f"NR_colorplot_detuning_exp_{experiment_id}.png")
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved NR color plot (detuning) to", plot_path)


def plot_raw_colorplot(power_grid, currents, frequencies, experiment_id, settings, folder, readout_type="nr"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(currents, frequencies / 1e9, power_grid.T, shading="auto", cmap="inferno")
    ax.set_xlabel("Current (A)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
    title = "Experiment " + experiment_id + " " + readout_type.upper() + " Raw Data\n"
    if settings:
        title += "Loop Att: " + str(settings.get("set_loop_att", "N/A")) + " dB"
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(c, ax=ax, label="Power (dBm)")
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    plt.tight_layout()
    plot_path = os.path.join(folder, f"{readout_type}_raw_colorplot_exp_{experiment_id}.png")
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved raw", readout_type.upper(), "color plot (current as X-axis) to", plot_path)


# -----------------------------------------------------------------------------
# Main Routine
# -----------------------------------------------------------------------------
def main():
    db_path = "../databases/NR_SUN_NITE_PPP_FINE.db"
    experiment_id = "294962de-dd80-49c6-81b5-394ae97b5838"
    # Frequency limits (Hz)
    colorplot_freq_min = 5.994e9
    colorplot_freq_max = 6.002e9
    cavity_freq_min = 5.995e9
    cavity_freq_max = 6.02e9
    yig_freq_min = 5.981e9
    yig_freq_max = 5.998e9
    current_min = -99
    current_max = 99

    engine = get_engine(db_path)
    print("Fetching cavity data...")
    cavity_power, cavity_currents, cavity_freqs, cavity_settings = get_data_from_db(
        engine, experiment_id, "cavity", cavity_freq_min, cavity_freq_max, current_min, current_max,
        independent_var="set_amperage"
    )
    print("Fetching YIG data...")
    yig_power, yig_currents, yig_freqs, yig_settings = get_data_from_db(
        engine, experiment_id, "yig", yig_freq_min, yig_freq_max, current_min, current_max,
        independent_var="set_amperage"
    )
    print("Fetching NR (hybridized) data...")
    nr_power, nr_currents, nr_freqs, nr_settings = get_data_from_db(
        engine, experiment_id, "nr", colorplot_freq_min, colorplot_freq_max, current_min, current_max,
        independent_var="set_amperage"
    )
    if cavity_power is None or yig_power is None or nr_power is None:
        print("Error: One or more data sets are missing. Exiting.")
        return

    raw_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP_raw")
    plot_raw_colorplot(nr_power, nr_currents, nr_freqs, experiment_id, nr_settings, raw_folder, readout_type="nr")

    if CAV_YIG_DEBUG:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        print("CAV_YIG_DEBUG MODE enabled: plotting individual cavity traces with Lorentzian fits...")
        debug_plot_individual_traces(cavity_power, cavity_currents, cavity_freqs, "cavity", debug_folder)
        print("CAV_YIG_DEBUG MODE enabled: plotting individual YIG traces with Lorentzian fits...")
        debug_plot_individual_traces(yig_power, yig_currents, yig_freqs, "yig", debug_folder)

    print("Fitting cavity traces...")
    cavity_df = process_all_traces(cavity_power, cavity_currents, cavity_freqs)
    print("Fitting YIG traces...")
    yig_df = process_all_traces(yig_power, yig_currents, yig_freqs)
    delta_df = compute_delta(cavity_df, yig_df)
    print("Computed detuning for each current.")

    if NR_DEBUG:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        print(
            "NR_DEBUG MODE enabled: plotting individual NR traces with new NR fitting (including detuning in title)...")
        debug_plot_individual_traces(nr_power, nr_currents, nr_freqs, "nr", debug_folder, detuning_data=delta_df)

    output_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP")
    plot_delta_colorplot(nr_power, nr_currents, nr_freqs, delta_df, experiment_id, nr_settings, output_folder)

    overlay_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_nr_peaks")
    if not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder)
    print("Plotting NR peak locations vs. Detuning overlay...")
    plot_nr_peaks_only_vs_detuning(nr_power, nr_currents, nr_freqs, delta_df, overlay_folder, experiment_id)


if __name__ == "__main__":
    main()
