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
  6. Applies a double Lorentzian model to each NR trace to extract its peak centers,
     and then plots an overlay scatter plot (with error bars) of the NR peak locations
     versus Δ.

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

# Set to True to enable extra debug plots (including individual trace plots with overlayed fits)
DEBUG_MODE = True


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
    pivot_table = data.pivot_table(index="current", columns="frequency_hz", values="power_dBm", aggfunc="first")
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
        print("Fit error:", e)
        return None
    return result


def fit_trace(current_value, frequencies, power_dbm):
    freqs_ghz = frequencies / 1e9
    peaks, _ = find_peaks(power_dbm, prominence=0.1)
    if len(peaks) == 0:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_result": None, "x_fit": None}
    peak_idx = peaks[0]
    center_guess = freqs_ghz[peak_idx]
    power_linear = 10 ** (power_dbm / 10)
    widths, _, _, _ = peak_widths(power_linear, [peak_idx], rel_height=0.5)
    if len(widths) == 0:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_result": None, "x_fit": None}
    freq_step = (freqs_ghz[-1] - freqs_ghz[0]) / (len(freqs_ghz) - 1)
    fwhm_guess = widths[0] * freq_step
    fit_range = 5 * fwhm_guess
    mask = (freqs_ghz >= center_guess - fit_range) & (freqs_ghz <= center_guess + fit_range)
    if mask.sum() < 5:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_result": None, "x_fit": None}
    x_fit = freqs_ghz[mask]
    y_fit = 10 ** (power_dbm[mask] / 10)
    fit_result = single_lorentzian_fit(x_fit, y_fit, center_guess)
    if fit_result is None:
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_result": None, "x_fit": x_fit}
    center = fit_result.params["lz_center"].value
    center_unc = fit_result.params["lz_center"].stderr if fit_result.params["lz_center"].stderr is not None else np.nan
    sigma = fit_result.params["lz_sigma"].value
    sigma_unc = fit_result.params["lz_sigma"].stderr if fit_result.params["lz_sigma"].stderr is not None else np.nan
    fwhm = 2 * sigma
    fwhm_unc = 2 * sigma_unc if not np.isnan(sigma_unc) else np.nan
    return {"current": current_value, "omega": center, "omega_unc": center_unc,
            "kappa": fwhm, "kappa_unc": fwhm_unc, "fit_result": fit_result, "x_fit": x_fit}


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
# Custom Double Lorentzian Fit for NR Traces
# -----------------------------------------------------------------------------
def fit_double_lorentzian_NR(frequencies, power_dbm, center_freq_hz=None, span_freq_hz=None, peak_midpoint=None):
    """
    Fit the NR trace with a double Lorentzian model.
    Frequencies are in Hz and converted to GHz.

    Returns a dictionary with:
      - 'fit_result': the lmfit result (or None)
      - 'x_fit': the x-data (in GHz) used for fitting
      - 'guess_centers': the two initial guess centers (in GHz)
    """
    if center_freq_hz is None:
        center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    if span_freq_hz is None:
        span_freq_hz = frequencies.max() - frequencies.min()
    freqs_ghz = frequencies / 1e9
    peaks, properties = find_peaks(power_dbm, prominence=0.1)
    if len(peaks) == 0:
        return None
    prominences = properties["prominences"]
    guess_freqs = freqs_ghz[peaks]
    center_guess_ghz = center_freq_hz / 1e9
    distances = abs(guess_freqs - center_guess_ghz)
    sigma = (span_freq_hz / 1e9) / 4
    distance_weighting = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    scores = prominences * distance_weighting
    sorted_indices = np.argsort(scores)[::-1]
    if len(peaks) >= 2:
        desired_peaks = 2
    else:
        desired_peaks = 1
    top_peaks = peaks[sorted_indices[:desired_peaks]]
    if desired_peaks == 1:
        guess_centers = [freqs_ghz[top_peaks[0]], freqs_ghz[top_peaks[0]] + 0.001]
    else:
        guess_centers = [freqs_ghz[top_peaks[0]], freqs_ghz[top_peaks[1]]]
    pmin = min(guess_centers)
    pmax = max(guess_centers)
    fit_range_factor = 5
    sigma_guess_value = (span_freq_hz / 1e9) * 0.001
    left_fit = pmin - fit_range_factor * sigma_guess_value
    right_fit = pmax + fit_range_factor * sigma_guess_value
    mask = (freqs_ghz >= left_fit) & (freqs_ghz <= right_fit)
    x_fit = freqs_ghz[mask]
    y_fit_linear = 10 ** (power_dbm[mask] / 10)
    lz1 = LorentzianModel(prefix="lz1_")
    lz2 = LorentzianModel(prefix="lz2_")
    mod = lz1 + lz2
    max_height = y_fit_linear.max()
    amp_guess = max_height * np.pi * sigma_guess_value
    pars = mod.make_params()
    if peak_midpoint is not None and len(guess_centers) == 2:
        c1, c2 = sorted(guess_centers)
        pars["lz1_center"].set(value=c1, min=x_fit.min(), max=peak_midpoint)
        pars["lz2_center"].set(value=c2, min=peak_midpoint, max=x_fit.max())
    else:
        pars["lz1_center"].set(value=guess_centers[0], min=x_fit.min(), max=x_fit.max())
        pars["lz2_center"].set(value=guess_centers[1], min=x_fit.min(), max=x_fit.max())
    pars["lz1_amplitude"].set(value=amp_guess, min=0)
    pars["lz1_sigma"].set(value=sigma_guess_value, min=1e-6)
    pars["lz2_amplitude"].set(value=amp_guess, min=0)
    pars["lz2_sigma"].set(value=sigma_guess_value, min=1e-6)
    try:
        out = mod.fit(y_fit_linear, pars, x=x_fit)
    except Exception as e:
        print("Double Lorentzian fit failed:", e)
        out = None
    return {"fit_result": out, "x_fit": x_fit, "guess_centers": guess_centers}


# -----------------------------------------------------------------------------
# Plotting Functions for NR Overlay and Debug
# -----------------------------------------------------------------------------
def plot_nr_peaks_only_vs_detuning(nr_power, nr_currents, nr_freqs, delta_df, output_folder, experiment_id, nr_freq_min, nr_freq_max):
    """
    For each NR trace, perform a double Lorentzian fit and plot the resulting peak centers
    (with uncertainties) versus the detuning Δ (from delta_df). The resulting overlay plot
    uses X = Δ (GHz) and Y = Peak Frequency (GHz) and sets the Y limits based solely on the data.
    """
    delta_map = delta_df.set_index("current")["Delta"]
    detuning_list = []
    peak_list = []
    peak_unc_list = []
    for i, current in enumerate(nr_currents):
        if current not in delta_map.index:
            continue
        delta_val = delta_map.loc[current]
        trace = nr_power[i, :]
        fit_data = fit_double_lorentzian_NR(nr_freqs, trace)
        if fit_data is None or fit_data.get("fit_result") is None:
            continue
        result = fit_data["fit_result"]
        try:
            center1 = result.params["lz1_center"].value
            unc1 = result.params["lz1_center"].stderr if result.params["lz1_center"].stderr is not None else 0.0
            detuning_list.append(delta_val)
            peak_list.append(center1)
            peak_unc_list.append(unc1)
        except Exception as e:
            print("Error extracting peak 1 for current", current, ":", e)
        try:
            center2 = result.params["lz2_center"].value
            unc2 = result.params["lz2_center"].stderr if result.params["lz2_center"].stderr is not None else 0.0
            detuning_list.append(delta_val)
            peak_list.append(center2)
            peak_unc_list.append(unc2)
        except Exception as e:
            print("Error extracting peak 2 for current", current, ":", e)
    if len(peak_list) == 0:
        print("No NR peaks extracted for the overlay plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(detuning_list, peak_list, yerr=peak_unc_list, fmt="o", ecolor="red", capsize=4,
                label="NR Hybridized Peaks")
    ax.set_xlabel("Detuning Δ (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Peak Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("NR Peak Locations vs. Detuning", fontsize=LABEL_FONT_SIZE)
    # Set y-limits based solely on the data (add a small margin)
    ax.set_ylim(nr_freq_min, nr_freq_max)
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


def plot_individual_trace(current_value, frequencies, power_dbm, readout_type, folder, fit_data):
    freqs_ghz = frequencies / 1e9
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(freqs_ghz, power_dbm, "b", label="Data")
    if fit_data is not None and fit_data.get("fit_result") is not None:
        fit_result = fit_data["fit_result"]
        if fit_data.get("x_fit") is not None:
            x_fit = fit_data["x_fit"]
        else:
            x_fit = np.linspace(freqs_ghz.min(), freqs_ghz.max(), 200)
        best_fit_linear = fit_result.eval(x=x_fit)
        best_fit_db = 10 * np.log10(best_fit_linear)
        ax.plot(x_fit, best_fit_db, "m--", label="Lorentzian Fit")
        center = fit_result.params["lz_center"].value
        peak_value_linear = fit_result.eval(x=np.array([center]))
        peak_value_db = 10 * np.log10(peak_value_linear)
        ax.plot(center, peak_value_db, "r*", markersize=10, label="Peak")
        sigma = fit_result.params["lz_sigma"].value
        fwhm = 2 * sigma
        left = center - fwhm / 2
        right = center + fwhm / 2
        height = peak_value_db - 3
        ax.hlines(height, left, right, color="green", linestyle="--", label="FWHM")
        ax.annotate("Peak: " + str(center) + " GHz", (center, peak_value_db),
                    textcoords="offset points", xytext=(0, 20), ha="center", color="red")
        ax.annotate("FWHM: " + str(fwhm * 1e3) + " MHz", ((left + right) / 2, height),
                    textcoords="offset points", xytext=(0, -20), ha="center", color="green")
    ax.set_xlabel("Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Power (dBm)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(readout_type.capitalize() + " Trace at " + str(current_value) + " A", fontsize=LABEL_FONT_SIZE)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(folder, f"{readout_type}_trace_current_{current_value}.png")
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved individual", readout_type, "trace plot for current =", current_value, "A to", plot_path)


def debug_plot_individual_traces(power_grid, currents, frequencies, readout_type, folder):
    debug_folder = os.path.join(folder, "debug", readout_type)
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    for i, current in enumerate(currents):
        trace = power_grid[i, :]
        fit_data = fit_trace(current, frequencies, trace)
        plot_individual_trace(current, frequencies, trace, readout_type, debug_folder, fit_data)


def debug_plot_individual_NR_traces(power_grid, currents, frequencies, folder):
    """
    Loop over each NR trace and plot an individual debug plot that shows the raw data,
    the double Lorentzian fit, the two fitted peaks, and the FWHM lines for each peak.
    """
    debug_folder = os.path.join(folder, "debug", "nr")
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    for i, current in enumerate(currents):
        trace = power_grid[i, :]
        fit_data = fit_double_lorentzian_NR(frequencies, trace)
        freqs_ghz = frequencies / 1e9
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(freqs_ghz, trace, "b", label="NR Data")
        if fit_data is not None and fit_data.get("fit_result") is not None:
            fit_result = fit_data["fit_result"]
            if fit_data.get("x_fit") is not None:
                x_fit = fit_data["x_fit"]
            else:
                x_fit = np.linspace(freqs_ghz.min(), freqs_ghz.max(), 200)
            best_fit_linear = fit_result.eval(x=x_fit)
            best_fit_db = 10 * np.log10(best_fit_linear)
            ax.plot(x_fit, best_fit_db, "m--", label="Double Lorentzian Fit")
            try:
                center1 = fit_result.params["lz1_center"].value
                peak_val_lin1 = fit_result.eval(x=np.array([center1]))
                peak_val_db1 = 10 * np.log10(peak_val_lin1)
                ax.plot(center1, peak_val_db1, "r*", markersize=10, label="Peak 1")
                sigma1 = fit_result.params["lz1_sigma"].value
                fwhm1 = 2 * sigma1
                left1 = center1 - fwhm1 / 2
                right1 = center1 + fwhm1 / 2
                height1 = peak_val_db1 - 3
                # ax.hlines(height1, left1, right1, color="green", linestyle="--", label="FWHM 1")
            except Exception as e:
                print("Error extracting NR peak 1 for current", current, ":", e)
            try:
                center2 = fit_result.params["lz2_center"].value
                peak_val_lin2 = fit_result.eval(x=np.array([center2]))
                peak_val_db2 = 10 * np.log10(peak_val_lin2)
                ax.plot(center2, peak_val_db2, "y*", markersize=10, label="Peak 2")
                sigma2 = fit_result.params["lz2_sigma"].value
                fwhm2 = 2 * sigma2
                left2 = center2 - fwhm2 / 2
                right2 = center2 + fwhm2 / 2
                height2 = peak_val_db2 - 3
                # ax.hlines(height2, left2, right2, color="green", linestyle="--", label="FWHM 2")
            except Exception as e:
                print("Error extracting NR peak 2 for current", current, ":", e)
        ax.set_xlabel("Frequency (GHz)", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Power (dBm)", fontsize=LABEL_FONT_SIZE)
        ax.set_title("NR Trace at " + str(current) + " A", fontsize=LABEL_FONT_SIZE)
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(debug_folder, f"nr_trace_current_{current}.png")
        plt.savefig(plot_path, dpi=SAVE_DPI)
        plt.close(fig)
        print("Saved debug NR trace with double fit for current =", current, "A to", plot_path)


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

    if DEBUG_MODE:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        print("DEBUG MODE enabled: plotting individual cavity traces with Lorentzian fits...")
        debug_plot_individual_traces(cavity_power, cavity_currents, cavity_freqs, "cavity", debug_folder)
        print("DEBUG MODE enabled: plotting individual YIG traces with Lorentzian fits...")
        debug_plot_individual_traces(yig_power, yig_currents, yig_freqs, "yig", debug_folder)
        print("DEBUG MODE enabled: plotting individual NR traces with double Lorentzian fits...")
        debug_plot_individual_NR_traces(nr_power, nr_currents, nr_freqs, debug_folder)

    print("Fitting cavity traces...")
    cavity_df = process_all_traces(cavity_power, cavity_currents, cavity_freqs)
    print("Fitting YIG traces...")
    yig_df = process_all_traces(yig_power, yig_currents, yig_freqs)
    delta_df = compute_delta(cavity_df, yig_df)
    print("Computed detuning for each current.")

    output_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP")
    plot_delta_colorplot(nr_power, nr_currents, nr_freqs, delta_df, experiment_id, nr_settings, output_folder)

    overlay_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_nr_peaks")
    if not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder)
    print("Plotting NR peak locations vs. Detuning overlay...")
    plot_nr_peaks_only_vs_detuning(nr_power, nr_currents, nr_freqs, delta_df, overlay_folder, experiment_id,
                                   colorplot_freq_min, colorplot_freq_max)


if __name__ == "__main__":
    main()
