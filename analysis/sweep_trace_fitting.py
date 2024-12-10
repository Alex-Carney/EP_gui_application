import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from lmfit.models import LorentzianModel
import pandas as pd
from sqlalchemy import create_engine

PLOTS_FOLDER = 'plots'
TABLE_NAME = 'expr'
LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 15
SAVE_DPI = 400


def find_and_fit_peaks(frequencies, power_dbm, center_freq_hz=None, span_freq_hz=None, num_peaks_to_find=1):
    """
    Find peaks using scipy, score them by prominence and distance from center,
    and fit the top peak with a Lorentzian.
    Returns a list of dict with peak info including Lorentzian fit if successful.
    """
    peaks_info = []

    if center_freq_hz is None:
        center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    if span_freq_hz is None:
        span_freq_hz = frequencies.max() - frequencies.min()

    # Convert dB to linear for initial analysis
    power_linear = 10 ** (power_dbm / 10)

    prominence = 0.1
    peaks, properties = find_peaks(power_dbm, prominence=prominence)
    if len(peaks) == 0:
        return peaks_info

    prominences = properties["prominences"]
    peak_freqs_hz = frequencies[peaks]

    # Distance weighting
    distances = abs(peak_freqs_hz - center_freq_hz)
    sigma = span_freq_hz / 4
    distance_weighting = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    scores = prominences * distance_weighting
    sorted_indices = np.argsort(scores)[::-1]
    num_peaks = min(num_peaks_to_find, len(peaks))
    top_peaks = peaks[sorted_indices[:num_peaks]]

    # Get FWHM from peak_widths
    results_half = peak_widths(power_linear, top_peaks, rel_height=0.5)
    widths = results_half[0]
    left_ips = results_half[2]
    right_ips = results_half[3]

    for i, peak_idx in enumerate(top_peaks):
        peak_freq_hz = frequencies[peak_idx]
        peak_freq_ghz = peak_freq_hz / 1e9
        peak_power_db = power_dbm[peak_idx]

        # Compute fwhm in Hz
        freq_step_hz = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
        fwhm_samples = widths[i]
        fwhm_hz = fwhm_samples * freq_step_hz
        fwhm_ghz = fwhm_hz / 1e9

        # Fit range
        fit_range_factor = 5
        left_fit_freq_ghz = peak_freq_ghz - fit_range_factor * fwhm_ghz
        right_fit_freq_ghz = peak_freq_ghz + fit_range_factor * fwhm_ghz

        fit_mask = (frequencies / 1e9 >= left_fit_freq_ghz) & (frequencies / 1e9 <= right_fit_freq_ghz)
        x_fit_ghz = frequencies[fit_mask] / 1e9
        y_fit_linear = power_linear[fit_mask]

        if len(x_fit_ghz) < 5:
            # Not enough points to fit
            peaks_info.append({
                'peak_freq_ghz': peak_freq_ghz,
                'fwhm_ghz': fwhm_ghz,
                'fit_result': None,
                'x_fit_ghz': x_fit_ghz,
                'y_fit_linear': y_fit_linear
            })
            continue

        # Lorentzian fit initial guesses
        max_height = y_fit_linear.max()
        sigma_guess = fwhm_ghz / 2 if fwhm_ghz > 0 else 0.001
        amp_guess = max_height * np.pi * sigma_guess

        lz = LorentzianModel(prefix='lz_')
        pars = lz.make_params()
        pars['lz_center'].set(value=peak_freq_ghz, min=x_fit_ghz.min(), max=x_fit_ghz.max())
        pars['lz_amplitude'].set(value=amp_guess, min=0)
        pars['lz_sigma'].set(value=sigma_guess, min=1e-6)

        try:
            out = lz.fit(y_fit_linear, pars, x=x_fit_ghz)
        except Exception as e:
            out = None

        peaks_info.append({
            'peak_freq_ghz': peak_freq_ghz,
            'fwhm_ghz': fwhm_ghz,
            'fit_result': out,
            'x_fit_ghz': x_fit_ghz,
            'y_fit_linear': y_fit_linear
        })

    return peaks_info


def fit_and_plot_single_trace(voltage, frequencies, power_dbm, output_folder):
    """
    Fit the single trace with Lorentzian and plot/save the figure.
    """
    # Compute center and span from data
    center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    span_freq_hz = frequencies.max() - frequencies.min()

    peaks_info = find_and_fit_peaks(frequencies, power_dbm, center_freq_hz=center_freq_hz, span_freq_hz=span_freq_hz,
                                    num_peaks_to_find=1)

    freqs_in_ghz = frequencies / 1e9

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs_in_ghz, power_dbm, 'b', label='Data')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm)')
    ax.set_title(f'Voltage = {voltage:.3f} V')

    if len(peaks_info) > 0:
        peak = peaks_info[0]
        peak_freq_ghz = peak['peak_freq_ghz']
        ax.plot(peak_freq_ghz, power_dbm[np.argmin(abs(freqs_in_ghz - peak_freq_ghz))], 'r*', markersize=10,
                label='Detected Peak')

        if peak['fit_result'] is not None:
            out = peak['fit_result']
            x_fit_ghz = peak['x_fit_ghz']
            y_fit_linear = out.best_fit
            y_fit_db = 10 * np.log10(y_fit_linear)
            ax.plot(x_fit_ghz, y_fit_db, 'm--', label='Lorentzian Fit')

    ax.legend()
    plt.tight_layout()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, f"trace_V_{voltage:.3f}V.png")
    plt.savefig(file_path, dpi=300)
    plt.close(fig)
    print(f"Saved trace plot with fit for voltage={voltage:.3f} V to {file_path}")


def __process_all_traces(power_grid, voltages, frequencies, peak_finding_function, experiment_id, readout_type):
    # Here we integrate the new fitting logic
    # Instead of just returning a DF, we also save individual trace plots.
    output_folder = os.path.join("traces_plots", f"{experiment_id}_{readout_type}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    voltage_list, peak_freqs_list, peak_powers_list = [], [], []
    for idx, voltage in enumerate(voltages):
        powers = power_grid[idx, :]
        # We can call fit_and_plot_single_trace for each
        fit_and_plot_single_trace(voltage, frequencies, powers, output_folder)

        # If you still want peak info in a DF, you can also call the simpler peak_finding_function:
        peak_freqs, peak_powers = peak_finding_function(frequencies, powers)
        voltage_list.extend([voltage] * len(peak_freqs))
        peak_freqs_list.extend(peak_freqs)
        peak_powers_list.extend(peak_powers)
    return pd.DataFrame({'voltage': voltage_list, 'peak_freq': peak_freqs_list, 'peak_power': peak_powers_list})


def __get_engine(db_path):
    return create_engine(f'sqlite:///{db_path}')


def __get_data_from_db(engine, experiment_id, readout_type, freq_min=1e9, freq_max=99e9, voltage_min=-2.0,
                       voltage_max=2.0):
    # Fetch settings for the experiment
    settings_query = f"""
    SELECT DISTINCT set_loop_phase_deg, set_loop_att, set_loopback_att,
                    set_yig_fb_phase_deg, set_yig_fb_att,
                    set_cavity_fb_phase_deg, set_cavity_fb_att
    FROM {TABLE_NAME}
    WHERE experiment_id = '{experiment_id}' AND readout_type = '{readout_type}'
    """
    settings_df = pd.read_sql_query(settings_query, engine)

    # Check if data is empty and handle accordingly
    if settings_df.empty:
        print(f"No settings found for experiment {experiment_id}, readout type: {readout_type}. Using default values.")
        settings = {
            "set_loop_phase_deg": "N/A",
            "set_loop_att": "N/A",
            "set_loopback_att": "N/A",
            "set_yig_fb_phase_deg": "N/A",
            "set_yig_fb_att": "N/A",
            "set_cavity_fb_phase_deg": "N/A",
            "set_cavity_fb_att": "N/A"
        }
    else:
        settings = settings_df.iloc[0]

    # Fetch measurement data
    data_query = f"""
    SELECT frequency_hz, set_voltage, power_dBm FROM {TABLE_NAME}
    WHERE experiment_id = '{experiment_id}'
    AND readout_type = '{readout_type}'
    AND set_voltage BETWEEN {voltage_min} AND {voltage_max}
    AND frequency_hz BETWEEN {freq_min} AND {freq_max}
    ORDER BY set_voltage, frequency_hz
    """
    data = pd.read_sql_query(data_query, engine)

    if not data.empty:
        pivot_table = data.pivot_table(index='set_voltage', columns='frequency_hz', values='power_dBm', aggfunc='first')
        voltages = pivot_table.index.values
        frequencies = pivot_table.columns.values
        power_grid = pivot_table.values
        return power_grid, voltages, frequencies, settings
    else:
        print(
            f"No data found for experiment {experiment_id}, readout type: {readout_type}, with voltage range: {voltage_min} to {voltage_max}")
        return None, None, None, None


def __default_peak_finding_function(frequencies, powers):
    # Detect peaks using SciPy
    peaks_indices, _ = find_peaks(powers, height=-30, prominence=0.1, distance=50)
    peak_freqs = frequencies[peaks_indices]
    peak_powers = powers[peaks_indices]
    return peak_freqs, peak_powers


def __generate_transmission_plot_with_peaks(power_grid, voltages, frequencies, peaks_df, experiment_id, readout_type,
                                            settings,
                                            folder, vmin=-40, vmax=8):
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create the 2D transmission plot
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(voltages, frequencies / 1e9, power_grid.T, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.scatter(peaks_df['voltage'], peaks_df['peak_freq'] / 1e9, color='white', s=10, label='Peak Positions')

    # Plot title with experimental details
    title = (f"Experiment ID: {experiment_id} - {readout_type.capitalize()} Readout\n"
             f"Loop Phase: {settings['set_loop_phase_deg']}°, Loop Att: {settings['set_loop_att']} dB, "
             f"Loopback Att: {settings['set_loopback_att']} dB\n"
             f"Cavity FB Phase: {settings['set_cavity_fb_phase_deg']}°, Cavity FB Att: {settings['set_cavity_fb_att']} dB, "
             f"YIG FB Phase: {settings['set_yig_fb_phase_deg']}°, YIG FB Att: {settings['set_yig_fb_att']} dB")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Voltage (V)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Readout Frequency (GHz)', fontsize=LABEL_FONT_SIZE)
    cbar = fig.colorbar(c, ax=ax, label='Power (dBm)', pad=0.02)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    plt.tight_layout()

    # Save the plot
    file_path = os.path.join(folder, f"{readout_type}_peaks_plot_experiment_{experiment_id}.png")
    plt.savefig(file_path, dpi=SAVE_DPI, transparent=False, facecolor='white')
    plt.close(fig)

    print(f"Plot with peaks saved to {file_path}")


def plot_all_experiments_with_peaks(db_path, freq_min=1e9, freq_max=99e9, voltage_min=-2.0, voltage_max=2.0,
                                    vmin_transmission=None, vmax_transmission=None):
    engine = __get_engine(db_path)
    experiment_ids = pd.read_sql_query(f"SELECT DISTINCT experiment_id FROM {TABLE_NAME}", engine)
    readout_types = ['normal', 'cavity', 'yig']

    database_folder = f"{os.path.basename(db_path)}_colorplots_with_peaks"
    plots_folder = os.path.join(PLOTS_FOLDER, database_folder)

    for experiment_id in experiment_ids['experiment_id']:
        for readout_type in readout_types:
            print(f"Processing experiment {experiment_id}, readout type: {readout_type}...")

            # Fetch data and settings
            power_grid, voltages, frequencies, settings = __get_data_from_db(
                engine, experiment_id, readout_type, freq_min, freq_max, voltage_min, voltage_max)
            if power_grid is None:
                continue

            # Detect peaks
            peaks_df = __process_all_traces(power_grid, voltages, frequencies,
                                            peak_finding_function=__default_peak_finding_function,
                                            experiment_id=experiment_id, readout_type=readout_type)

            # Generate and save the plot
            __generate_transmission_plot_with_peaks(power_grid, voltages, frequencies, peaks_df,
                                                    experiment_id, readout_type, settings, folder=plots_folder,
                                                    vmin=vmin_transmission, vmax=vmax_transmission)


if __name__ == "__main__":
    db_path = '../databases/12_9_overnight.db'
    plot_all_experiments_with_peaks(db_path=db_path,
                                    voltage_min=-3,
                                    voltage_max=0,
                                    freq_min=6.0185e9,
                                    freq_max=6.0205e9, )
