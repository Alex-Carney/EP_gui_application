import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

PLOTS_FOLDER = 'plots'

TABLE_NAME = 'expr'

LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 15
SAVE_DPI = 400


def __get_engine(db_path):
    return create_engine(f'sqlite:///{db_path}')


def __get_data_from_db(engine, experiment_id, readout_type, freq_min=1e9, freq_max=99e9, voltage_min=-2.0, voltage_max=2.0):
    # Fetch settings for the experiment
    settings_query = f"""
    SELECT DISTINCT set_loop_phase_deg, set_loop_att, set_loopback_att,
                    set_yig_fb_phase_deg, set_yig_fb_att,
                    set_cavity_fb_phase_deg, set_cavity_fb_att
    FROM {TABLE_NAME}
    WHERE experiment_id = '{experiment_id}' AND readout_type = '{readout_type}'
    """
    settings = pd.read_sql_query(settings_query, engine).iloc[0]

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
        print(f"No data found for experiment {experiment_id}, readout type: {readout_type}")
        return None, None, None, None


def __default_peak_finding_function(frequencies, powers):
    # Detect peaks using SciPy
    peaks_indices, _ = find_peaks(powers, height=-30, prominence=0.2, distance=10)
    peak_freqs = frequencies[peaks_indices]
    peak_powers = powers[peaks_indices]
    return peak_freqs, peak_powers


def __process_all_traces(power_grid, voltages, frequencies, peak_finding_function):
    # Extract peaks for all voltage traces
    voltage_list, peak_freqs_list, peak_powers_list = [], [], []
    for idx, voltage in enumerate(voltages):
        powers = power_grid[idx, :]
        peak_freqs, peak_powers = peak_finding_function(frequencies, powers)
        voltage_list.extend([voltage] * len(peak_freqs))
        peak_freqs_list.extend(peak_freqs)
        peak_powers_list.extend(peak_powers)
    return pd.DataFrame({'voltage': voltage_list, 'peak_freq': peak_freqs_list, 'peak_power': peak_powers_list})


def __generate_transmission_plot_with_peaks(power_grid, voltages, frequencies, peaks_df, experiment_id, readout_type, settings,
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
                                    vmin_transmission=-40, vmax_transmission=8):
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
                                            peak_finding_function=__default_peak_finding_function)

            # Generate and save the plot
            __generate_transmission_plot_with_peaks(power_grid, voltages, frequencies, peaks_df,
                                                    experiment_id, readout_type, settings, folder=plots_folder,
                                                    vmin=vmin_transmission, vmax=vmax_transmission)


if __name__ == "__main__":
    db_path = './databases/experiment_data.db'
    plot_all_experiments_with_peaks(db_path=db_path,
                                    voltage_min=0.0,
                                    voltage_max=1.0)
