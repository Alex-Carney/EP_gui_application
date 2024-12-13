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


def find_and_fit_peaks(frequencies, power_dbm, readout_type, center_freq_hz=None, span_freq_hz=None,
                       num_peaks_to_find=1):
    """
    Find peaks and fit them with Lorentzian(s).
    - Normal mode: double Lorentzian fit
    - YIG/Cavity mode: single Lorentzian fit

    Returns a list of one dictionary.

    For normal mode:
        {
            'readout_type': readout_type,
            'peak_freqs_ghz': [center1, center2],
            'fit_result': out,
            'x_fit_ghz': x_fit_ghz,
            'y_fit_linear': y_fit_linear
        }

    For single-peak modes:
        {
            'readout_type': readout_type,
            'peak_freq_ghz': center,
            'fwhm_ghz': fwhm,
            'fit_result': out,
            'x_fit_ghz': x_fit_ghz,
            'y_fit_linear': y_fit_linear
        }
    """

    peaks_info = []

    if center_freq_hz is None:
        center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    if span_freq_hz is None:
        span_freq_hz = frequencies.max() - frequencies.min()

    power_linear = 10 ** (power_dbm / 10)
    prominence = 0.1
    peaks, properties = find_peaks(power_dbm, prominence=prominence)

    if len(peaks) == 0:
        return peaks_info

    prominences = properties["prominences"]
    peak_freqs_hz = frequencies[peaks]

    # Scoring peaks
    distances = abs(peak_freqs_hz - center_freq_hz)
    sigma = span_freq_hz / 4
    distance_weighting = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    scores = prominences * distance_weighting
    sorted_indices = np.argsort(scores)[::-1]

    if readout_type == "normal":
        # We want 2 peaks
        if len(peaks) >= 2:
            desired_peaks = 2
        else:
            # Only one peak found, we still want two:
            desired_peaks = 1
    else:
        desired_peaks = min(num_peaks_to_find, len(peaks))

    top_peaks = peaks[sorted_indices[:desired_peaks]]

    if readout_type == "normal":
        # Double Lorentzian fitting
        if len(top_peaks) == 0:
            # No peaks at all
            peaks_info.append({
                'readout_type': readout_type,
                'peak_freqs_ghz': [],
                'fit_result': None,
                'x_fit_ghz': None,
                'y_fit_linear': None
            })
            return peaks_info

        if len(top_peaks) == 1:
            # One peak found by scipy
            # Use this peak as first guess
            peak_freq_hz1 = frequencies[top_peaks[0]]
            peak_freq_ghz1 = peak_freq_hz1 / 1e9
            # For the second peak guess, add a small offset (0.001 GHz):
            peak_freq_ghz2 = peak_freq_ghz1 + 0.001
        else:
            # Two peaks found
            peak_idx1, peak_idx2 = top_peaks[:2]
            peak_freq_hz1 = frequencies[peak_idx1];
            peak_freq_ghz1 = peak_freq_hz1 / 1e9
            peak_freq_hz2 = frequencies[peak_idx2];
            peak_freq_ghz2 = peak_freq_hz2 / 1e9

        # Determine fit range
        # Just use min and max of these guesses:
        pmin = min(peak_freq_ghz1, peak_freq_ghz2)
        pmax = max(peak_freq_ghz1, peak_freq_ghz2)
        fit_range_factor = 5
        # Just pick a sigma guess:
        # If we have at least one peak from scipy, we can get fwhm from that
        # but user doesn't want fwhm lines. We just guess sigma from span
        sigma_guess = (span_freq_hz / 1e9) * 0.001
        left_fit_freq_ghz = pmin - fit_range_factor * sigma_guess
        right_fit_freq_ghz = pmax + fit_range_factor * sigma_guess

        fit_mask = (frequencies / 1e9 >= left_fit_freq_ghz) & (frequencies / 1e9 <= right_fit_freq_ghz)
        x_fit_ghz = frequencies[fit_mask] / 1e9
        y_fit_linear = power_linear[fit_mask]

        if len(x_fit_ghz) < 5:
            peaks_info.append({
                'readout_type': readout_type,
                'peak_freqs_ghz': [peak_freq_ghz1, peak_freq_ghz2],
                'fit_result': None,
                'x_fit_ghz': x_fit_ghz,
                'y_fit_linear': y_fit_linear
            })
            return peaks_info

        # Double Lorentzian fit
        lz1 = LorentzianModel(prefix='lz1_')
        lz2 = LorentzianModel(prefix='lz2_')
        mod = lz1 + lz2
        max_height = y_fit_linear.max()

        # Initial sigma guesses
        sigma_guess1 = 0.0005
        sigma_guess2 = 0.0005
        amp_guess1 = max_height * np.pi * sigma_guess1
        amp_guess2 = max_height * np.pi * sigma_guess2

        pars = mod.make_params()
        pars['lz1_center'].set(value=peak_freq_ghz1, min=x_fit_ghz.min(), max=x_fit_ghz.max())
        pars['lz1_amplitude'].set(value=amp_guess1, min=0)
        pars['lz1_sigma'].set(value=sigma_guess1, min=1e-6)

        pars['lz2_center'].set(value=peak_freq_ghz2, min=x_fit_ghz.min(), max=x_fit_ghz.max())
        pars['lz2_amplitude'].set(value=amp_guess2, min=0)
        pars['lz2_sigma'].set(value=sigma_guess2, min=1e-6)

        try:
            out = mod.fit(y_fit_linear, pars, x=x_fit_ghz)
        except Exception:
            out = None

        peaks_info.append({
            'readout_type': readout_type,
            'peak_freqs_ghz': [peak_freq_ghz1, peak_freq_ghz2],
            'fit_result': out,
            'x_fit_ghz': x_fit_ghz,
            'y_fit_linear': y_fit_linear
        })
        return peaks_info

    else:
        # Single peak mode
        peak_idx = top_peaks[0]
        peak_freq_hz = frequencies[peak_idx]
        peak_freq_ghz = peak_freq_hz / 1e9

        results_half = peak_widths(power_linear, [peak_idx], rel_height=0.5)
        widths = results_half[0]
        freq_step_hz = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
        fwhm_hz = widths[0] * freq_step_hz
        fwhm_ghz = fwhm_hz / 1e9

        fit_range_factor = 5
        left_fit_freq_ghz = peak_freq_ghz - fit_range_factor * fwhm_ghz
        right_fit_freq_ghz = peak_freq_ghz + fit_range_factor * fwhm_ghz

        fit_mask = (frequencies / 1e9 >= left_fit_freq_ghz) & (frequencies / 1e9 <= right_fit_freq_ghz)
        x_fit_ghz = frequencies[fit_mask] / 1e9
        y_fit_linear = power_linear[fit_mask]

        if len(x_fit_ghz) < 5:
            peaks_info.append({
                'readout_type': readout_type,
                'peak_freq_ghz': peak_freq_ghz,
                'fwhm_ghz': fwhm_ghz,
                'fit_result': None,
                'x_fit_ghz': x_fit_ghz,
                'y_fit_linear': y_fit_linear
            })
            return peaks_info

        # Single Lorentzian fit
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
        except Exception:
            out = None

        peaks_info.append({
            'readout_type': readout_type,
            'peak_freq_ghz': peak_freq_ghz,
            'fwhm_ghz': fwhm_ghz,
            'fit_result': out,
            'x_fit_ghz': x_fit_ghz,
            'y_fit_linear': y_fit_linear
        })
        return peaks_info


def fit_and_plot_single_trace(voltage, frequencies, power_dbm, output_folder, readout_type):
    center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    span_freq_hz = frequencies.max() - frequencies.min()

    peaks_info = find_and_fit_peaks(frequencies, power_dbm, readout_type,
                                    center_freq_hz=center_freq_hz, span_freq_hz=span_freq_hz)
    freqs_in_ghz = frequencies / 1e9

    result_dict = {
        'voltage': voltage,
        'omega': np.nan,
        'omega_unc': np.nan,
        'kappa': np.nan,
        'kappa_unc': np.nan
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs_in_ghz, power_dbm, 'b', label='Data')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm)')
    ax.set_title(f'Voltage = {voltage:.3f} V, {readout_type}')

    if len(peaks_info) > 0:
        peak = peaks_info[0]
        out = peak['fit_result']

        if readout_type == 'normal':
            # Double Lorentzian
            peak_freqs_ghz = peak['peak_freqs_ghz']
            # Plot initial guesses as red stars
            for pf in peak_freqs_ghz:
                peak_power_initial = power_dbm[np.argmin(abs(freqs_in_ghz - pf))]
                ax.plot(pf, peak_power_initial, 'r*', markersize=10)

            if out is not None:
                # Evaluate full fit
                x_fit_ghz = peak['x_fit_ghz']
                y_fit_linear = out.best_fit
                y_fit_db = 10 * np.log10(y_fit_linear)
                ax.plot(x_fit_ghz, y_fit_db, 'm--', label='Double Lorentzian Sum')

                # Evaluate components
                comps = out.eval_components(x=x_fit_ghz)
                # comps keys: 'lz1_' and 'lz2_'
                lz1 = comps['lz1_']
                lz2 = comps['lz2_']

                y_fit_db_lz1 = 10 * np.log10(lz1)
                y_fit_db_lz2 = 10 * np.log10(lz2)

                ax.plot(x_fit_ghz, y_fit_db_lz1, 'c:', label='Lorentzian 1')
                ax.plot(x_fit_ghz, y_fit_db_lz2, 'y:', label='Lorentzian 2')

                # Get centers from fit
                center1 = out.params['lz1_center'].value
                center2 = out.params['lz2_center'].value
                center1_unc = out.params['lz1_center'].stderr
                center2_unc = out.params['lz2_center'].stderr

                # Mark each lorentzian center
                # Evaluate at center for power
                pwr_lin1 = out.eval(x=np.array([center1]))
                pwr_db1 = 10 * np.log10(pwr_lin1)
                ax.plot(center1, pwr_db1, 'c*', markersize=10, label='Lorentzian 1 Peak')

                pwr_lin2 = out.eval(x=np.array([center2]))
                pwr_db2 = 10 * np.log10(pwr_lin2)
                ax.plot(center2, pwr_db2, 'y*', markersize=10, label='Lorentzian 2 Peak')

                # Annotate peak freq (no fwhm)
                ann1 = f"Peak1: {center1:.6f} GHz"
                if center1_unc is not None:
                    ann1 += f" ± {(center1_unc * 1e3):.3f} MHz"
                ax.annotate(
                    ann1,
                    (center1, pwr_db1), textcoords="offset points", xytext=(0, 20), ha='center', color='c'
                )

                ann2 = f"Peak2: {center2:.6f} GHz"
                if center2_unc is not None:
                    ann2 += f" ± {(center2_unc * 1e3):.3f} MHz"
                ax.annotate(
                    ann2,
                    (center2, pwr_db2), textcoords="offset points", xytext=(0, 20), ha='center', color='y'
                )

                # For normal mode, we must pick something for result_dict's omega/kappa?
                # The user didn't specify how to store result_dict for two peaks.
                # We'll just store the first peak (lorentzian 1) in result_dict:
                result_dict['omega'] = center1
                result_dict['omega_unc'] = center1_unc if center1_unc is not None else np.nan
                # No kappa in normal mode requested by user explicitly?
                # The user did not say we can't store them. They said no fwhm line on plot.
                # We'll just not store kappa as we have two peaks. Or store NaN:
                result_dict['kappa'] = np.nan
                result_dict['kappa_unc'] = np.nan

            else:
                # No fit result
                # Just return no fit info
                pass

        else:
            # Single peak mode
            peak_freq_ghz = peak['peak_freq_ghz']
            peak_power_initial = power_dbm[np.argmin(abs(freqs_in_ghz - peak_freq_ghz))]
            ax.plot(peak_freq_ghz, peak_power_initial, 'r*', markersize=10, label='Initial Peak Guess')

            if out is not None:
                center = out.params['lz_center'].value
                sigma = out.params['lz_sigma'].value
                center_unc = out.params['lz_center'].stderr
                sigma_unc = out.params['lz_sigma'].stderr

                fwhm = 2 * sigma
                fwhm_unc = None
                if sigma_unc is not None:
                    fwhm_unc = 2 * sigma_unc

                peak_power_linear = out.eval(x=np.array([center]))
                peak_power_db_fit = 10 * np.log10(peak_power_linear)

                # Blue star for Lorentzian peak
                ax.plot(center, peak_power_db_fit, 'b*', markersize=10, label='Lorentzian Peak')

                # Annotate peak freq
                peak_annotation = f"Peak: {center:.6f} GHz"
                if center_unc is not None:
                    peak_annotation += f" ± {(center_unc * 1e3):.3f} MHz"
                ax.annotate(
                    peak_annotation,
                    (center, peak_power_db_fit), textcoords="offset points", xytext=(0, 20), ha='center', color='blue'
                )

                # Lorentzian fit curve
                x_fit_ghz = peak['x_fit_ghz']
                y_fit_linear = out.best_fit
                y_fit_db = 10 * np.log10(y_fit_linear)
                ax.plot(x_fit_ghz, y_fit_db, 'm--', label='Lorentzian Fit')

                # FWHM line
                height = peak_power_db_fit - 3
                left_freq = center - fwhm / 2
                right_freq = center + fwhm / 2
                ax.hlines(height, left_freq, right_freq, color='green', linestyle='--')
                fwhm_annotation = f"FWHM: {(fwhm * 1e3):.3f} MHz"
                if fwhm_unc is not None:
                    fwhm_annotation += f" ± {(fwhm_unc * 1e3):.3f} MHz"
                ax.annotate(
                    fwhm_annotation,
                    ((left_freq + right_freq) / 2, height), textcoords="offset points", xytext=(0, -20), ha='center',
                    color='green'
                )

                # Update result_dict
                result_dict['omega'] = center
                result_dict['omega_unc'] = center_unc if center_unc is not None else np.nan
                result_dict['kappa'] = fwhm
                result_dict['kappa_unc'] = fwhm_unc if fwhm_unc is not None else np.nan

    ax.legend()
    plt.tight_layout()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, f"trace_V_{voltage:.3f}V.png")
    plt.savefig(file_path, dpi=300)
    plt.close(fig)
    print(f"Saved trace plot with fit for voltage={voltage:.3f} V to {file_path}")

    return result_dict


def __process_all_traces(power_grid, voltages, frequencies, peak_finding_function, experiment_id, readout_type):
    # Here we integrate the new fitting logic
    # Instead of just returning a DF, we also save individual trace plots.
    output_folder = os.path.join("traces_plots", f"{experiment_id}_{readout_type}")
    output_folder_csv = os.path.join("csv", f"{experiment_id}_{readout_type}")

    if not os.path.exists(output_folder_csv):
        os.makedirs(output_folder_csv)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    voltage_list, peak_freqs_list, peak_powers_list = [], [], []
    results_list_for_saving_to_csv = []
    for idx, voltage in enumerate(voltages):
        powers = power_grid[idx, :]
        # We can call fit_and_plot_single_trace for each
        trace_result = fit_and_plot_single_trace(voltage, frequencies, powers, output_folder, readout_type)

        trace_result['readout_type'] = readout_type
        results_list_for_saving_to_csv.append(trace_result)

        # If you still want peak info in a DF, you can also call the simpler peak_finding_function:
        peak_freqs, peak_powers = peak_finding_function(frequencies, powers)
        voltage_list.extend([voltage] * len(peak_freqs))
        peak_freqs_list.extend(peak_freqs)
        peak_powers_list.extend(peak_powers)

    # Create a DataFrame of results
    df = pd.DataFrame(results_list_for_saving_to_csv,
                      columns=['voltage', 'readout_type', 'omega', 'omega_unc', 'kappa', 'kappa_unc'])
    # Save DF to CSV
    csv_file = os.path.join(output_folder_csv, f"lorentzian_fit_results_{experiment_id}_{readout_type}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved lorentzian fit results to {csv_file}")

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
                                    freq_min=6.017e9,
                                    freq_max=6.022e9, )
