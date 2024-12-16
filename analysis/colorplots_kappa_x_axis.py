import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from lmfit.models import LorentzianModel
from sqlalchemy import create_engine

PLOTS_FOLDER = 'plots'
TABLE_NAME = 'expr'
LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 15
SAVE_DPI = 400

SAVE_VOLTAGE_TRACES = True


def get_engine(db_path):
    return create_engine(f'sqlite:///{db_path}')


def get_data_from_db(engine, experiment_id, readout_type, freq_min=1e9, freq_max=99e9, voltage_min=-2.0,
                     voltage_max=2.0):
    settings_query = f"""
    SELECT DISTINCT set_loop_phase_deg, set_loop_att, set_loopback_att,
                    set_yig_fb_phase_deg, set_yig_fb_att,
                    set_cavity_fb_phase_deg, set_cavity_fb_att
    FROM {TABLE_NAME}
    WHERE experiment_id = '{experiment_id}' AND readout_type = '{readout_type}'
    """
    settings_df = pd.read_sql_query(settings_query, engine)
    if settings_df.empty:
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

    data_query = f"""
    SELECT frequency_hz, set_voltage, power_dBm FROM {TABLE_NAME}
    WHERE experiment_id = '{experiment_id}'
    AND readout_type = '{readout_type}'
    AND set_voltage BETWEEN {voltage_min} AND {voltage_max}
    AND frequency_hz BETWEEN {freq_min} AND {freq_max}
    ORDER BY set_voltage, frequency_hz
    """
    data = pd.read_sql_query(data_query, engine)
    if data.empty:
        return None, None, None, None

    pivot_table = data.pivot_table(index='set_voltage', columns='frequency_hz', values='power_dBm', aggfunc='first')
    voltages = pivot_table.index.values
    frequencies = pivot_table.columns.values
    power_grid = pivot_table.values
    return power_grid, voltages, frequencies, settings


def find_and_fit_peaks(frequencies, power_dbm, readout_type, center_freq_hz=None, span_freq_hz=None,
                       num_peaks_to_find=1):
    if center_freq_hz is None:
        center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    if span_freq_hz is None:
        span_freq_hz = frequencies.max() - frequencies.min()

    power_linear = 10 ** (power_dbm / 10)
    peaks, properties = find_peaks(power_dbm, prominence=0.1)
    peaks_info = []
    if len(peaks) == 0:
        return peaks_info

    prominences = properties["prominences"]
    peak_freqs_hz = frequencies[peaks]
    distances = abs(peak_freqs_hz - center_freq_hz)
    sigma = span_freq_hz / 4
    distance_weighting = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    scores = prominences * distance_weighting
    sorted_indices = np.argsort(scores)[::-1]

    if readout_type == "normal":
        if len(peaks) >= 2:
            desired_peaks = 2
        else:
            desired_peaks = 1
    else:
        desired_peaks = min(num_peaks_to_find, len(peaks))

    top_peaks = peaks[sorted_indices[:desired_peaks]]

    def double_lorentzian_fit(x_fit_ghz, y_fit_linear, guess_centers):
        lz1 = LorentzianModel(prefix='lz1_')
        lz2 = LorentzianModel(prefix='lz2_')
        mod = lz1 + lz2
        max_height = y_fit_linear.max()
        sigma_guess = 0.0005
        amp_guess1 = max_height * np.pi * sigma_guess
        amp_guess2 = amp_guess1
        pars = mod.make_params()
        pars['lz1_center'].set(value=guess_centers[0], min=x_fit_ghz.min(), max=x_fit_ghz.max())
        pars['lz1_amplitude'].set(value=amp_guess1, min=0)
        pars['lz1_sigma'].set(value=sigma_guess, min=1e-6)

        pars['lz2_center'].set(value=guess_centers[1], min=x_fit_ghz.min(), max=x_fit_ghz.max())
        pars['lz2_amplitude'].set(value=amp_guess2, min=0)
        pars['lz2_sigma'].set(value=sigma_guess, min=1e-6)
        try:
            out = mod.fit(y_fit_linear, pars, x=x_fit_ghz)
        except:
            out = None
        return out

    def single_lorentzian_fit(x_fit_ghz, y_fit_linear, center_guess):
        max_height = y_fit_linear.max()
        sigma_guess = 0.001
        amp_guess = max_height * np.pi * sigma_guess
        lz = LorentzianModel(prefix='lz_')
        pars = lz.make_params()
        pars['lz_center'].set(value=center_guess, min=x_fit_ghz.min(), max=x_fit_ghz.max())
        pars['lz_amplitude'].set(value=amp_guess, min=0)
        pars['lz_sigma'].set(value=sigma_guess, min=1e-6)
        try:
            out = lz.fit(y_fit_linear, pars, x=x_fit_ghz)
        except:
            out = None
        return out

    if readout_type == 'normal':
        if len(top_peaks) == 0:
            peaks_info.append(
                {'readout_type': readout_type, 'peak_freqs_ghz': [], 'fit_result': None, 'x_fit_ghz': None,
                 'y_fit_linear': None})
            return peaks_info
        if len(top_peaks) == 1:
            p_freq_hz = frequencies[top_peaks[0]]
            p_freq_ghz = p_freq_hz / 1e9
            p_freq_ghz2 = p_freq_ghz + 0.001
            guess_centers = [p_freq_ghz, p_freq_ghz2]
        else:
            p1 = frequencies[top_peaks[0]] / 1e9
            p2 = frequencies[top_peaks[1]] / 1e9
            guess_centers = [p1, p2]

        pmin = min(guess_centers)
        pmax = max(guess_centers)
        fit_range_factor = 5
        sigma_guess = (span_freq_hz / 1e9) * 0.001
        left_fit_freq_ghz = pmin - fit_range_factor * sigma_guess
        right_fit_freq_ghz = pmax + fit_range_factor * sigma_guess
        fit_mask = (frequencies / 1e9 >= left_fit_freq_ghz) & (frequencies / 1e9 <= right_fit_freq_ghz)
        x_fit_ghz = frequencies[fit_mask] / 1e9
        y_fit_linear = 10 ** (power_dbm[fit_mask] / 10)

        if len(x_fit_ghz) < 5:
            peaks_info.append({'readout_type': readout_type, 'peak_freqs_ghz': guess_centers, 'fit_result': None,
                               'x_fit_ghz': x_fit_ghz, 'y_fit_linear': y_fit_linear})
            return peaks_info
        out = double_lorentzian_fit(x_fit_ghz, y_fit_linear, guess_centers)
        peaks_info.append(
            {'readout_type': readout_type, 'peak_freqs_ghz': guess_centers, 'fit_result': out, 'x_fit_ghz': x_fit_ghz,
             'y_fit_linear': y_fit_linear})
        return peaks_info
    else:
        peak_idx = top_peaks[0]
        peak_freq_hz = frequencies[peak_idx]
        peak_freq_ghz = peak_freq_hz / 1e9
        power_linear = 10 ** (power_dbm / 10)
        res_half = peak_widths(power_linear, [peak_idx], rel_height=0.5)
        widths = res_half[0]
        freq_step_hz = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)
        fwhm_hz = widths[0] * freq_step_hz
        fwhm_ghz = fwhm_hz / 1e9

        fit_range_factor = 5
        left_fit_freq_ghz = peak_freq_ghz - fit_range_factor * fwhm_ghz
        right_fit_freq_ghz = peak_freq_ghz + fit_range_factor * fwhm_ghz
        fit_mask = (frequencies / 1e9 >= left_fit_freq_ghz) & (frequencies / 1e9 <= right_fit_freq_ghz)
        x_fit_ghz = frequencies[fit_mask] / 1e9
        y_fit_linear = 10 ** (power_dbm[fit_mask] / 10)

        if len(x_fit_ghz) < 5:
            peaks_info.append(
                {'readout_type': readout_type, 'peak_freq_ghz': peak_freq_ghz, 'fwhm_ghz': fwhm_ghz, 'fit_result': None,
                 'x_fit_ghz': x_fit_ghz, 'y_fit_linear': y_fit_linear})
            return peaks_info
        out = single_lorentzian_fit(x_fit_ghz, y_fit_linear, peak_freq_ghz)
        peaks_info.append(
            {'readout_type': readout_type, 'peak_freq_ghz': peak_freq_ghz, 'fwhm_ghz': fwhm_ghz, 'fit_result': out,
             'x_fit_ghz': x_fit_ghz, 'y_fit_linear': y_fit_linear})
        return peaks_info


def fit_and_plot_single_trace(voltage, frequencies, power_dbm, output_folder, readout_type):
    center_freq_hz = (frequencies.min() + frequencies.max()) / 2
    span_freq_hz = frequencies.max() - frequencies.min()
    peaks_info = find_and_fit_peaks(frequencies, power_dbm, readout_type, center_freq_hz=center_freq_hz,
                                    span_freq_hz=span_freq_hz)
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
            peak_freqs_ghz = peak['peak_freqs_ghz']
            for pf in peak_freqs_ghz:
                peak_power_initial = power_dbm[np.argmin(abs(freqs_in_ghz - pf))]
                ax.plot(pf, peak_power_initial, 'r*', markersize=10)

            if out is not None:
                x_fit_ghz = peak['x_fit_ghz']
                y_fit_linear = out.best_fit
                y_fit_db = 10 * np.log10(y_fit_linear)
                ax.plot(x_fit_ghz, y_fit_db, 'm--', label='Double Lorentzian Sum')
                comps = out.eval_components(x=x_fit_ghz)
                lz1 = comps['lz1_']
                lz2 = comps['lz2_']
                y_fit_db_lz1 = 10 * np.log10(lz1)
                y_fit_db_lz2 = 10 * np.log10(lz2)
                ax.plot(x_fit_ghz, y_fit_db_lz1, 'c:', label='Lorentzian 1')
                ax.plot(x_fit_ghz, y_fit_db_lz2, 'y:', label='Lorentzian 2')
                center1 = out.params['lz1_center'].value
                center1_unc = out.params['lz1_center'].stderr
                sigma1 = out.params['lz1_sigma'].value
                sigma1_unc = out.params['lz1_sigma'].stderr
                pwr_lin1 = out.eval(x=np.array([center1]))
                pwr_db1 = 10 * np.log10(pwr_lin1)
                ax.plot(center1, pwr_db1, 'c*', markersize=10, label='Lorentzian 1 Peak')
                center2 = out.params['lz2_center'].value
                center2_unc = out.params['lz2_center'].stderr
                pwr_lin2 = out.eval(x=np.array([center2]))
                pwr_db2 = 10 * np.log10(pwr_lin2)
                ax.plot(center2, pwr_db2, 'y*', markersize=10, label='Lorentzian 2 Peak')

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

                fwhm1 = 2 * sigma1
                fwhm1_unc = None
                if sigma1_unc is not None:
                    fwhm1_unc = 2 * sigma1_unc
                result_dict['omega'] = center1
                result_dict['omega_unc'] = center1_unc if center1_unc else np.nan
                result_dict['kappa'] = fwhm1
                result_dict['kappa_unc'] = fwhm1_unc if fwhm1_unc else np.nan

        else:
            if out is not None:
                center = out.params['lz_center'].value
                sigma = out.params['lz_sigma'].value
                center_unc = out.params['lz_center'].stderr
                sigma_unc = out.params['lz_sigma'].stderr
                fwhm = 2 * sigma
                fwhm_unc = None
                if sigma_unc is not None:
                    fwhm_unc = 2 * sigma_unc
                peak_freq_ghz = peak['peak_freq_ghz']
                peak_power_initial = power_dbm[np.argmin(abs(freqs_in_ghz - peak_freq_ghz))]
                ax.plot(peak_freq_ghz, peak_power_initial, 'r*', markersize=10, label='Initial Peak Guess')
                peak_power_linear = out.eval(x=np.array([center]))
                peak_power_db_fit = 10 * np.log10(peak_power_linear)
                ax.plot(center, peak_power_db_fit, 'b*', markersize=10, label='Lorentzian Peak')
                peak_annotation = f"Peak: {center:.6f} GHz"
                if center_unc is not None:
                    peak_annotation += f" ± {(center_unc * 1e3):.3f} MHz"
                ax.annotate(
                    peak_annotation,
                    (center, peak_power_db_fit), textcoords="offset points", xytext=(0, 20), ha='center', color='blue'
                )
                x_fit_ghz = peak['x_fit_ghz']
                y_fit_linear = out.best_fit
                y_fit_db = 10 * np.log10(y_fit_linear)
                ax.plot(x_fit_ghz, y_fit_db, 'm--', label='Lorentzian Fit')
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
                result_dict['omega'] = center
                result_dict['omega_unc'] = center_unc if center_unc else np.nan
                result_dict['kappa'] = fwhm
                result_dict['kappa_unc'] = fwhm_unc if fwhm_unc else np.nan

    ax.legend()
    plt.tight_layout()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, f"trace_V_{voltage:.3f}V.png")

    if SAVE_VOLTAGE_TRACES:
        plt.savefig(file_path, dpi=SAVE_DPI)
        print(f"Saved trace plot with fit for voltage={voltage:.3f} V to {file_path}")
    plt.close(fig)
    return result_dict


def process_all_traces(power_grid, voltages, frequencies, peak_finding_function, experiment_id, readout_type):
    output_folder = os.path.join("traces_plots", f"{experiment_id}_{readout_type}")
    output_folder_csv = os.path.join("csv", f"{experiment_id}_{readout_type}")
    if not os.path.exists(output_folder_csv):
        os.makedirs(output_folder_csv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    voltage_list, peak_freqs_list, peak_powers_list = [], [], []
    results_list_for_csv = []
    for idx, voltage in enumerate(voltages):
        powers = power_grid[idx, :]
        trace_result = fit_and_plot_single_trace(voltage, frequencies, powers, output_folder, readout_type)
        trace_result['readout_type'] = readout_type
        results_list_for_csv.append(trace_result)
        pfreqs, ppows = peak_finding_function(frequencies, powers)
        voltage_list.extend([voltage] * len(pfreqs))
        peak_freqs_list.extend(pfreqs)
        peak_powers_list.extend(ppows)
    df = pd.DataFrame(results_list_for_csv,
                      columns=['voltage', 'readout_type', 'omega', 'omega_unc', 'kappa', 'kappa_unc'])
    csv_file = os.path.join(output_folder_csv, f"lorentzian_fit_results_{experiment_id}_{readout_type}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved lorentzian fit results to {csv_file}")
    return pd.DataFrame({'voltage': voltage_list, 'peak_freq': peak_freqs_list, 'peak_power': peak_powers_list})


def compute_capital_kappa_and_capital_delta(cavity_df, yig_df):
    cavity_df = cavity_df.rename(
        columns={'omega': 'omega_c', 'omega_unc': 'omega_c_unc', 'kappa': 'kappa_c', 'kappa_unc': 'kappa_c_unc'})
    yig_df = yig_df.rename(
        columns={'omega': 'omega_y', 'omega_unc': 'omega_y_unc', 'kappa': 'kappa_y', 'kappa_unc': 'kappa_y_unc'})
    merged_df = pd.merge(cavity_df[['voltage', 'omega_c', 'omega_c_unc', 'kappa_c', 'kappa_c_unc']],
                         yig_df[['voltage', 'omega_y', 'omega_y_unc', 'kappa_y', 'kappa_y_unc']],
                         on='voltage')
    merged_df['Kappa'] = merged_df['kappa_c'] - merged_df['kappa_y']
    merged_df['Kappa_unc'] = np.sqrt(merged_df['kappa_c_unc'] ** 2 + merged_df['kappa_y_unc'] ** 2)
    merged_df['Delta'] = merged_df['omega_c'] - merged_df['omega_y']
    merged_df['Delta_unc'] = np.sqrt(merged_df['omega_c_unc'] ** 2 + merged_df['omega_y_unc'] ** 2)
    return merged_df


def plot_kappa_colorplot(power_grid, voltages, frequencies, merged_df, experiment_id, readout_type, settings,
                         folder, vmin=None, vmax=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    kappa_map = merged_df[['voltage', 'Kappa', 'Kappa_unc']].drop_duplicates().set_index('voltage')
    avail_voltages = np.intersect1d(voltages, kappa_map.index.values)
    mask = np.isin(voltages, avail_voltages)
    power_grid = power_grid[mask, :]
    voltages = voltages[mask]

    kappas = kappa_map.loc[voltages, 'Kappa'].values
    sort_idx = np.argsort(kappas)
    kappas_sorted = kappas[sort_idx]
    power_grid_sorted = power_grid[sort_idx, :]

    kappa_edges = np.zeros(len(kappas_sorted) + 1)
    if len(kappas_sorted) > 1:
        kappa_edges[1:-1] = (kappas_sorted[:-1] + kappas_sorted[1:]) / 2
        kappa_edges[0] = kappas_sorted[0] - (kappas_sorted[1] - kappas_sorted[0]) / 2
        kappa_edges[-1] = kappas_sorted[-1] + (kappas_sorted[-1] - kappas_sorted[-2]) / 2
    else:
        kappa_edges[0] = kappas_sorted[0] - 0.001
        kappa_edges[-1] = kappas_sorted[0] + 0.001

    freq_edges = np.zeros(len(frequencies) + 1)
    if len(frequencies) > 1:
        freq_edges[1:-1] = (frequencies[:-1] + frequencies[1:]) / 2
        freq_edges[0] = frequencies[0] - (frequencies[1] - frequencies[0]) / 2
        freq_edges[-1] = frequencies[-1] + (frequencies[-1] - frequencies[-2]) / 2
    else:
        freq_edges[0] = frequencies[0] - 1e6
        freq_edges[-1] = frequencies[-1] + 1e6

    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(kappa_edges, freq_edges / 1e9, power_grid_sorted.T, shading='auto', cmap='inferno', vmin=vmin,
                      vmax=vmax)
    title = (f"Exp ID: {experiment_id} - {readout_type.capitalize()} Readout\n"
             f"Loop Phase: {settings['set_loop_phase_deg']}°, Loop Att: {settings['set_loop_att']} dB, "
             f"Loopback Att: {settings['set_loopback_att']} dB\n"
             f"Cavity FB Phase: {settings['set_cavity_fb_phase_deg']}°, Cavity FB Att: {settings['set_cavity_fb_att']} dB, "
             f"YIG FB Phase: {settings['set_yig_fb_phase_deg']}°, YIG FB Att: {settings['set_yig_fb_att']} dB")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Kappa (GHz)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Frequency (GHz)', fontsize=LABEL_FONT_SIZE)
    cbar = fig.colorbar(c, ax=ax, label='Power (dBm)', pad=0.02)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    plt.tight_layout()
    file_path = os.path.join(folder, f"{readout_type}_Kappa_plot_experiment_{experiment_id}.png")
    plt.savefig(file_path, dpi=SAVE_DPI, transparent=False, facecolor='white')
    plt.close(fig)
    print(f"Kappa-based plot saved to {file_path}")


def replot_normal_mode_traces_with_kappa(normal_df, merged_df, power_grid, voltages, frequencies, experiment_id):
    temp = pd.merge(normal_df, merged_df[['voltage', 'Kappa']], on='voltage', how='inner')
    output_folder = os.path.join("traces_plots", f"{experiment_id}_normal_by_kappa")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def plot_normal_trace_by_kappa(kappa_value, voltage, frequencies, power_dbm, peaks_info):
        freqs_in_ghz = frequencies / 1e9
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs_in_ghz, power_dbm, 'b', label='Data')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Power (dBm)')
        ax.set_title(f'Kappa = {kappa_value:.6f} GHz (Voltage={voltage:.3f} V, Normal Mode)')

        if len(peaks_info) > 0:
            peak = peaks_info[0]
            out = peak['fit_result']
            peak_freqs_ghz = peak['peak_freqs_ghz']
            for pf in peak_freqs_ghz:
                peak_power_initial = power_dbm[np.argmin(abs(freqs_in_ghz - pf))]
                ax.plot(pf, peak_power_initial, 'r*', markersize=10)
            if out is not None:
                x_fit_ghz = peak['x_fit_ghz']
                y_fit_linear = out.best_fit
                y_fit_db = 10 * np.log10(y_fit_linear)
                ax.plot(x_fit_ghz, y_fit_db, 'm--', label='Double Lorentzian Sum')
                comps = out.eval_components(x=x_fit_ghz)
                lz1 = comps['lz1_']
                lz2 = comps['lz2_']
                y_fit_db_lz1 = 10 * np.log10(lz1)
                y_fit_db_lz2 = 10 * np.log10(lz2)
                ax.plot(x_fit_ghz, y_fit_db_lz1, 'c:', label='Lorentzian 1')
                ax.plot(x_fit_ghz, y_fit_db_lz2, 'y:', label='Lorentzian 2')
                center1 = out.params['lz1_center'].value
                center2 = out.params['lz2_center'].value
                pwr_lin1 = out.eval(x=np.array([center1]))
                pwr_db1 = 10 * np.log10(pwr_lin1)
                ax.plot(center1, pwr_db1, 'c*', markersize=10, label='Lorentzian 1 Peak')
                pwr_lin2 = out.eval(x=np.array([center2]))
                pwr_db2 = 10 * np.log10(pwr_lin2)
                ax.plot(center2, pwr_db2, 'y*', markersize=10, label='Lorentzian 2 Peak')
        ax.legend()
        plt.tight_layout()
        file_path = os.path.join(output_folder, f"trace_K_{kappa_value:.6f}.png")
        plt.savefig(file_path, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"Saved normal mode trace plot with Kappa={kappa_value:.6f} GHz to {file_path}")

    for i, row in temp.iterrows():
        idx = np.argmin(abs(voltages - row['voltage']))
        powers = power_grid[idx, :]
        # Re-run find_and_fit_peaks for normal mode
        peaks_info = find_and_fit_peaks(frequencies, powers, 'normal')
        plot_normal_trace_by_kappa(row['Kappa'], row['voltage'], frequencies, powers, peaks_info)


def save_peak_differences_vs_kappa(normal_df, merged_df, power_grid, voltages, frequencies, output_folder,
                                   experiment_id):
    """
    Save the difference between the two Lorentzian peaks vs. Kappa to a CSV file.

    Parameters:
        normal_df (pd.DataFrame): DataFrame with normal mode data.
        merged_df (pd.DataFrame): DataFrame with merged Kappa data.
        power_grid (np.ndarray): Power grid data.
        voltages (np.ndarray): Array of voltages.
        frequencies (np.ndarray): Array of frequencies.
        output_folder (str): Path to save the CSV file.
        experiment_id (str): Experiment identifier.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    temp = pd.merge(normal_df, merged_df[['voltage', 'Kappa', 'Kappa_unc']], on='voltage', how='inner')

    peak_diff_data = []
    for i, row in temp.iterrows():
        idx = np.argmin(abs(voltages - row['voltage']))
        powers = power_grid[idx, :]
        peaks_info = find_and_fit_peaks(frequencies, powers, 'normal')

        if len(peaks_info) > 0:
            peak = peaks_info[0]
            out = peak['fit_result']
            if out is not None:
                center1 = out.params['lz1_center'].value
                center2 = out.params['lz2_center'].value
                center1_unc = out.params['lz1_center'].stderr
                center2_unc = out.params['lz2_center'].stderr

                if center1_unc is None:
                    center1_unc = 0
                if center2_unc is None:
                    center2_unc = 0

                peak_difference = abs(center1 - center2)
                # uncertainty in difference is sqrt(center1_unc² + center2_unc²)
                peak_diff_unc = np.sqrt(center1_unc ** 2 + center2_unc ** 2)

                peak_diff_data.append({
                    'Kappa': row['Kappa'],
                    'Kappa_unc': row['Kappa_unc'],
                    'Peak1 (GHz)': center1,
                    'Peak2 (GHz)': center2,
                    'Peak1_unc(GHz)': center1_unc,
                    'Peak2_unc(GHz)': center2_unc,
                    'Peak Difference (GHz)': peak_difference,
                    'Peak Difference_unc (GHz)': peak_diff_unc
                })

    peak_diff_df = pd.DataFrame(peak_diff_data)
    file_path = os.path.join(output_folder, f"peak_differences_vs_kappa_{experiment_id}.csv")
    peak_diff_df.to_csv(file_path, index=False)
    print(f"Saved peak differences vs. Kappa to {file_path}")

    plot_peak_differences_vs_kappa(file_path, output_folder)

    return peak_diff_df, file_path


def eigenvalues(gain_loss, detuning, coupling, phi):
    """
    Calculate eigenvalues of the given 2x2 matrix system.
    gain_loss: K/J (dimensionless)
    detuning: Δ/J (dimensionless)
    coupling: J (we can set this to 1 for simplicity)
    phi: phase parameter
    """
    matrix = np.array([
        [((detuning * -1j) / 2) - (gain_loss / 2), -1j * coupling * np.exp(1j * phi)],
        [-1j * coupling, ((detuning * 1j) / 2) + (gain_loss / 2)]
    ])
    eigenvals = np.linalg.eigvals(matrix)
    return eigenvals


def get_imag_diff_trace(K_min, K_max, N=100, phi=0.0, delta=0.0, coupling=1.0):
    """
    Compute the imaginary-part difference of the eigenvalues as a function of K/J.

    Parameters:
        K_min (float): Minimum value of K/J.
        K_max (float): Maximum value of K/J.
        N (int): Number of points in the range.
        phi (float): Phase parameter (default 0).
        delta (float): Δ/J (default 0).
        coupling (float): J (default 1.0).

    Returns:
        K_values (ndarray): Array of K/J values.
        imag_diff (ndarray): Array of the imaginary-part differences of eigenvalues at each K/J.
    """
    K_values = np.linspace(K_min, K_max, N)
    imag_diff = np.zeros_like(K_values)

    for i, K in enumerate(K_values):
        evs = eigenvalues(K, delta, coupling, phi)
        # Imag part difference
        imag_diff[i] = np.abs(evs[0].imag - evs[1].imag)
    return K_values, imag_diff


def plot_peak_differences_vs_kappa(csv_file, output_folder):
    """
    Plot the experimental data for the difference between two Lorentzian peaks vs. Kappa, scaled by J.
    Also overlay the theoretical imaginary-part difference trace.
    """
    # Read the CSV file
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    peak_diff_df = pd.read_csv(csv_file)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate J (half the max peak splitting)
    max_peak_diff = peak_diff_df['Peak Difference (GHz)'].max()
    max_peak_diff_unc = peak_diff_df.loc[
        peak_diff_df['Peak Difference (GHz)'] == max_peak_diff,
        'Peak Difference_unc (GHz)'
    ].values[0]

    J = max_peak_diff / 2
    J_unc = max_peak_diff_unc / 2

    # Scale the data by J
    peak_diff_df['Scaled Kappa'] = peak_diff_df['Kappa'] / J
    peak_diff_df['Scaled Peak Difference'] = peak_diff_df['Peak Difference (GHz)'] / J

    # Calculate uncertainties for scaled values
    # Uncertainty propagation formula
    peak_diff_df['Scaled Kappa_unc'] = np.abs(peak_diff_df['Scaled Kappa']) * np.sqrt(
        (peak_diff_df['Kappa_unc'] / peak_diff_df['Kappa']) ** 2 + (J_unc / J) ** 2
    )
    peak_diff_df['Scaled Peak Difference_unc'] = np.abs(peak_diff_df['Scaled Peak Difference']) * np.sqrt(
        (peak_diff_df['Peak Difference_unc (GHz)'] / peak_diff_df['Peak Difference (GHz)']) ** 2 + (J_unc / J) ** 2
    )

    # Plot scaled peak difference vs scaled Kappa with error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        peak_diff_df['Scaled Kappa'], peak_diff_df['Scaled Peak Difference'],
        xerr=peak_diff_df['Scaled Kappa_unc'], yerr=peak_diff_df['Scaled Peak Difference_unc'],
        fmt='o', ecolor='red', capsize=4, label='Scaled Peak Splitting (Exp)', markersize=4
    )

    # Draw a vertical line at K/J = -2 (theoretical EP line)
    ax.axvline(x=-2, color='green', linestyle='--', label='Theoretical EP Line')

    # ----------------------------
    # Overlay Theoretical Imag Diff
    # ----------------------------
    # Customize these parameters as needed
    K_min_theory = -3.0
    K_max_theory = 0.0
    phi_theory = np.deg2rad(0)
    delta_theory = 0.0
    coupling_theory = 1

    K_vals_th, imag_diff_th = get_imag_diff_trace(K_min_theory, K_max_theory, N=200,
                                                  phi=phi_theory, delta=delta_theory,
                                                  coupling=coupling_theory)
    # Plot the theoretical line
    ax.plot(K_vals_th - 0, imag_diff_th, label='Theory (Imag Diff)', color='blue', lw=2)

    # Axis labels and title
    ax.set_xlabel('$K / J$', fontsize=14)
    ax.set_ylabel('Splitting / J', fontsize=14)
    ax.set_title('Scaled Double Lorentzian Peak Splitting vs Scaled Kappa', fontsize=16)
    ax.grid(True)
    ax.legend()

    # Set the Y lim based on the max and min values of the data, not the error bars
    y_min = -.25
    y_max = 3

    # Set plot limits
    ax.set_ylim([y_min, y_max])

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "scaled_peak_differences_vs_kappa_plot.png")
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved scaled peak differences vs. Kappa plot to {plot_path}")


def plot_peak_locations_vs_kappa(normal_df, merged_df, power_grid, voltages, frequencies, output_folder, experiment_id,
                                 peak_diff_file):
    """
    Plots the individual Lorentzian peak locations (in normal mode) vs Kappa/J on the X-axis,
    with frequency (GHz) on the Y-axis. Each row yields up to 2 peaks from the double Lorentzian.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Merge normal_df with merged_df to get Kappa
    temp = pd.merge(normal_df, merged_df[['voltage', 'Kappa', 'Kappa_unc']], on='voltage', how='inner')

    # We'll accumulate each row's peaks in a list
    peak_locations_data = []

    for i, row in temp.iterrows():
        idx = np.argmin(abs(voltages - row['voltage']))
        powers = power_grid[idx, :]

        # Re-run find_and_fit_peaks for normal mode to get double Lorentzian
        peaks_info = find_and_fit_peaks(frequencies, powers, 'normal')

        if len(peaks_info) > 0:
            peak = peaks_info[0]
            out = peak['fit_result']
            if out is not None:
                # Extract double Lorentzian centers
                center1 = out.params['lz1_center'].value
                center2 = out.params['lz2_center'].value
                c1_unc = out.params['lz1_center'].stderr or 0.0
                c2_unc = out.params['lz2_center'].stderr or 0.0

                # Save them so we can plot afterwards
                peak_locations_data.append({
                    'Kappa': row['Kappa'],
                    'Kappa_unc': row['Kappa_unc'],
                    'PeakFreqGHz': center1,  # in GHz
                    'PeakFreqUncGHz': c1_unc
                })
                peak_locations_data.append({
                    'Kappa': row['Kappa'],
                    'Kappa_unc': row['Kappa_unc'],
                    'PeakFreqGHz': center2,
                    'PeakFreqUncGHz': c2_unc
                })

    if not peak_locations_data:
        print("No normal-mode peaks found. Exiting plot_peak_locations_vs_kappa.")
        return

    if not os.path.exists(peak_diff_file):
        print(f"CSV file not found: {peak_diff_file}")
        return

    peak_diff_df = pd.read_csv(peak_diff_file)

    peak_locs_df = pd.DataFrame(peak_locations_data)

    # Calculate J (half the max peak splitting)
    max_peak_diff = peak_diff_df['Peak Difference (GHz)'].max()
    max_peak_diff_unc = peak_diff_df.loc[
        peak_diff_df['Peak Difference (GHz)'] == max_peak_diff,
        'Peak Difference_unc (GHz)'
    ].values[0]

    J = max_peak_diff / 2
    J_unc = max_peak_diff_unc / 2

    # Add scaled columns
    peak_locs_df['Scaled Kappa'] = peak_locs_df['Kappa'] / J
    peak_locs_df['Scaled Kappa_unc'] = np.abs(peak_locs_df['Scaled Kappa']) * np.sqrt(
        (peak_locs_df['Kappa_unc'] / peak_locs_df['Kappa']) ** 2 + (J_unc / J) ** 2
    )

    # Now we do a scatter plot: X = scaled Kappa, Y = PeakFreqGHz
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        peak_locs_df['Scaled Kappa'], peak_locs_df['PeakFreqGHz'],
        xerr=peak_locs_df['Scaled Kappa_unc'], yerr=peak_locs_df['PeakFreqUncGHz'],
        fmt='o', ecolor='red', capsize=4, label='Normal Mode Peaks', markersize=4
    )

    ax.set_xlabel('K / J')
    ax.set_ylabel('Peak Frequency (GHz)')
    ax.set_title('Normal Mode Peak Locations vs K/J')
    ax.grid(True)
    ax.legend()

    # set y lim based on min and max values, from the data
    y_min = peak_locs_df['PeakFreqGHz'].min()
    y_max = peak_locs_df['PeakFreqGHz'].max()
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"peak_locations_vs_kappa_{experiment_id}.png")
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"Saved peak locations vs. K/J plot to {plot_path}")


if __name__ == "__main__":
    db_path = './databases/THE_SECOND_MANUAL.db'
    # db_path = '../databases/12_11_big_long.db'
    engine = get_engine(db_path)
    experiment_id = 'ABCD'
    # experiment_id = 'a7e52acf-5ea1-41c7-92ca-fab6b1381c6a'
    freq_min = 1e9
    freq_max = 99e9
    voltage_min = -2.8
    voltage_max = 0
    readout_types = ['normal', 'cavity', 'yig']

    all_results = {}


    def __default_peak_finding_function(frequencies, powers):
        peaks_indices, _ = find_peaks(powers, height=-30, prominence=0.1, distance=50)
        peak_freqs = frequencies[peaks_indices]
        peak_powers = powers[peaks_indices]
        return peak_freqs, peak_powers


    for rt in readout_types:
        power_grid, voltages, frequencies, settings = get_data_from_db(engine, experiment_id, rt, freq_min, freq_max,
                                                                       voltage_min, voltage_max)
        if power_grid is None:
            continue
        peaks_df = process_all_traces(power_grid, voltages, frequencies, __default_peak_finding_function, experiment_id,
                                      rt)
        all_results[rt] = (
            pd.read_csv(
                os.path.join("csv", f"{experiment_id}_{rt}", f"lorentzian_fit_results_{experiment_id}_{rt}.csv")),
            settings, power_grid, voltages, frequencies
        )

    # Compute Kappa & Delta
    cavity_df = all_results['cavity'][0]
    yig_df = all_results['yig'][0]
    merged_df = compute_capital_kappa_and_capital_delta(cavity_df, yig_df)

    database_folder = f"{os.path.basename(db_path)}_kappa_colorplots"
    plots_folder = os.path.join(PLOTS_FOLDER, database_folder)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    for rt in readout_types:
        if rt not in all_results:
            continue
        df_fits, settings, power_grid, voltages, frequencies = all_results[rt]
        plot_kappa_colorplot(power_grid, voltages, frequencies, merged_df, experiment_id, rt, settings,
                             folder=plots_folder)

    # Re-plot normal mode traces by Kappa
    normal_df = all_results['normal'][0]
    replot_normal_mode_traces_with_kappa(normal_df, merged_df, all_results['normal'][2], all_results['normal'][3],
                                         all_results['normal'][4], experiment_id)

    # Save and plot peak differences vs. Kappa for normal mode double lorentzian
    output_folder = os.path.join("csv", f"{experiment_id}_peak_differences")
    peak_diff_df, peak_diff_file = save_peak_differences_vs_kappa(
        normal_df, merged_df, all_results['normal'][2], all_results['normal'][3], all_results['normal'][4],
        output_folder, experiment_id
    )

    peak_locations_output_folder = os.path.join("csv", f"{experiment_id}_peak_locations")
    plot_peak_locations_vs_kappa(
        normal_df, merged_df, all_results['normal'][2],
        all_results['normal'][3], all_results['normal'][4],
        peak_locations_output_folder, experiment_id, peak_diff_file
    )
