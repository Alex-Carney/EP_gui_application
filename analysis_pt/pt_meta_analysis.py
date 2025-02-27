import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sqlalchemy import create_engine

import pt_config_loader as cl
import pt_simulation as pt_sim
import pt_plotting as pt_plot
import pt_voltage_data_loader as pt_data
import pt_fitting as pt_fit

# ------------------ COMMON MACROS / CONFIGS ------------------
CONFIG_NAME = "cab_nr2"
CAV_YIG_DEBUG = False
NR_DEBUG = False

NUM_SIMULATION_SHOTS = 1250
PLOTS_FOLDER = "plots"
METADB_FOLDER = "metadatabase"
SEED = 12345
CONFIG_PATH = "pt_expr_config.json"
np.random.seed(SEED)


def flatten_value(val):
    """
    Convert lists, dicts, and numpy arrays to JSON strings.
    Other types are returned unchanged.
    """
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    elif isinstance(val, np.ndarray):
        return json.dumps(val.tolist())
    else:
        return val


def flatten_dict(d):
    """
    For a dictionary d, ensure that each value is flattened.
    """
    return {k: flatten_value(v) for k, v in d.items()}


def save_metadata(metadata, config_name):
    """
    Saves the collected metadata into an SQLite database using SQLAlchemy.
    DataFrames are stored directly.
    For dicts (or other types), we flatten any nested dict/list/ndarray
    values into JSON strings before storing.
    """
    os.makedirs(METADB_FOLDER, exist_ok=True)
    db_file = os.path.join(METADB_FOLDER, f"{config_name}.db")
    engine = create_engine(f"sqlite:///{db_file}")

    for key, value in metadata.items():
        if isinstance(value, pd.DataFrame):
            value.to_sql(key, engine, if_exists="replace", index=False)
        elif isinstance(value, dict):
            if key == "nr_fit_dict":
                # For nr_fit_dict, each record is stored as a row.
                nr_data = []
                for voltage, data in value.items():
                    row = {"voltage": voltage}
                    for k, v in data.items():
                        row[k] = flatten_value(v)
                    nr_data.append(row)
                pd.DataFrame(nr_data).to_sql("nr_fit", engine, if_exists="replace", index=False)
            else:
                flat_value = flatten_dict(value)
                pd.DataFrame([flat_value]).to_sql(key, engine, if_exists="replace", index=False)
        else:
            pd.DataFrame({"value": [flatten_value(value)]}).to_sql(key, engine, if_exists="replace", index=False)


def main():
    # ------------------ LOAD CONFIGURATION ------------------
    config: cl.PTExperimentConfig = cl.load_config(CONFIG_PATH, CONFIG_NAME)
    experiment_id = config.experiment_id
    db_path = config.db_path
    readout_type = config.readout_type
    colorplot_freq_min = config.colorplot_freq_min
    colorplot_freq_max = config.colorplot_freq_max
    cavity_freq_min = config.cavity_freq_min
    cavity_freq_max = config.cavity_freq_max
    yig_freq_min = config.yig_freq_min
    yig_freq_max = config.yig_freq_max
    voltage_min = config.voltage_min
    voltage_max = config.voltage_max

    # ------------------ LOAD DATA FROM DATABASE ------------------
    cavity_loader = pt_data.PTVoltageDataLoader(
        db_path, experiment_id, "cavity",
        cavity_freq_min, cavity_freq_max,
        voltage_min, voltage_max, independent_var="set_voltage"
    )
    yig_loader = pt_data.PTVoltageDataLoader(
        db_path, experiment_id, "yig",
        yig_freq_min, yig_freq_max,
        voltage_min, voltage_max, independent_var="set_voltage"
    )
    pt_loader = pt_data.PTVoltageDataLoader(
        db_path, experiment_id, readout_type,
        colorplot_freq_min, colorplot_freq_max,
        voltage_min, voltage_max, independent_var="set_voltage"
    )
    cavity_power, cavity_voltages, cavity_freqs, cavity_settings = cavity_loader.load_data()
    yig_power, yig_voltages, yig_freqs, yig_settings = yig_loader.load_data()
    pt_power, nr_voltages, pt_freqs, nr_settings = pt_loader.load_data()

    if cavity_power is None or yig_power is None or pt_power is None:
        print("Error: One or more datasets are missing. Exiting.")
        return

    # ------------------ INITIALIZE METADATA STORAGE ------------------
    # Convert raw arrays to JSON strings so that they are safely stored.
    metadata = {}
    metadata["raw_cavity"] = {
        "power": json.dumps(cavity_power.tolist()),
        "voltages": json.dumps(cavity_voltages.tolist()),
        "freqs": json.dumps(cavity_freqs.tolist()),
        "settings": json.dumps(cavity_settings),
    }
    metadata["raw_yig"] = {
        "power": json.dumps(yig_power.tolist()),
        "voltages": json.dumps(yig_voltages.tolist()),
        "freqs": json.dumps(yig_freqs.tolist()),
        "settings": json.dumps(yig_settings),
    }
    metadata["raw_pt"] = {
        "power": json.dumps(pt_power.tolist()),
        "voltages": json.dumps(nr_voltages.tolist()),
        "freqs": json.dumps(pt_freqs.tolist()),
        "settings": json.dumps(nr_settings),
    }

    # ------------------ RAW COLORPLOT GENERATION ------------------
    raw_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP_raw")
    pt_plot.plot_raw_colorplot(pt_power, nr_voltages, pt_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type=readout_type)
    pt_plot.plot_raw_colorplot(yig_power, yig_voltages, yig_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type="yig")
    pt_plot.plot_raw_colorplot(cavity_power, cavity_voltages, cavity_freqs, experiment_id, nr_settings,
                               raw_folder, readout_type="cavity")

    # ------------------ OPTIONAL CAVITY/YIG DEBUG TRACES ------------------
    if CAV_YIG_DEBUG:
        debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
        for rt, power, voltages, freqs in [
            ("cavity", cavity_power, cavity_voltages, cavity_freqs),
            ("yig", yig_power, yig_voltages, yig_freqs)
        ]:
            for i, v in enumerate(voltages):
                trace = power[i, :]
                fit_data = pt_fit.fit_trace(v, freqs, trace, peak_selection_option="amplitude")
                pt_plot.plot_individual_trace(v, freqs, trace, rt, debug_folder, fit_data)

    # ------------------ ANALYZE CAVITY & YIG TRACES ------------------
    cavity_results = pt_fit.process_all_traces(
        cavity_power, cavity_voltages, cavity_freqs,
        peak_selection_option="amplitude"
    )
    yig_results = pt_fit.process_all_traces(
        yig_power, yig_voltages, yig_freqs,
        apply_pre_smoothing=True, peak_selection_option="amplitude"
    )
    cavity_df = pd.DataFrame(cavity_results)
    yig_df = pd.DataFrame(yig_results)
    K_df = pt_fit.compute_K(cavity_df, yig_df)

    metadata["cavity"] = cavity_df
    metadata["yig"] = yig_df
    metadata["K"] = K_df

    # Build maps for simulation and further analysis
    cavity_map = cavity_df.set_index("voltage")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    yig_map = yig_df.set_index("voltage")[["omega", "kappa", "omega_unc", "kappa_unc"]].to_dict(orient="index")
    K_map = K_df.set_index("voltage")["K"].to_dict()

    # ------------------ NR THEORY SIMULATIONS AND FITTING ------------------
    nr_fit_dict = {}
    for i, v in enumerate(nr_voltages):
        if v not in K_map or v not in cavity_map or v not in yig_map:
            continue

        trace = pt_power[i, :]

        # Unpack cavity and YIG parameters
        omega_c = cavity_map[v]["omega"]
        kappa_c = cavity_map[v]["kappa"]
        omega_c_unc = cavity_map[v]["omega_unc"]
        kappa_c_unc = cavity_map[v]["kappa_unc"]
        omega_y = yig_map[v]["omega"]
        kappa_y = yig_map[v]["kappa"]
        omega_y_unc = yig_map[v]["omega_unc"]
        kappa_y_unc = yig_map[v]["kappa_unc"]

        all_peak_freqs = []
        drive = (1, 0)
        readout = (1, 0) if readout_type == "normal" else (0, 1)
        fast_func = pt_sim.setup_fast_simulation(drive=drive, readout=readout)
        for shot in range(NUM_SIMULATION_SHOTS):
            result = pt_sim.run_single_theory_shot_fast(
                fast_func,
                config.optimal_J, config.optimal_J_unc,
                omega_c, omega_c_unc,
                omega_y, omega_y_unc,
                kappa_c, kappa_c_unc,
                kappa_y, kappa_y_unc,
                pt_freqs,
            )
            all_peak_freqs.append(result)
        all_peak_freqs = np.array(all_peak_freqs)
        sim_trace_avg = pt_sim.simulate_trace(
            config.optimal_J, omega_c, omega_y, kappa_c, kappa_y,
            pt_freqs / 1e9,
            readout=[1, 0] if readout_type == "normal" else [0, 1]
        )
        sim_peaks_idx_avg, _ = find_peaks(sim_trace_avg, prominence=0.0001)
        lowest_peak_range = (np.nanmin(all_peak_freqs[:, 0]), np.nanmax(all_peak_freqs[:, 0]))
        highest_peak_range = (np.nanmin(all_peak_freqs[:, 1]), np.nanmax(all_peak_freqs[:, 1]))
        fit_data = pt_fit.theory_supported_PT_fit(
            v, pt_freqs, trace, sim_trace_avg, sim_peaks_idx_avg,
            config.amplitude_threshold_overfitting
        )
        nr_fit_dict[v] = {
            "trace": trace,
            "sim_trace": sim_trace_avg,
            "sim_peaks_idx": sim_peaks_idx_avg,
            "fit_data": fit_data,
            "theory_lower_min": lowest_peak_range[0],
            "theory_lower_max": lowest_peak_range[1],
            "theory_upper_min": highest_peak_range[0],
            "theory_upper_max": highest_peak_range[1],
        }
        if NR_DEBUG:
            debug_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_debug")
            order_prefix = f"{i + 1:03d}_"
            pt_plot.plot_individual_trace(
                v,
                pt_freqs,
                trace,
                readout_type,
                debug_folder,
                fit_data,
                K_val=K_map[v],
                order_prefix=order_prefix,
                simulated_trace=sim_trace_avg,
                simulated_trace_peak_idxs=sim_peaks_idx_avg,
                simulated_vertical_offset=config.simulated_vertical_offset,
                peak1_lower_bound=lowest_peak_range[0],
                peak1_upper_bound=lowest_peak_range[1],
                peak2_lower_bound=highest_peak_range[0],
                peak2_upper_bound=highest_peak_range[1],
            )
    metadata["nr_fit_dict"] = nr_fit_dict

    # ------------------ NR DETUNING COLORPLOT ------------------
    output_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_NR_EP")
    pt_plot.plot_K_colorplot(pt_power, nr_voltages, pt_freqs, K_df, experiment_id, nr_settings, output_folder)

    # ------------------ NR PEAK LOCATION OVERLAY PLOT ------------------
    overlay_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_nr_peaks")
    os.makedirs(overlay_folder, exist_ok=True)
    print("Plotting NR peak locations vs. Detuning overlay...")
    exp_data = {"K_list": [], "peak_list": [], "peak_unc_list": [],
                "linewidth_list": [], "splitting_ratio_list": [], "K_double_list": []}
    theory_data = {"theory_K_list": [], "theory_lower_min_list": [],
                   "theory_lower_max_list": [], "theory_upper_min_list": [],
                   "theory_upper_max_list": []}
    for v, data in nr_fit_dict.items():
        fit_data = data.get("fit_data")
        if fit_data is None:
            continue
        K_val = K_map[v]
        if fit_data.get("fit_type") == "single":
            exp_data["K_list"].append(K_val)
            exp_data["peak_list"].append(fit_data.get("omega"))
            exp_data["peak_unc_list"].append(fit_data.get("omega_unc", 0.0))
            exp_data["linewidth_list"].append(fit_data.get("linewidth", 0.0))
        elif fit_data.get("fit_type") == "double":
            exp_data["K_list"].extend([K_val, K_val])
            exp_data["peak_list"].extend([fit_data.get("peak1"), fit_data.get("peak2")])
            exp_data["peak_unc_list"].extend([fit_data.get("peak1_unc", 0.0),
                                              fit_data.get("peak2_unc", 0.0)])
            exp_data["linewidth_list"].extend([fit_data.get("peak1_linewidth", 0.0),
                                               fit_data.get("peak2_linewidth", 0.0)])
            splitting = abs(fit_data.get("peak1") - fit_data.get("peak2"))
            avg_linewidth = (fit_data.get("peak1_linewidth") + fit_data.get("peak2_linewidth")) / 2
            ratio = splitting / avg_linewidth if avg_linewidth > 0 else 0
            exp_data["splitting_ratio_list"].append(ratio)
            exp_data["K_double_list"].append(K_val)
        theory_data["theory_K_list"].append(K_val)
        theory_data["theory_lower_min_list"].append(data["theory_lower_min"])
        theory_data["theory_lower_max_list"].append(data["theory_lower_max"])
        theory_data["theory_upper_min_list"].append(data["theory_upper_min"])
        theory_data["theory_upper_max_list"].append(data["theory_upper_max"])
    if not exp_data["peak_list"]:
        print("No NR peaks extracted for the overlay plot.")
        return
    K_array = np.array(exp_data["K_list"])
    peak_array = np.array(exp_data["peak_list"])
    peak_unc_array = np.array(exp_data["peak_unc_list"])
    linewidth_array = np.array(exp_data["linewidth_list"])
    splitting_ratio_array = np.array(exp_data["splitting_ratio_list"])
    detuning_double_array = np.array(exp_data["K_double_list"])
    theory_K_array = np.array(theory_data["theory_K_list"])
    theory_lower_min_array = np.array(theory_data["theory_lower_min_list"])
    theory_lower_max_array = np.array(theory_data["theory_lower_max_list"])
    theory_upper_min_array = np.array(theory_data["theory_upper_min_list"])
    theory_upper_max_array = np.array(theory_data["theory_upper_max_list"])
    sort_idx = np.argsort(theory_K_array)
    theory_K_array = theory_K_array[sort_idx]
    theory_lower_min_array = theory_lower_min_array[sort_idx]
    theory_lower_max_array = theory_lower_max_array[sort_idx]
    theory_upper_min_array = theory_upper_min_array[sort_idx]
    theory_upper_max_array = theory_upper_max_array[sort_idx]

    average_kappa = np.mean(cavity_df["kappa"].values)
    cavity_freq = np.mean(cavity_df["omega"].values)
    yig_freq = np.mean(yig_df["omega"].values)
    delta_values = cavity_df["omega"].values - yig_df["omega"].values

    print("\nAverage kappa_C:", average_kappa)
    print("Average delta:", np.mean(delta_values))
    print("Standard deviation of delta:", np.std(delta_values))
    print("Cavity freq average:", cavity_freq)
    print("Yig freq average:", yig_freq)
    print("Max yig kappa:", np.max(yig_df["kappa"].values))
    print("Min yig kappa:", np.min(yig_df["kappa"].values))
    print("Average yig kappa:", np.mean(yig_df["kappa"].values))

    pt_plot.plot_final_peak_plot(
        theory_K_array=theory_K_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        optimal_J=config.optimal_J,
        K_array=K_array,
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

    pt_plot.plot_final_peak_plot(
        theory_K_array=theory_K_array,
        theory_lower_min_array=theory_lower_min_array,
        theory_lower_max_array=theory_lower_max_array,
        optimal_J=config.optimal_J,
        K_array=K_array,
        peak_array=peak_array,
        experiment_id=experiment_id,
        overlay_folder=overlay_folder,
        theory_upper_min_array=theory_upper_min_array,
        theory_upper_max_array=theory_upper_max_array,
        peak_unc_array=linewidth_array,
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end,
        errorbar_color="cyan",
        filename_prepend="fwhm_"
    )

    pt_plot.plot_linewidth_vs_K(
        K_array, linewidth_array, config.optimal_J, linewidth_unc_array=None,
        experiment_id=experiment_id, output_folder=overlay_folder,
        errorbar_color="blue",
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end
    )

    pt_plot.plot_sum_linewidth_vs_K(
        K_array, linewidth_array, config.optimal_J, linewidth_unc_array=None,
        experiment_id=experiment_id, output_folder=overlay_folder,
        errorbar_color="blue",
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end
    )

    pt_plot.plot_splitting_ratio_vs_K(
        detuning_double_array,
        splitting_ratio_array,
        optimal_J=config.optimal_J,
        experiment_id=experiment_id,
        output_folder=overlay_folder,
        errorbar_color="magenta",
        overlap_region_start=config.overlap_region_start,
        overlap_region_end=config.overlap_region_end
    )

    fig3_folder = os.path.join(PLOTS_FOLDER, f"{experiment_id}_FIG3")
    ep_threshold = 2 * config.optimal_J
    single_peak_frequencies = []
    for v, data in nr_fit_dict.items():
        fit_data = data.get("fit_data")
        if fit_data is None:
            continue
        if fit_data.get("fit_type") == "single":
            K_val = K_map.get(v)
            if K_val is None:
                continue
            if K_val < ep_threshold:
                omega_val = fit_data.get("omega")
                if omega_val is not None:
                    single_peak_frequencies.append(omega_val)
    if single_peak_frequencies:
        avg_single_peak_frequency = np.mean(single_peak_frequencies)
        print("Average frequency for single peaks with detuning less than EP threshold:",
              avg_single_peak_frequency)
        pt_plot.plot_fig3_PT(
            pt_power, nr_voltages, pt_freqs, K_df,
            experiment_id, nr_settings, avg_single_peak_frequency, fig3_folder,
            optimal_J=config.optimal_J, frequency_radius=0.0006
        )
    else:
        print("WARNING: No single peaks with detuning less than EP threshold were found.")

    # ------------------ STORE CONFIGURATION METADATA ------------------
    metadata["config"] = {
        "CONFIG_NAME": CONFIG_NAME,
        "experiment_id": experiment_id,
        "db_path": db_path,
        "readout_type": readout_type,
        "colorplot_freq_min": colorplot_freq_min,
        "colorplot_freq_max": colorplot_freq_max,
        "cavity_freq_min": cavity_freq_min,
        "cavity_freq_max": cavity_freq_max,
        "yig_freq_min": yig_freq_min,
        "yig_freq_max": yig_freq_max,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "NUM_SIMULATION_SHOTS": NUM_SIMULATION_SHOTS,
        "SEED": SEED,
    }

    # ------------------ SAVE ALL METADATA TO THE DATABASE ------------------
    save_metadata(metadata, CONFIG_NAME)


if __name__ == "__main__":
    main()
