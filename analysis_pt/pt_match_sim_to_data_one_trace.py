# main_overlay_regression_improved.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from lmfit.models import LorentzianModel  # Used in the double Lorentzian fit

# Import your simulation routines and model definitions.
# (Adjust the import paths as needed.)
from symbolic_module import (
    setup_symbolic_equations,
    ModelParams,
    get_steady_state_response_transmission,
    compute_photon_numbers_transmission,
)

FREQUENCY_RADIUS = 0.003

# ------------------ USER-DEFINED N RANGE ------------------
# Instead of using a command line argument, define the range of N here.
N_min = 1
N_max = 50  # change these values as desired


# ------------------ NR DOUBLE LORENTZIAN FIT FUNCTION ------------------
def double_lorentzian_fit_PT(x, y, guess1, guess2):
    """
    Fit a double Lorentzian model to data given initial guesses guess1 and guess2.
    (Note: The y data must be in linear units.)
    """
    lz1 = LorentzianModel(prefix="lz1_")
    lz2 = LorentzianModel(prefix="lz2_")
    mod = lz1 + lz2
    sigma_guess = 0.001
    amp_guess = y.max() * np.pi * sigma_guess
    pars = mod.make_params()
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


# ------------------ DATABASE & SIMULATION SETUP ------------------
# Hardcoded values:
db_path = r"C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\databases\12_12_ovn_one.db"
experiment_id = "adf1420f-9206-404b-8c8a-a680dc9d41b2"
readout_type = "normal"

# Create engine.
engine = create_engine(f"sqlite:///{db_path}")

# --- Query DB for settings (done once) ---
settings_query = f"""
SELECT DISTINCT set_loop_att, set_loopback_att,
       set_yig_fb_phase_deg, set_yig_fb_att,
       set_cavity_fb_phase_deg, set_cavity_fb_att,
       omega_C, omega_Y, kappa_C, kappa_Y, Delta, K
FROM expr
WHERE experiment_id = '{experiment_id}'
  AND readout_type = '{readout_type}'
"""
settings_df = pd.read_sql_query(settings_query, engine)
if settings_df.empty:
    raise ValueError("No settings found for the given experiment and readout type.")

# Compute the absolute detuning and sort settings in descending order.
settings_df["K"] = abs(settings_df["kappa_C"] - settings_df["kappa_Y"])
# For K , ascending. For Delta, descending.
sorted_settings_df = settings_df.sort_values(by="K", ascending=True)

# Setup symbolic equations (they are independent of J).
symbols_dict = setup_symbolic_equations()

# Arrays to store the optimal J values and amplitude differences for each trace.
optimal_J_values = []
amplitude_differences = []  # New list to hold the amplitude differences (in dBm)

# ------------------ LOOP OVER N VALUES ------------------
for N in range(N_min, N_max + 1):
    print("\n" + "=" * 40)
    print(f"Processing N = {N} ...")
    if N > len(sorted_settings_df):
        print(f"N = {N} is larger than the number of available settings ({len(sorted_settings_df)}). Skipping.")
        continue

    # Select the Nth best match (largest K first)
    best_match = sorted_settings_df.iloc[N - 1]
    db_cavity_kappa = best_match["kappa_C"]  # in GHz
    db_yig_kappa = best_match["kappa_Y"]  # in GHz
    db_gamma_vec = np.array([db_cavity_kappa, db_yig_kappa])

    db_omega_C = best_match["omega_C"]
    db_omega_Y = best_match["omega_Y"]
    print("Best match from DB settings (based on K):")
    print(best_match)

    # --- Query DB for measured trace ---
    mean_freq = (db_omega_C + db_omega_Y) / 2  # GHz
    freq_min = (mean_freq - FREQUENCY_RADIUS) * 1e9  # Hz
    freq_max = (mean_freq + FREQUENCY_RADIUS) * 1e9  # Hz

    data_query = f"""
    SELECT *
    FROM expr
    WHERE experiment_id = '{experiment_id}'
      AND readout_type = '{readout_type}'
      AND ABS(kappa_C - {db_cavity_kappa}) < 1e-05
      AND ABS(kappa_Y - {db_yig_kappa}) < 1e-05
      AND frequency_hz BETWEEN {freq_min} AND {freq_max}
    ORDER BY frequency_hz
    """
    print("Data query:")
    print(data_query)
    data_df = pd.read_sql_query(data_query, engine)
    if data_df.empty:
        raise ValueError("No trace data found matching the best-fit settings.")

    # Pivot the data so that each row corresponds to a current and each column to a frequency.
    pivot_table = data_df.pivot_table(index="set_voltage", columns="frequency_hz",
                                      values="power_dBm", aggfunc="first")
    measured_freqs = pivot_table.columns.values / 1e9  # Convert Hz to GHz.
    measured_power = pivot_table.values
    # Use one trace (e.g., the first/lowest current) for overlay.
    trace_index = 0
    measured_trace_dbm = measured_power[trace_index, :]

    # ------------------ EXPERIMENTAL PEAK FINDING ------------------
    peaks_exp, props = find_peaks(measured_trace_dbm, height=-30, prominence=0.025, distance=25)
    if len(peaks_exp) == 0:
        raise ValueError("No peaks found in experimental data for initial guesses.")
    elif len(peaks_exp) == 1:
        initial_guess_freqs = np.array([measured_freqs[peaks_exp[0]],
                                        measured_freqs[peaks_exp[0]] + 0.001])
    else:
        prominences = props["prominences"]
        sorted_idx = np.argsort(prominences)[::-1]  # Highest prominence first
        best_two = peaks_exp[sorted_idx[:2]]
        best_two = np.sort(best_two)
        initial_guess_freqs = measured_freqs[best_two]
    print("Initial experimental peak guesses (GHz):", initial_guess_freqs)

    # Convert the measured trace from dBm to linear scale for the fitting.
    measured_trace_linear = 10 ** (measured_trace_dbm / 10)

    # ------------------ DOUBLE LORENTZIAN FIT ON EXPERIMENTAL TRACE ------------------
    double_fit_result = double_lorentzian_fit_PT(measured_freqs, measured_trace_linear,
                                                 initial_guess_freqs[0], initial_guess_freqs[1])
    if double_fit_result is None:
        raise ValueError("Double Lorentzian fit failed for experimental data.")

    # Extract the fitted peak centers (in GHz).
    fitted_peak1 = double_fit_result.params["lz1_center"].value
    fitted_peak2 = double_fit_result.params["lz2_center"].value
    exp_fitted_peaks = np.sort(np.array([fitted_peak1, fitted_peak2]))
    print("Experimental fitted peaks (GHz) from double Lorentzian fit:", exp_fitted_peaks)

    # ------------------ Compute amplitude difference from fitted curve ------------------
    # Evaluate the double Lorentzian fit at the two fitted centers.
    fitted_y_linear_peaks = double_fit_result.eval(x=exp_fitted_peaks)
    # Convert the evaluated linear values to dBm.
    fitted_y_dbm_peaks = 10 * np.log10(fitted_y_linear_peaks)
    # Compute the absolute difference in dBm.
    amp_diff = abs(fitted_y_dbm_peaks[0] - fitted_y_dbm_peaks[1])
    print("Absolute amplitude difference between the two peaks (dBm): {:.6f}".format(amp_diff))
    amplitude_differences.append(amp_diff)
    # ------------------------------------------------------------------------------------

    # For plotting: evaluate the fit over a dense grid and convert back to dBm.
    x_fit = np.linspace(measured_freqs.min(), measured_freqs.max(), 1000)
    y_fit_linear = double_fit_result.eval(x=x_fit)
    y_fit_dbm = 10 * np.log10(y_fit_linear)

    # ------------------ SIMULATION TRACE & COST FUNCTION ------------------
    lo_freqs = np.linspace(mean_freq - FREQUENCY_RADIUS, mean_freq + FREQUENCY_RADIUS, 1000)  # GHz for simulation
    VERTICAL_OFFSET = 10.5  # Vertical offset applied to simulated trace


    def simulate_trace(J_val):
        """
        Given a trial J_val, simulate the trace.
        """
        sim_params = ModelParams(
            J_val=J_val,
            g_val=0,
            cavity_freq=db_omega_C,
            w_y=db_omega_Y,
            gamma_vec=db_gamma_vec,
            drive_vector=np.array([1, 0]),
            readout_vector=np.array([1, 0]),
            phi_val=np.deg2rad(0),
        )
        nr_ep_ss_eqn = get_steady_state_response_transmission(symbols_dict, sim_params)
        photon_numbers_sim = compute_photon_numbers_transmission(nr_ep_ss_eqn, lo_freqs)
        sim_trace = np.log10(photon_numbers_sim) - VERTICAL_OFFSET
        return sim_trace


    def cost_function(J_val):
        """
        For a given J_val, simulate the trace, find its two peaks, and return
        the sum of squared differences between the simulated peak frequencies and
        the experimentally fitted peak frequencies.
        """
        sim_trace = simulate_trace(J_val)
        sim_peaks_idx, _ = find_peaks(sim_trace, prominence=0.001)
        if len(sim_peaks_idx) < 2:
            return 1e6  # Penalize if fewer than 2 peaks are found.
        if len(sim_peaks_idx) > 2:
            # Choose the two highest peaks.
            peak_vals = sim_trace[sim_peaks_idx]
            highest = np.argsort(peak_vals)[-2:]
            sim_peaks_idx = sim_peaks_idx[highest]
        sim_peaks_freq = np.sort(lo_freqs[sim_peaks_idx])
        cost = np.sum((sim_peaks_freq - exp_fitted_peaks) ** 2)
        return cost


    # Optimize J_value.
    initial_J = 0.0005
    bounds = (0.0005, .005)
    opt_options = {
        'maxiter': 1000,  # Increase maximum iterations
        'xatol': 1e-8,  # Tighten the tolerance on the parameter
    }
    res = minimize_scalar(cost_function, bounds=bounds, method='bounded', options=opt_options)
    best_J = res.x
    print(f"Optimal J_value found: {best_J:.6f} with cost {res.fun:.3e}")
    optimal_J_values.append(best_J)

    # Recompute simulation trace with the optimal J.
    simulated_trace_opt = simulate_trace(best_J)
    simulated_trace_initial = simulate_trace(initial_J)

    # # Extreme debug: plot the simulated traces
    # plt.figure()
    # plt.plot(lo_freqs, simulated_trace_opt, label=f"Simulated Trace (J = {best_J:.6f})")
    # plt.plot(lo_freqs, simulated_trace_initial, label=f"Simulated Trace (J = {initial_J:.6f})")
    # plt.xlabel("Frequency (GHz)")
    # plt.ylabel("Power (dBm) / log10(Photon Number)")
    # plt.title("Simulated Traces")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('EXTREME_simulated_traces.png', dpi=400)



    # Find simulated peaks.
    sim_peaks_idx, _ = find_peaks(simulated_trace_opt, prominence=0.001)
    if len(sim_peaks_idx) == 0:
        raise ValueError("No peaks found in simulated trace.")
    elif len(sim_peaks_idx) == 1:
        sim_peaks_idx = np.append(sim_peaks_idx, sim_peaks_idx[0] + 1)
    else:
        if len(sim_peaks_idx) > 2:
            peak_vals = simulated_trace_opt[sim_peaks_idx]
            highest = np.argsort(peak_vals)[-2:]
            sim_peaks_idx = sim_peaks_idx[highest]
    simulated_peaks_freq = np.sort(lo_freqs[sim_peaks_idx])
    print("Simulated peaks (GHz):", simulated_peaks_freq)

    # ------------------ PLOTTING THE OVERLAY WITH MARKERS ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(measured_freqs, measured_trace_dbm, label="Measured Trace (Exp)", color="blue")
    plt.plot(lo_freqs, simulated_trace_opt, label=f"Simulated Trace (J = {best_J:.6f})",
             color="red", linestyle="--")
    plt.plot(lo_freqs, simulated_trace_initial, label=f"Simulated Trace (J = {initial_J:.6f})",
             color="orange", linestyle="--")
    plt.plot(x_fit, y_fit_dbm, label="Double Lorentzian Fit (Exp)", color="green", linestyle=":")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power (dBm) / log10(Photon Number)")
    plt.title(f"Overlay: Simulated vs. Measured Trace\n(Detuning = {db_cavity_kappa - db_yig_kappa} GHz)")
    plt.grid(True)

    # --- Mark the experimental initial guesses ---
    initial_guess_indices = [np.argmin(np.abs(measured_freqs - guess)) for guess in initial_guess_freqs]
    plt.plot(measured_freqs[initial_guess_indices], measured_trace_dbm[initial_guess_indices],
             'm*', markersize=12, label="Exp Initial Guesses")

    # --- Mark the experimental fitted peaks (from double Lorentzian) ---
    plt.plot(exp_fitted_peaks, fitted_y_dbm_peaks, 'c*', markersize=12, label="Exp Fitted Peaks")

    # --- Mark the simulated peaks ---
    simulated_y = np.array([simulated_trace_opt[np.argmin(np.abs(lo_freqs - f))] for f in simulated_peaks_freq])
    plt.plot(simulated_peaks_freq, simulated_y, 'r*', markersize=12, label="Simulated Peaks")

    # --- Mark the peaks for the initial J value ---
    initial_peaks_idx, _ = find_peaks(simulated_trace_initial, prominence=0.001)
    initial_peaks_freq = lo_freqs[initial_peaks_idx]
    initial_peaks_y = np.array([simulated_trace_initial[np.argmin(np.abs(lo_freqs - f))] for f in initial_peaks_freq])
    plt.plot(initial_peaks_freq, initial_peaks_y, 'y*', markersize=12, label="Simulated Peaks (Initial J)")

    plt.legend()
    plt.tight_layout()
    # Create the folder for output plots
    output_folder = os.path.join("plots", experiment_id)
    os.makedirs(output_folder, exist_ok=True)

    # Construct the filename using the current N value.
    output_filename = os.path.join(output_folder, f"{N}_overlay_trace_with_regression_improved.png")

    # Save the plot and close the figure.
    plt.savefig(output_filename, dpi=400)
    plt.close()
    print(f"Overlay plot saved to {output_filename}")

# ------------------ PRINT OPTIMAL J VALUES SUMMARY ------------------
if optimal_J_values:
    optimal_J_array = np.array(optimal_J_values)
    mean_J = np.mean(optimal_J_array)
    stderr_J = np.std(optimal_J_array)
    print("\n" + "=" * 40)
    print("Optimal J values for each N in the range:")
    print(optimal_J_values)
    print(f"Average optimal J: {mean_J:.6f}")
    print(f"Median optimal J: {np.median(optimal_J_array):.6f}")
    print(f"Standard error: {stderr_J:.6f}")
else:
    print("No optimal J values were found.")

# ------------------ PRINT AMPLITUDE DIFFERENCES SUMMARY ------------------
if amplitude_differences:
    amplitude_differences_array = np.array(amplitude_differences)
    mean_amp_diff = np.mean(amplitude_differences_array)
    stderr_amp_diff = np.std(amplitude_differences_array, ddof=1) / np.sqrt(len(amplitude_differences_array))
    print("\n" + "=" * 40)
    print("Absolute amplitude differences between the two fitted peaks (dBm) for each N:")
    print(amplitude_differences)
    print(f"Average amplitude difference: {mean_amp_diff:.6f} dBm")
    print(f"Standard error: {stderr_amp_diff:.6f} dBm")
else:
    print("No amplitude differences were computed.")
