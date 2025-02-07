# main_overlay_regression_improved.py
import os
import sys
import argparse
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

FREQUENCY_RADIUS = 0.004
DEFAULT_N = 1

# ------------------ PARSE COMMAND LINE ARGUMENTS ------------------
parser = argparse.ArgumentParser(
    description="Overlay measured and simulated trace with regression."
)
parser.add_argument(
    "-N",
    "--Nth",
    type=int,
    default=DEFAULT_N,  # Default value (if no argument is supplied, this value is used)
    help="Select the Nth row (largest detuning) from the DB settings (default: DEFAULT_N).",
)
args = parser.parse_args()
N = args.Nth  # This value now comes from the command line if provided.
print(f"Using N = {N} for selecting the Nth best match from the DB settings.")


# ------------------ NR DOUBLE LORENTZIAN FIT FUNCTION ------------------
def double_lorentzian_fit_NR(x, y, guess1, guess2):
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
db_path = r"C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\databases\NR_SUN_NITE_PPP_FINE.db"
experiment_id = "294962de-dd80-49c6-81b5-394ae97b5838"
readout_type = "nr"

# Create engine.
engine = create_engine(f"sqlite:///{db_path}")

# --- Query DB for settings ---
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

# Select the row with the largest absolute detuning |ω_C - ω_Y|
settings_df["detuning"] = abs(settings_df["omega_C"] - settings_df["omega_Y"])
# Sort the DataFrame by 'detuning' in descending order
sorted_settings_df = settings_df.sort_values(by="detuning", ascending=False)
# Select the Nth row (N = 1 would be the idxmax)
if N > len(sorted_settings_df):
    raise ValueError(f"N = {N} is larger than the number of available settings ({len(sorted_settings_df)}).")
best_match = sorted_settings_df.iloc[N - 1]
db_cavity_freq = best_match["omega_C"]  # GHz
db_w_y = best_match["omega_Y"]  # GHz
db_gamma_vec = np.array([best_match["kappa_C"], best_match["kappa_Y"]])
print("Best match from DB settings (largest detuning):")
print(best_match)

# Setup symbolic equations (they are independent of J).
symbols_dict = setup_symbolic_equations()

# --- Query DB for measured trace ---
mean_freq = (db_cavity_freq + db_w_y) / 2  # GHz
freq_min = (mean_freq - FREQUENCY_RADIUS) * 1e9  # Hz
freq_max = (mean_freq + FREQUENCY_RADIUS) * 1e9  # Hz

data_query = f"""
SELECT *
FROM expr
WHERE experiment_id = '{experiment_id}'
  AND readout_type = '{readout_type}'
  AND ABS(omega_C - {db_cavity_freq}) < 1e-05
  AND ABS(omega_Y - {db_w_y}) < 1e-05
  AND frequency_hz BETWEEN {freq_min} AND {freq_max}
ORDER BY frequency_hz
"""
print("Data query:")
print(data_query)
data_df = pd.read_sql_query(data_query, engine)
if data_df.empty:
    raise ValueError("No trace data found matching the best-fit settings.")

# Pivot the data so that each row is a current and each column a frequency.
pivot_table = data_df.pivot_table(index="set_amperage", columns="frequency_hz",
                                  values="power_dBm", aggfunc="first")
measured_currents = pivot_table.index.values
# Convert frequency from Hz to GHz.
measured_freqs = pivot_table.columns.values / 1e9
measured_power = pivot_table.values
# For overlay, choose one trace (e.g., at the lowest current).
trace_index = 0
measured_trace_dbm = measured_power[trace_index, :]

# ------------------ EXPERIMENTAL PEAK FINDING ------------------
# Use scipy.find_peaks on the dBm data to get initial guesses.
peaks_exp, props = find_peaks(measured_trace_dbm, height=-30, prominence=0.025, distance=25)
if len(peaks_exp) == 0:
    raise ValueError("No peaks found in experimental data for initial guesses.")
elif len(peaks_exp) == 1:
    initial_guess_freqs = np.array([measured_freqs[peaks_exp[0]], measured_freqs[peaks_exp[0]] + 0.001])
else:
    prominences = props["prominences"]
    sorted_idx = np.argsort(prominences)[::-1]  # highest prominence first
    best_two = peaks_exp[sorted_idx[:2]]
    best_two = np.sort(best_two)
    initial_guess_freqs = measured_freqs[best_two]
print("Initial experimental peak guesses (GHz):", initial_guess_freqs)

# IMPORTANT: Convert the measured trace from dBm to linear scale for fitting.
measured_trace_linear = 10 ** (measured_trace_dbm / 10)

# ------------------ DOUBLE LORENTZIAN FIT ON EXPERIMENTAL TRACE ------------------
# Use the initial guesses (in GHz) obtained above.
double_fit_result = double_lorentzian_fit_NR(measured_freqs, measured_trace_linear,
                                             initial_guess_freqs[0], initial_guess_freqs[1])
if double_fit_result is None:
    raise ValueError("Double Lorentzian fit failed for experimental data.")

# Extract fitted peak centers (in GHz).
fitted_peak1 = double_fit_result.params["lz1_center"].value
fitted_peak2 = double_fit_result.params["lz2_center"].value
exp_fitted_peaks = np.sort(np.array([fitted_peak1, fitted_peak2]))
print("Experimental fitted peaks (GHz) from double Lorentzian fit:", exp_fitted_peaks)

# For plotting, evaluate the fit and convert back to dBm.
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
        cavity_freq=db_cavity_freq,
        w_y=db_w_y,
        gamma_vec=db_gamma_vec,
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([0, 1]),
        phi_val=np.deg2rad(180),
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
initial_J = 0.008414
bounds = (0.007, 0.009)
res = minimize_scalar(cost_function, bounds=bounds, method='bounded')
best_J = res.x
print(f"Optimal J_value found: {best_J:.6f} with cost {res.fun:.3e}")

# Recompute simulation trace with the optimal J.
simulated_trace_opt = simulate_trace(best_J)
simulated_trace_initial = simulate_trace(initial_J)
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
plt.title(f"Overlay: Simulated vs. Measured Trace\n(Detuning = {db_cavity_freq - db_w_y} GHz)")
plt.grid(True)

# --- Mark the experimental initial guesses ---
initial_guess_indices = [np.argmin(np.abs(measured_freqs - guess)) for guess in initial_guess_freqs]
plt.plot(measured_freqs[initial_guess_indices], measured_trace_dbm[initial_guess_indices],
         'm*', markersize=12, label="Exp Initial Guesses")

# --- Mark the experimental fitted peaks (from double Lorentzian) ---
fitted_y_linear = double_fit_result.eval(x=exp_fitted_peaks)
fitted_y_dbm = 10 * np.log10(fitted_y_linear)
plt.plot(exp_fitted_peaks, fitted_y_dbm, 'c*', markersize=12, label="Exp Fitted Peaks")

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
output_filename = f"{N}_overlay_trace_with_regression_improved.png"
plt.savefig(output_filename, dpi=400)
print(f"Overlay plot saved to {output_filename}")
