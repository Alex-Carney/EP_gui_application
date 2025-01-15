#!/usr/bin/env python3
"""
best_fit_j.py

A script to load an existing peak-difference CSV (already created by a previous script),
filter out spurious data, then perform a least-squares search for the best-fit J
that aligns the experiment with a theoretical simulation from 'full_simulation_expr' (fse).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

import full_simulation_expr as fse  # Make sure this is in your Python path

# -------------- Configurable parameters --------------
PEAK_DIFF_CSV = r"C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\analysis\csv\96187cb2-5d3d-40eb-9b3a-6d1d69b7ac0a_peak_differences\peak_differences_vs_kappa_96187cb2-5d3d-40eb-9b3a-6d1d69b7ac0a.csv"
OUTPUT_FOLDER = "best_fit_j_plots"
Y_THRESHOLD = 0.0020  # Spurious peak differences ABOVE this are ignored
J_SEARCH_MIN = 0.0008077761452159  # Lower bound (or bracket) for J searching
J_SEARCH_MAX = 0.000977761452159  # Upper bound (or bracket) for J searching

OMEGA_C_VAL = 5.9990958310959375
OMEGA_Y_VAL = 5.999105573076081
KAPPA_C_VAL = 0.0010269773575826


# -----------------------------------------------------

def error_function(J_val, df_data, omega_c, omega_y, kappa_c):
    """
    Compute a least-squares error between the 'good' data and an FSE-simulated curve for a given J_val.

    Parameters
    ----------
    J_val : float
        Candidate coupling value (GHz).
    df_data : pd.DataFrame
        DataFrame containing the "good" data (Kappa, Peak Difference).
    omega_c, omega_y, kappa_c : float
        Additional parameters used to run the simulation.

    Returns
    -------
    float
        A scalar error measure (mean squared residual).
    """
    # Run the FSE simulation with the candidate J
    print('Running simulation with J =', J_val)
    sim_df = fse.run_simulated_experiment(
        j_coupling=J_val,
        omega_c=omega_c,
        omega_y=omega_y,
        kappa_c=kappa_c
    )
    # sim_df has columns e.g. ['K', 'hybrid_peak_difference']

    # We'll compare experiment's (Kappa, PeakDiff) to the theory's (K, hybrid_peak_difference).
    # Let's do a 1D interpolation of the theory so we can get a predicted Splitting
    # at each experimental Kappa.
    theory_K = sim_df['K'].values
    theory_split = sim_df['hybrid_peak_difference'].values

    # Interpolate the theory:
    # bounds_error=False => outside range will give NaN
    f_theory = interp1d(theory_K, theory_split, kind='linear', bounds_error=False, fill_value=np.nan)

    residuals = []
    for _, row in df_data.iterrows():
        exp_K = row['Kappa']  # X
        exp_pd = row['Peak Difference (GHz)']  # Y
        th_pd = f_theory(exp_K)
        if not np.isnan(th_pd):
            residuals.append((exp_pd - th_pd) ** 2)
        else:
            # If outside the interpolation range, just ignore or penalize:
            residuals.append(0.2)  # or some penalty

    # Return the average of squared residuals
    if len(residuals) == 0:
        return 1e9  # If all got filtered out, return a huge error
    return np.mean(residuals)


def find_best_J(df_data, omega_c, omega_y, kappa_c, bracket=None):
    """
    Minimize the error_function over J in a specified bracket or range.

    Parameters
    ----------
    df_data : pd.DataFrame
        Experimental data (already filtered to ignore spurious points).
    omega_c, omega_y, kappa_c : float
        Parameters for fse.run_simulated_experiment(...).
    bracket : tuple or None
        If not None, pass to minimize_scalar(...).

    Returns
    -------
    (best_J, best_err) : (float, float)
        The best-fit J value and the corresponding minimum error.
    """

    def objective(J_val):
        return error_function(J_val, df_data, omega_c, omega_y, kappa_c)

    if bracket is not None:
        # 'brent' or 'golden' method with bracket
        result = minimize_scalar(objective, method='brent', bracket=bracket)
    else:
        # 'bounded' method with explicit min/max
        result = minimize_scalar(objective, bounds=(J_SEARCH_MIN, J_SEARCH_MAX), method='bounded')

    return result.x, result.fun


def main():
    # 1) Read the CSV with your "peak differences vs Kappa" data
    if not os.path.exists(PEAK_DIFF_CSV):
        print(f"[ERROR] CSV not found: {PEAK_DIFF_CSV}")
        return
    df = pd.read_csv(PEAK_DIFF_CSV)

    # 2) Filter out spurious data
    #    Keep only rows where 'Peak Difference (GHz)' >= Y_THRESHOLD
    #    so we ignore the ones below 0.0020 GHz
    df_good = df[df['Peak Difference (GHz)'] <= Y_THRESHOLD].copy()
    if df_good.empty:
        print("[WARNING] All data was below threshold. Cannot fit J.")
        return

    print(f"Using placeholders: omega_c={OMEGA_C_VAL}, omega_y={OMEGA_Y_VAL}, kappa_c={KAPPA_C_VAL}")

    # 4) Find best-fit J using a bracket or bounded approach
    #    E.g. bracket = (0.0001, 0.001) if you suspect that range

    # for soem reason bracket is different than bound, so we need to find the best bracket
    BRACKET_MIN = 1e-4
    BRACKET_MAX = 1e-3

    ENABLE_BRACKET = False
    bracket = (BRACKET_MIN, BRACKET_MAX) if ENABLE_BRACKET else None

    best_J, best_err = find_best_J(df_good, OMEGA_C_VAL, OMEGA_Y_VAL, KAPPA_C_VAL, bracket=bracket)
    print(f"[INFO] Best-fit J={best_J:.7g} GHz, with error={best_err:.7g}")

    # 5) Re-run the simulation with that best-fit J
    sim_df = fse.run_simulated_experiment(
        j_coupling=best_J,
        omega_c=OMEGA_C_VAL,
        omega_y=OMEGA_Y_VAL,
        kappa_c=KAPPA_C_VAL
    )
    # sim_df => columns: ['K', 'hybrid_peak_difference']

    # 6) Plot the good data vs. best-fit simulation
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        df_good['Kappa'], df_good['Peak Difference (GHz)'],
        xerr=df_good['Kappa_unc'], yerr=df_good['Peak Difference_unc (GHz)'],
        fmt='o', ecolor='red', capsize=4, label='Exp (filtered)', markersize=4
    )

    # Plot best-fit theory
    ax.plot(sim_df['K'], sim_df['hybrid_peak_difference'], 'r-', label=f"Theory (best-fit J={best_J:.6g} GHz)")

    # Format the figure
    ax.set_xlabel("Kappa (GHz)", fontsize=14)
    ax.set_ylabel("Peak Splitting (GHz)", fontsize=14)
    ax.set_title("Peak Splitting vs Kappa (Best-Fit J)", fontsize=16)
    ax.grid(True)
    ax.legend(fontsize=12)

    # Optionally set x-limits or y-limits to see relevant region
    # x_min = df_good['Kappa'].min()
    # x_max = df_good['Kappa'].max()
    # ax.set_xlim([x_min, x_max])

    plot_path = os.path.join(OUTPUT_FOLDER, "best_fit_j_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved best-fit plot to {plot_path}")


if __name__ == "__main__":
    main()
