"""
Simulates the experiment

Uses the same model parameters as the dimer model, but measures the YIG
and the Cavity separately, measuring Kappa and Delta using the same method
as in the experiment.

Then, that data is used to calculate the stuffs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import symbolic_module as sm
import pandas as pd


def analyze_peak_fwhm(photon_numbers, frequencies):
    """
    Analyze the response data to calculate the Full Width at Half Maximum (FWHM)
    and locate the peak positions.

    Parameters:
        photon_numbers: Array of photon numbers (response values in linear scale).
        frequencies: Array of corresponding frequencies.

    Returns:
        fwhm: The FWHM of the main peak in frequency units (None if no valid peak).
        peak_freq: The frequency of the main peak (None if no peak found).
        peak_value: The value of the main peak (None if no peak found).
    """

    # Find peaks in the response
    peaks, _ = find_peaks(photon_numbers)
    if len(peaks) == 0:
        print('WARNING: NO PEAKS FOUND')
        return None, None, None  # No peaks found

    # Use the highest peak (modify logic if you want the first peak instead)
    peak_idx = peaks[np.argmax(photon_numbers[peaks])]
    peak_value = photon_numbers[peak_idx]
    peak_freq = frequencies[peak_idx]

    # Define the half-maximum value
    half_max = peak_value / 2

    # Find indices where the response crosses the half-maximum
    indices_below = np.where(photon_numbers[:peak_idx] < half_max)[0]
    indices_above = np.where(photon_numbers[peak_idx:] < half_max)[0] + peak_idx

    if len(indices_below) == 0 or len(indices_above) == 0:
        return None, peak_freq, peak_value  # No valid FWHM found

    lower_half_max_idx = indices_below[-1]
    upper_half_max_idx = indices_above[0]

    # Calculate the frequencies at the half-maximum
    lower_freq = frequencies[lower_half_max_idx]
    upper_freq = frequencies[upper_half_max_idx]

    # FWHM is the difference between the upper and lower frequencies
    fwhm = upper_freq - lower_freq
    return fwhm, peak_freq, peak_value


J_COUPLING = .1
J_OFF = 0
CAVITY_READOUT = np.array([1, 0])
YIG_READOUT = np.array([0, 1])
KAPPA_C = .15
OMEGA_C = 6.0
OMEGA_Y = 6.0


def main():
    # Initialize DataFrame to store results
    results = pd.DataFrame(columns=[
        "kappa_y",
        "yig_fwhm", "yig_peak_freq", "yig_peak_value",
        "cavity_fwhm", "cavity_peak_freq", "cavity_peak_value",
        "hybrid_peak_difference"
    ])

    symbols_dict = sm.setup_symbolic_equations()

    params = sm.ModelParams(
        J_val=J_COUPLING,
        g_val=0,  # Difference between gamma values
        cavity_freq=OMEGA_C,
        w_y=OMEGA_Y,
        gamma_vec=np.array([KAPPA_C, KAPPA_C]),  # Initial gamma values
        drive_vector=CAVITY_READOUT,
        readout_vector=CAVITY_READOUT,
        phi_val=0,  # Phase difference
    )

    kappa_y_sweep = np.linspace(KAPPA_C, 1, 1000)

    # Define LO frequencies around the cavity resonance
    lo_freqs = np.linspace(params.cavity_freq - 1, params.cavity_freq + 1, 10000)

    for kappa_y in kappa_y_sweep:
        params.gamma_vec[1] = kappa_y

        # Step 1 - Measure the YIG by itself
        params.readout_vector = YIG_READOUT
        params.drive_vector = YIG_READOUT
        params.J_val = J_OFF

        # Get YIG response
        yig_response = sm.get_steady_state_response_transmission(symbols_dict, params)
        photon_numbers_yig = sm.compute_photon_numbers_transmission(yig_response, lo_freqs)

        # Analyze the YIG response
        yig_fwhm, yig_peak_freq, yig_peak_value = analyze_peak_fwhm(photon_numbers_yig, lo_freqs)

        # Step 2 - Measure the Cavity by itself
        params.readout_vector = CAVITY_READOUT
        params.drive_vector = CAVITY_READOUT
        params.J_val = J_OFF

        # Get Cavity response
        cavity_response = sm.get_steady_state_response_transmission(symbols_dict, params)
        photon_numbers_cavity = sm.compute_photon_numbers_transmission(cavity_response, lo_freqs)

        # Analyze the Cavity response
        cavity_fwhm, cavity_peak_freq, cavity_peak_value = analyze_peak_fwhm(photon_numbers_cavity, lo_freqs)

        # Step 3 - Measure the coupled system
        params.readout_vector = CAVITY_READOUT
        params.drive_vector = CAVITY_READOUT
        params.J_val = J_COUPLING

        # Get hybridized response
        hybrid_response = sm.get_steady_state_response_transmission(symbols_dict, params)
        photon_numbers_hybrid = sm.compute_photon_numbers_transmission(hybrid_response, lo_freqs)
        hybrid_peaks, _ = find_peaks(photon_numbers_hybrid)

        # Calculate the difference between hybrid peak frequencies
        if len(hybrid_peaks) >= 2:
            hybrid_peak_difference = abs(lo_freqs[hybrid_peaks[1]] - lo_freqs[hybrid_peaks[0]])
        else:
            hybrid_peak_difference = None

        # Update DataFrame with results
        results = pd.concat([
            results,
            pd.DataFrame([{
                "kappa_y": kappa_y,
                "yig_fwhm": yig_fwhm,
                "yig_peak_freq": yig_peak_freq,
                "yig_peak_value": yig_peak_value,
                "cavity_fwhm": cavity_fwhm,
                "cavity_peak_freq": cavity_peak_freq,
                "cavity_peak_value": cavity_peak_value,
                "hybrid_peak_difference": hybrid_peak_difference,
            }])
        ], ignore_index=True)

    # Add a colum for K, which is the difference between the cavity and YIG peak frequencies
    results["Delta"] = results["cavity_peak_freq"] - results["yig_peak_freq"]
    # Add a column for Delta, which is the difference between the cavity adn YIG peak frequencies
    results["K"] = results["cavity_fwhm"] - results["yig_fwhm"]

    # Plot kappa_y VS. Delta
    plt.figure()
    plt.plot(results["kappa_y"], results["Delta"], label="Delta")
    plt.xlabel("Kappa Y")
    plt.ylabel("Delta")
    plt.title("Kappa Y vs. Delta")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_DELTA.png")

    # Plot kappa_y VS. YIG FWHM
    plt.figure()
    plt.plot(results["kappa_y"], results["yig_fwhm"], label="YIG FWHM")
    plt.xlabel("Kappa Y")
    plt.ylabel("YIG FWHM")
    plt.title("Kappa Y vs. YIG FWHM")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_YIG_FWHM.png")

    # Plot K VS. YIG FWHM
    plt.figure()
    plt.plot(results["K"], results["yig_fwhm"], label="YIG FWHM")
    plt.xlabel("K")
    plt.ylabel("YIG FWHM")
    plt.title("K vs. YIG FWHM")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_YIG_FWHM_K.png")

    # Plot K VS. Peak Difference
    plt.figure()
    plt.plot(results["K"], results["hybrid_peak_difference"], label="Peak Diff")
    plt.xlabel("K")
    plt.ylabel("Peak Diff")
    plt.title("K vs. Peak Diff")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/PEAK_DIFF_K.png")

if __name__ == '__main__':
    main()
