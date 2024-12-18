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

PLOT_TRACES_INDIVIDUALLY = False

CAVITY_READOUT = np.array([1, 0])
YIG_READOUT = np.array([0, 1])
J_OFF = 0


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


def analyze_2_peak_fwhm(photon_numbers, frequencies):
    """
    Analyze the response data to calculate the Full Width at Half Maximum (FWHM)
    for up to TWO peaks in the response. Returns arrays/lists with length=2
    (or fewer if not enough peaks found).

    Parameters:
        photon_numbers (1D array): The linear-scale response values (e.g., photon number).
        frequencies (1D array): Corresponding frequencies (same length as photon_numbers).

    Returns:
        fwhms (list): FWHMs for up to 2 peaks (None for missing peaks).
        peak_freqs (list): Frequencies of up to 2 peaks (None for missing peaks).
        peak_values (list): Peak values for up to 2 peaks (None for missing peaks).

    Notes:
        - If fewer than 2 peaks are found, the extra entry in the returned lists is None.
        - Uses the same "half-max" logic as the single-peak version.
    """
    # Find all peaks
    all_peaks, _ = find_peaks(photon_numbers)
    if len(all_peaks) == 0:
        print("WARNING: NO PEAKS FOUND")
        # Return arrays with two None entries (since no peaks)
        return [None, None], [None, None], [None, None]

    # Sort peaks by their amplitude (descending), pick top two
    peak_amplitudes = photon_numbers[all_peaks]
    sorted_idx = np.argsort(peak_amplitudes)[::-1]
    top_peaks = all_peaks[sorted_idx[:2]]  # up to 2 peaks

    fwhms = []
    peak_freqs = []
    peak_values = []

    for i, peak_idx in enumerate(top_peaks):
        peak_val = photon_numbers[peak_idx]
        peak_fr = frequencies[peak_idx]

        # half-max
        half_max = peak_val / 2.0

        # Find crossing indices
        # left side
        indices_below = np.where(photon_numbers[:peak_idx] < half_max)[0]
        # right side
        indices_above = np.where(photon_numbers[peak_idx:] < half_max)[0] + peak_idx

        if len(indices_below) == 0 or len(indices_above) == 0:
            # Can't define FWHM
            fwhm = None
        else:
            lower_idx = indices_below[-1]
            upper_idx = indices_above[0]
            fwhm = frequencies[upper_idx] - frequencies[lower_idx]

        fwhms.append(fwhm)
        peak_freqs.append(peak_fr)
        peak_values.append(peak_val)

    # If only 1 peak was found, fill the second slot with None
    if len(top_peaks) < 2:
        fwhms.append(None)
        peak_freqs.append(None)
        peak_values.append(None)

    return fwhms, peak_freqs, peak_values


def plot_single_trace(lo_freqs, photon_numbers_yig, photon_numbers_cavity, photon_numbers_hybrid, yig_fwhm, cavity_fwhm,
                      yig_peak_value, cavity_peak_value, peak_values, fwhms):
    plt.figure()
    plt.plot(lo_freqs, photon_numbers_yig, label="YIG")
    plt.plot(lo_freqs, photon_numbers_cavity, label="Cavity")
    plt.plot(lo_freqs, photon_numbers_hybrid, label="Hybrid")

    # Annotate FWHM for YIG
    if yig_fwhm is not None:
        K = cavity_fwhm - yig_fwhm
        # Calculate FWHM horizontal line limits for YIG
        indices_below = np.where(photon_numbers_yig[:np.argmax(photon_numbers_yig)] < yig_peak_value / 2)[0]
        indices_above = np.where(photon_numbers_yig[np.argmax(photon_numbers_yig):] < yig_peak_value / 2)[
                            0] + np.argmax(photon_numbers_yig)
        if len(indices_below) > 0 and len(indices_above) > 0:
            lower_freq_yig = lo_freqs[indices_below[-1]]
            upper_freq_yig = lo_freqs[indices_above[0]]
            plt.hlines(y=yig_peak_value / 2, xmin=lower_freq_yig, xmax=upper_freq_yig, colors="blue",
                       linestyles="--", label="YIG FWHM")

    # Annotate FWHM for Cavity
    if cavity_fwhm is not None:
        # Calculate FWHM horizontal line limits for Cavity
        indices_below = \
            np.where(photon_numbers_cavity[:np.argmax(photon_numbers_cavity)] < cavity_peak_value / 2)[0]
        indices_above = \
            np.where(photon_numbers_cavity[np.argmax(photon_numbers_cavity):] < cavity_peak_value / 2)[
                0] + np.argmax(photon_numbers_cavity)
        if len(indices_below) > 0 and len(indices_above) > 0:
            lower_freq_cavity = lo_freqs[indices_below[-1]]
            upper_freq_cavity = lo_freqs[indices_above[0]]
            plt.hlines(y=cavity_peak_value / 2, xmin=lower_freq_cavity, xmax=upper_freq_cavity, colors="green",
                       linestyles="--", label="Cavity FWHM")

    # Annotate FWHMs for Hybrid Peaks
    if fwhms[0] is not None:
        # Calculate FWHM horizontal line limits for the left hybrid peak
        indices_below = np.where(photon_numbers_hybrid[:np.argmax(photon_numbers_hybrid)] < peak_values[0] / 2)[
            0]
        indices_above = np.where(photon_numbers_hybrid[np.argmax(photon_numbers_hybrid):] < peak_values[0] / 2)[
                            0] + np.argmax(photon_numbers_hybrid)
        if len(indices_below) > 0 and len(indices_above) > 0:
            lower_freq_hybrid_left = lo_freqs[indices_below[-1]]
            upper_freq_hybrid_left = lo_freqs[indices_above[0]]
            plt.hlines(y=peak_values[0] / 2, xmin=lower_freq_hybrid_left, xmax=upper_freq_hybrid_left,
                       colors="red", linestyles="--", label="Left Hybrid FWHM")

    if fwhms[1] is not None:
        # Calculate FWHM horizontal line limits for the right hybrid peak
        indices_below = np.where(photon_numbers_hybrid[:np.argmax(photon_numbers_hybrid)] < peak_values[1] / 2)[
            0]
        indices_above = np.where(photon_numbers_hybrid[np.argmax(photon_numbers_hybrid):] < peak_values[1] / 2)[
                            0] + np.argmax(photon_numbers_hybrid)
        if len(indices_below) > 0 and len(indices_above) > 0:
            lower_freq_hybrid_right = lo_freqs[indices_below[-1]]
            upper_freq_hybrid_right = lo_freqs[indices_above[0]]
            plt.hlines(y=peak_values[1] / 2, xmin=lower_freq_hybrid_right, xmax=upper_freq_hybrid_right,
                       colors="orange", linestyles="--", label="Right Hybrid FWHM")

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Photon Number")
    plt.title(f"K value: {K}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots_simulation/SIM_{K}.png")
    plt.close()


def run_simulated_experiment(j_coupling, omega_c, omega_y, kappa_c):
    # Initialize DataFrame to store results
    results = pd.DataFrame(columns=[
        "kappa_y",
        "yig_fwhm", "yig_peak_freq", "yig_peak_value",
        "cavity_fwhm", "cavity_peak_freq", "cavity_peak_value",
        "hybrid_peak_difference"
    ])

    symbols_dict = sm.setup_symbolic_equations()

    params = sm.ModelParams(
        J_val=j_coupling,
        g_val=0,  # Difference between gamma values
        cavity_freq=omega_c,
        w_y=omega_y,
        gamma_vec=np.array([kappa_c, kappa_c]),  # Initial gamma values
        drive_vector=CAVITY_READOUT,
        readout_vector=CAVITY_READOUT,
        phi_val=0,  # Phase difference
    )
    """
    
    
    WHEN CHANGING STUFF, MAKE SURE THAT HTE X XXIS IS BIG ENOUGH, .001 WASN'T ENOUGH, CHANGE THAT TOO TOHERWISE THERE 
    ARE NANS EVEERWHERE
    
    
    """

    kappa_y_sweep = np.linspace(kappa_c, 0.002332 * 10, 1000)

    # Define LO frequencies around the cavity resonance
    lo_freqs = np.linspace(params.cavity_freq - .0075, params.cavity_freq + .0075, 10000)

    random_points = np.random.choice(kappa_y_sweep, size=10, replace=False)

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
        params.J_val = j_coupling

        # Get hybridized response
        hybrid_response = sm.get_steady_state_response_transmission_hybrid(symbols_dict, params)
        photon_numbers_hybrid_vec = sm.compute_photon_numbers_transmission_hybrid(hybrid_response, lo_freqs)
        photon_numbers_hybrid = np.sum(photon_numbers_hybrid_vec, axis=0).ravel()
        hybrid_peaks, _ = find_peaks(photon_numbers_hybrid)
        # use the 2 peak version
        fwhms, peak_freqs, peak_values = analyze_2_peak_fwhm(photon_numbers_hybrid, lo_freqs)

        # Calculate the difference between hybrid peak frequencies
        if len(hybrid_peaks) >= 2:
            hybrid_peak_difference = abs(lo_freqs[hybrid_peaks[1]] - lo_freqs[hybrid_peaks[0]])
        else:
            hybrid_peak_difference = None

        # If this kapap_y is one of the random ones, plot
        if PLOT_TRACES_INDIVIDUALLY:
            plot_single_trace(lo_freqs, photon_numbers_yig, photon_numbers_cavity, photon_numbers_hybrid, yig_fwhm,
                              cavity_fwhm, yig_peak_value, cavity_peak_value, peak_values, fwhms)

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
                "left_hybrid_fwhm": fwhms[0],
                "right_hybrid_fwhm": fwhms[1],
            }])
        ], ignore_index=True)

    # Add a colum for K, which is the difference between the cavity and YIG peak frequencies
    results["Delta"] = results["cavity_peak_freq"] - results["yig_peak_freq"]
    # Add a column for Delta, which is the difference between the cavity adn YIG peak frequencies
    results["K"] = results["cavity_fwhm"] - results["yig_fwhm"]

    return results


# def main():
#     # Initialize DataFrame to store results
#     results = pd.DataFrame(columns=[
#         "kappa_y",
#         "yig_fwhm", "yig_peak_freq", "yig_peak_value",
#         "cavity_fwhm", "cavity_peak_freq", "cavity_peak_value",
#         "hybrid_peak_difference"
#     ])
#
#     symbols_dict = sm.setup_symbolic_equations()
#
#     params = sm.ModelParams(
#         J_val=J_COUPLING,
#         g_val=0,  # Difference between gamma values
#         cavity_freq=OMEGA_C,
#         w_y=OMEGA_Y,
#         gamma_vec=np.array([KAPPA_C, KAPPA_C]),  # Initial gamma values
#         drive_vector=CAVITY_READOUT,
#         readout_vector=CAVITY_READOUT,
#         phi_val=0,  # Phase difference
#     )
#
#     kappa_y_sweep = np.linspace(KAPPA_C, 0.002332, 1000)
#
#     # Define LO frequencies around the cavity resonance
#     lo_freqs = np.linspace(params.cavity_freq - .001, params.cavity_freq + .001, 10000)
#
#     random_points = np.random.choice(kappa_y_sweep, size=10, replace=False)
#
#     for kappa_y in kappa_y_sweep:
#         params.gamma_vec[1] = kappa_y
#
#         # Step 1 - Measure the YIG by itself
#         params.readout_vector = YIG_READOUT
#         params.drive_vector = YIG_READOUT
#         params.J_val = J_OFF
#
#         # Get YIG response
#         yig_response = sm.get_steady_state_response_transmission(symbols_dict, params)
#         photon_numbers_yig = sm.compute_photon_numbers_transmission(yig_response, lo_freqs)
#
#         # Analyze the YIG response
#         yig_fwhm, yig_peak_freq, yig_peak_value = analyze_peak_fwhm(photon_numbers_yig, lo_freqs)
#
#         # Step 2 - Measure the Cavity by itself
#         params.readout_vector = CAVITY_READOUT
#         params.drive_vector = CAVITY_READOUT
#         params.J_val = J_OFF
#
#         # Get Cavity response
#         cavity_response = sm.get_steady_state_response_transmission(symbols_dict, params)
#         photon_numbers_cavity = sm.compute_photon_numbers_transmission(cavity_response, lo_freqs)
#
#         # Analyze the Cavity response
#         cavity_fwhm, cavity_peak_freq, cavity_peak_value = analyze_peak_fwhm(photon_numbers_cavity, lo_freqs)
#
#         # Step 3 - Measure the coupled system
#         params.readout_vector = CAVITY_READOUT
#         params.drive_vector = CAVITY_READOUT
#         params.J_val = J_COUPLING
#
#         # Get hybridized response
#         hybrid_response = sm.get_steady_state_response_transmission(symbols_dict, params)
#         photon_numbers_hybrid = sm.compute_photon_numbers_transmission(hybrid_response, lo_freqs)
#         hybrid_peaks, _ = find_peaks(photon_numbers_hybrid)
#         # use the 2 peak version
#         fwhms, peak_freqs, peak_values = analyze_2_peak_fwhm(photon_numbers_hybrid, lo_freqs)
#
#         # Calculate the difference between hybrid peak frequencies
#         if len(hybrid_peaks) >= 2:
#             hybrid_peak_difference = abs(lo_freqs[hybrid_peaks[1]] - lo_freqs[hybrid_peaks[0]])
#         else:
#             hybrid_peak_difference = None
#
#         # If this kapap_y is one of the random ones, plot
#         if False:
#             plt.figure()
#             plt.plot(lo_freqs, photon_numbers_yig, label="YIG")
#             plt.plot(lo_freqs, photon_numbers_cavity, label="Cavity")
#             plt.plot(lo_freqs, photon_numbers_hybrid, label="Hybrid")
#
#             # Annotate FWHM for YIG
#             if yig_fwhm is not None:
#                 K = cavity_fwhm - yig_fwhm
#                 # Calculate FWHM horizontal line limits for YIG
#                 indices_below = np.where(photon_numbers_yig[:np.argmax(photon_numbers_yig)] < yig_peak_value / 2)[0]
#                 indices_above = np.where(photon_numbers_yig[np.argmax(photon_numbers_yig):] < yig_peak_value / 2)[
#                                     0] + np.argmax(photon_numbers_yig)
#                 if len(indices_below) > 0 and len(indices_above) > 0:
#                     lower_freq_yig = lo_freqs[indices_below[-1]]
#                     upper_freq_yig = lo_freqs[indices_above[0]]
#                     plt.hlines(y=yig_peak_value / 2, xmin=lower_freq_yig, xmax=upper_freq_yig, colors="blue",
#                                linestyles="--", label="YIG FWHM")
#
#             # Annotate FWHM for Cavity
#             if cavity_fwhm is not None:
#                 # Calculate FWHM horizontal line limits for Cavity
#                 indices_below = \
#                     np.where(photon_numbers_cavity[:np.argmax(photon_numbers_cavity)] < cavity_peak_value / 2)[0]
#                 indices_above = \
#                     np.where(photon_numbers_cavity[np.argmax(photon_numbers_cavity):] < cavity_peak_value / 2)[
#                         0] + np.argmax(photon_numbers_cavity)
#                 if len(indices_below) > 0 and len(indices_above) > 0:
#                     lower_freq_cavity = lo_freqs[indices_below[-1]]
#                     upper_freq_cavity = lo_freqs[indices_above[0]]
#                     plt.hlines(y=cavity_peak_value / 2, xmin=lower_freq_cavity, xmax=upper_freq_cavity, colors="green",
#                                linestyles="--", label="Cavity FWHM")
#
#             # Annotate FWHMs for Hybrid Peaks
#             if fwhms[0] is not None:
#                 # Calculate FWHM horizontal line limits for the left hybrid peak
#                 indices_below = np.where(photon_numbers_hybrid[:np.argmax(photon_numbers_hybrid)] < peak_values[0] / 2)[
#                     0]
#                 indices_above = np.where(photon_numbers_hybrid[np.argmax(photon_numbers_hybrid):] < peak_values[0] / 2)[
#                                     0] + np.argmax(photon_numbers_hybrid)
#                 if len(indices_below) > 0 and len(indices_above) > 0:
#                     lower_freq_hybrid_left = lo_freqs[indices_below[-1]]
#                     upper_freq_hybrid_left = lo_freqs[indices_above[0]]
#                     plt.hlines(y=peak_values[0] / 2, xmin=lower_freq_hybrid_left, xmax=upper_freq_hybrid_left,
#                                colors="red", linestyles="--", label="Left Hybrid FWHM")
#
#             if fwhms[1] is not None:
#                 # Calculate FWHM horizontal line limits for the right hybrid peak
#                 indices_below = np.where(photon_numbers_hybrid[:np.argmax(photon_numbers_hybrid)] < peak_values[1] / 2)[
#                     0]
#                 indices_above = np.where(photon_numbers_hybrid[np.argmax(photon_numbers_hybrid):] < peak_values[1] / 2)[
#                                     0] + np.argmax(photon_numbers_hybrid)
#                 if len(indices_below) > 0 and len(indices_above) > 0:
#                     lower_freq_hybrid_right = lo_freqs[indices_below[-1]]
#                     upper_freq_hybrid_right = lo_freqs[indices_above[0]]
#                     plt.hlines(y=peak_values[1] / 2, xmin=lower_freq_hybrid_right, xmax=upper_freq_hybrid_right,
#                                colors="orange", linestyles="--", label="Right Hybrid FWHM")
#
#             plt.xlabel("Frequency (GHz)")
#             plt.ylabel("Photon Number")
#             plt.title(f"K value: {K}")
#             plt.grid(True)
#             plt.legend()
#             plt.savefig(f"plots_simulation/SIM_{K}.png")
#             plt.close()
#
#         # Update DataFrame with results
#         results = pd.concat([
#             results,
#             pd.DataFrame([{
#                 "kappa_y": kappa_y,
#                 "yig_fwhm": yig_fwhm,
#                 "yig_peak_freq": yig_peak_freq,
#                 "yig_peak_value": yig_peak_value,
#                 "cavity_fwhm": cavity_fwhm,
#                 "cavity_peak_freq": cavity_peak_freq,
#                 "cavity_peak_value": cavity_peak_value,
#                 "hybrid_peak_difference": hybrid_peak_difference,
#                 "left_hybrid_fwhm": fwhms[0],
#                 "right_hybrid_fwhm": fwhms[1],
#             }])
#         ], ignore_index=True)
#
#     # Add a colum for K, which is the difference between the cavity and YIG peak frequencies
#     results["Delta"] = results["cavity_peak_freq"] - results["yig_peak_freq"]
#     # Add a column for Delta, which is the difference between the cavity adn YIG peak frequencies
#     results["K"] = results["cavity_fwhm"] - results["yig_fwhm"]

def plot_results(result_df, j_coupling):
    # Plot kappa_y VS. Delta
    plt.figure()
    plt.plot(result_df["kappa_y"], result_df["Delta"], label="Delta")
    plt.xlabel("Kappa Y")
    plt.ylabel("Delta")
    plt.title("Kappa Y vs. Delta")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_DELTA.png")

    # Plot kappa_y VS. YIG FWHM
    plt.figure()
    plt.plot(result_df["kappa_y"], result_df["yig_fwhm"], label="YIG FWHM")
    plt.xlabel("Kappa Y")
    plt.ylabel("YIG FWHM")
    plt.title("Kappa Y vs. YIG FWHM")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_YIG_FWHM.png")

    # Plot K VS. YIG FWHM
    plt.figure()
    plt.plot(result_df["K"], result_df["yig_fwhm"], label="YIG FWHM")
    plt.xlabel("K")
    plt.ylabel("YIG FWHM")
    plt.title("K vs. YIG FWHM")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_YIG_FWHM_K.png")

    # Plot K VS. Hybrid FWHMs
    plt.figure()
    plt.plot(result_df["K"], result_df["left_hybrid_fwhm"], label="Left Hybrid FWHM")
    plt.plot(result_df["K"], result_df["right_hybrid_fwhm"], label="Right Hybrid FWHM")
    plt.xlabel("K")
    plt.ylabel("Hybrid FWHM")
    plt.title("K vs. Hybrid FWHMs")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/SIM_HYBRID_FWHM_K.png")

    # Plot K VS. Peak Difference
    plt.figure()
    plt.plot(result_df["K"] / j_coupling, result_df["hybrid_peak_difference"] / j_coupling, label="Peak Diff")
    plt.xlabel("K/J")
    plt.ylabel("Peak Diff / J")
    plt.title("K vs. Peak Diff")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots_simulation/PEAK_DIFF_K.png")


if __name__ == '__main__':
    # J_COUPLING = 0.000541
    J_COUPLING = 0.000541 # This is the more accurate J value    # KAPPA_C = 0.000464
    KAPPA_C = 0.000464
    # OMEGA_C = 6.002266969210131
    # OMEGA_Y = 6.002270293141932
    OMEGA_C = 6.0022
    OMEGA_Y = 6.0022

    result = run_simulated_experiment(J_COUPLING, OMEGA_C, OMEGA_Y, KAPPA_C)
    plot_results(result, J_COUPLING)
