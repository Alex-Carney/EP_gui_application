#!/usr/bin/env python3
"""
PTEP Simulation Module

This module simulates a PTEP trace using the symbolic model.
For the PTEP, the phase is fixed at 0 and the simulation is performed
as a function of probe frequency (in Hz). (The kappa detuning K = κ₍c₎ – κ₍y₎
is used only in the overall theory analysis/colorplots.)
The simulated trace (photon number vs. frequency) is saved to:

    plots/<expr_id>/debug_analysis/ptep

The saved file name is prefixed with an order index and includes the nominal kappa detuning.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import symbolic_module as sm

# You can adjust this if you want a different DPI.
SAVE_DPI = 300


def simulate_trace(J_val, cavity_freq, w_y, kappa_c, kappa_y, freqs):
    """
    Given a trial J_val, simulate the PTEP trace over a frequency sweep.

    Parameters:
        J_val : float
            The coupling parameter.
        cavity_freq : float
            The cavity frequency (GHz).
        w_y : float
            The YIG frequency (GHz).
        kappa_c : float
            The cavity damping (linewidth) (GHz).
        kappa_y : float
            The YIG damping (linewidth) (GHz).
        freqs : 1D array
            Array of frequencies (in Hz) over which to evaluate the response.

    Returns:
        sim_trace : 1D array
            The log₁₀ of the computed photon numbers.
    """
    symbols_dict = sm.setup_symbolic_equations()
    sim_params = sm.ModelParams(
        J_val=J_val,
        g_val=0,
        cavity_freq=cavity_freq,
        w_y=w_y,
        gamma_vec=np.array([kappa_c, kappa_y]),
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([0, 1]),
        phi_val=0,  # Phase is 0 for PTEP.
    )
    ss_eqn = sm.get_steady_state_response_transmission(symbols_dict, sim_params)
    photon_numbers = sm.compute_photon_numbers_transmission(ss_eqn, freqs)
    sim_trace = np.log10(photon_numbers)
    return sim_trace


def setup_fast_simulation(drive=(1, 0), readout=(0, 1)):
    """
    Wrapper that calls the new 'setup_fast_transmission_function'
    and returns the resulting function.
    """
    return sm.setup_fast_transmission_function(drive=drive, readout=readout)


def fast_simulate_trace(fast_func, J_val, cavity_freq, w_y, kappa_c, kappa_y, freqs, phi_val=0):
    """
    Evaluate the transmitted amplitude² (photon number)
    in log₁₀ scale, for an array of frequency points.

    Parameters:
        fast_func : callable
            The lambdified function from setup_fast_simulation().
        J_val, cavity_freq, w_y, kappa_c, kappa_y : float
            Numeric values for the model.
        freqs : 1D array (in Hz)
            Frequencies at which to evaluate.
        phi_val : float
            Phase in radians (default 0 for PTEP).

    Returns:
        sim_trace : 1D array
            The log₁₀ of the photon numbers at each frequency point.
    """
    photon_numbers = sm.compute_photon_numbers_fast(
        fast_func,
        J_val,
        cavity_freq,
        w_y,
        kappa_c,
        kappa_y,
        phi_val,
        freqs
    )
    return np.log10(photon_numbers)


def simulate_and_plot_ptep_trace(j_coupling, omega_c, w_y, kappa_c, kappa_y, freqs, expr_id, order_index="000"):
    """
    Simulate the PTEP trace using the symbolic model and save a plot.

    Parameters:
        j_coupling : float
            The coupling parameter J.
        omega_c : float
            The cavity frequency (GHz).
        w_y : float
            The YIG frequency (GHz).
        kappa_c : float
            The cavity linewidth (GHz).
        kappa_y : float
            The YIG linewidth (GHz).
        freqs : array-like
            A 1D numpy array of frequencies (in Hz) over which to evaluate the response.
        expr_id : str
            The experiment identifier (used to form the folder path).
        order_index : str, optional
            A zero-padded order prefix to include in the filename (default "000").

    Returns:
        photon_numbers : np.ndarray
            The simulated photon numbers (log₁₀ scale) corresponding to freqs.
    """
    symbols_dict = sm.setup_symbolic_equations()
    params = sm.ModelParams(
        J_val=j_coupling,
        g_val=0,
        cavity_freq=omega_c,
        w_y=w_y,
        gamma_vec=np.array([kappa_c, kappa_y]),
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([0, 1]),
        phi_val=0,  # Phase is 0 for PTEP.
    )
    print(f"Simulating PTEP trace for order {order_index}... with params: {str(params)}")

    ss_eqn = sm.get_steady_state_response_transmission(symbols_dict, params)
    photon_numbers = sm.compute_photon_numbers_transmission(ss_eqn, freqs)
    photon_numbers = np.log10(photon_numbers)

    # Compute the nominal kappa detuning, K = kappa_c - kappa_y.
    K_nominal = kappa_c - kappa_y

    # Determine the output folder: plots/<expr_id>/debug_analysis/ptep
    output_folder = os.path.join("plots", expr_id, "debug_analysis", "ptep")
    os.makedirs(output_folder, exist_ok=True)

    # Create a plot of the simulated PTEP trace versus frequency.
    plt.figure(figsize=(8, 6))
    plt.plot(freqs / 1e9, photon_numbers, "b-", label="PTEP Trace")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Photon Number (log₁₀ scale)")
    plt.title(f"PTEP Trace, Nominal K Detuning = {K_nominal:.6f} GHz")
    plt.legend()
    plt.tight_layout()

    file_name = f"{order_index}_ptep_trace_K_{K_nominal:.6f}.png"
    file_path = os.path.join(output_folder, file_name)
    plt.savefig(file_path, dpi=SAVE_DPI)
    plt.close()
    print(f"Saved simulated PTEP trace plot to {file_path}")

    return photon_numbers


# For testing purposes:
if __name__ == "__main__":
    # Example parameters (adjust as needed)
    j_coupling = 0.01      # Example J value (GHz)
    omega_c = 6.0          # Cavity frequency (GHz)
    w_y = 6.0              # YIG frequency (GHz)
    kappa_c = 0.050        # Cavity linewidth (GHz)
    kappa_y = 0.040        # YIG linewidth (GHz)
    # Create a frequency sweep (in Hz) – for example, around 6 GHz ± 0.5 GHz:
    freqs = np.linspace((omega_c - 0.5)*1e9, (omega_c + 0.5)*1e9, 10000)
    expr_id = "test_experiment_ptep"

    simulate_and_plot_ptep_trace(j_coupling, omega_c, w_y, kappa_c, kappa_y, freqs, expr_id, order_index="001")
