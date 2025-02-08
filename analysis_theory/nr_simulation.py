#!/usr/bin/env python3
"""
NR EP Simulation Module

This function simulates an NR trace using the symbolic model. It uses the input
parameters (j_coupling, ω₍c₎, ω₍y₎, κ₍c₎, κ₍y₎ and the frequency sweep array)
to compute the steady‐state response and the photon numbers. Then it saves a plot of the
simulated NR trace in a folder of the form:

    plots/<expr_id>/debug_analysis/nr

The saved file name is prefixed with an order index (so that it sorts in the desired order)
and includes "nr_trace" with the detuning value (ω₍c₎ – ω₍y₎) in the name.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import symbolic_module as sm

# You can adjust this if you want a different DPI.
SAVE_DPI = 300


def simulate_trace(J_val, cavity_freq, w_y, kappa_c, kappa_y, freqs):
    """
    Given a trial J_val, simulate the trace.
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
        phi_val=np.deg2rad(180),
    )
    nr_ep_ss_eqn = sm.get_steady_state_response_transmission(symbols_dict, sim_params)
    photon_numbers_sim = sm.compute_photon_numbers_transmission(nr_ep_ss_eqn, freqs)
    sim_trace = np.log10(photon_numbers_sim)
    return sim_trace


def setup_fast_simulation(drive=(1, 0), readout=(0, 1)):
    """
    Wrapper that calls your new 'setup_fast_transmission_function'
    and returns the resulting function.
    """
    return sm.setup_fast_transmission_function(drive=drive, readout=readout)


def fast_simulate_trace(
        fast_func,  # the lambdified function from setup_fast_simulation()
        J_val, cavity_freq, w_y, kappa_c, kappa_y, freqs, phi_val=np.deg2rad(180)
):
    """
    Evaluate the transmitted amplitude^2 (photon number)
    in log10 scale, for an array of freq points.

    Parameters
    ----------
    fast_func : callable
        The lambdified function from setup_fast_simulation().
    J_val, cavity_freq, w_y, kappa_c, kappa_y : float
        Numeric values for the model.
    freqs : 1D array (in GHz)
        Frequencies at which to evaluate.
    phi_val : float
        Phase in radians (default pi).

    Returns
    -------
    sim_trace : 1D array
        The log10 of photon numbers at each freq point.
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


def simulate_and_plot_nr_trace(j_coupling, omega_c, omega_y, kappa_c, kappa_y, nr_freqs, expr_id, order_index="000"):
    """
    Simulate the NR trace using the symbolic model and save a plot.

    Parameters:
        j_coupling: float
            The coupling parameter J.
        omega_c: float
            The cavity frequency (GHz).
        omega_y: float
            The YIG frequency (GHz).
        kappa_c: float
            The cavity damping (or linewidth) (GHz).
        kappa_y: float
            The YIG damping (or linewidth) (GHz).
        nr_freqs: array-like
            A 1D numpy array of frequencies (in Hz) over which to evaluate the response.
        expr_id: str
            The experiment identifier (used to form the folder path).
        order_index: str, optional
            A zero-padded order prefix to include in the filename (default "000").

    Returns:
        photon_numbers: np.ndarray
            The simulated photon numbers (linear scale) corresponding to nr_freqs.
    """
    # Setup the symbolic equations and the model parameters.
    symbols_dict = sm.setup_symbolic_equations()
    params = sm.ModelParams(
        J_val=j_coupling,
        g_val=0,  # g is always 0 in our experiment.
        cavity_freq=omega_c,
        w_y=omega_y,
        gamma_vec=np.array([kappa_c, kappa_y]),
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([0, 1]),
        phi_val=np.pi,
    )
    print(f"Simulating NR trace for order {order_index}...with params: {str(params)}")

    # Get the steady-state response and compute the photon numbers.
    nr_ep_ss_eqn = sm.get_steady_state_response_transmission(symbols_dict, params)
    photon_numbers = sm.compute_photon_numbers_transmission(nr_ep_ss_eqn, nr_freqs)
    photon_numbers = np.log10(photon_numbers)

    # Compute the detuning Δ = ω_c - ω_y
    delta = omega_c - omega_y

    # Determine the output folder: plots/<expr_id>/debug_analysis/nr
    output_folder = os.path.join("plots", expr_id, "debug_analysis", "nr")
    os.makedirs(output_folder, exist_ok=True)

    # Create a plot of the simulated NR trace.
    # (Assume nr_freqs is in Hz; convert to GHz for plotting.)
    plt.figure(figsize=(8, 6))
    plt.plot(nr_freqs / 1e9, photon_numbers, "b-", label="NR Trace")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Photon Number")
    plt.title(f"NR Trace, Detuning = {delta:.6f} GHz")
    plt.legend()
    plt.tight_layout()

    # Create the file name.
    file_name = f"{order_index}_nr_trace_Delta_{delta:.6f}.png"
    file_path = os.path.join(output_folder, file_name)
    plt.savefig(file_path, dpi=SAVE_DPI)
    plt.close()
    print(f"Saved simulated NR trace plot to {file_path}")

    return photon_numbers


# For testing purposes:
if __name__ == "__main__":
    # Example parameters (adjust these as needed)
    j_coupling = 0.01  # Example J value (GHz)
    omega_c = 6.0  # Cavity frequency (GHz)
    omega_y = 6.0  # YIG frequency (GHz)
    kappa_c = 0.040464  # Cavity linewidth (GHz)
    kappa_y = 0.050500  # YIG linewidth (GHz)
    # Create a frequency sweep (in Hz). For example, 6 GHz ± 7.5 MHz:
    nr_freqs = np.linspace((omega_c - .5), (omega_c + 0.5), 10000)
    expr_id = "test_experiment"

    # Call the simulation function with an order index "001"
    simulate_and_plot_nr_trace(j_coupling, omega_c, omega_y, kappa_c, kappa_y, nr_freqs, expr_id, order_index="001")
