from dataclasses import dataclass
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


@dataclass
class ModelSymbolics:
    J: sp.Symbol
    w_f: sp.Symbol
    w_y: sp.Symbol
    gam_y: sp.Symbol
    g: sp.Symbol
    w0: sp.MutableDenseMatrix
    gamma: sp.MutableDenseMatrix
    F: sp.MutableDenseMatrix
    phi_val: sp.Symbol
    steady_state_eqns: sp.Expr
    cavity_dynamics_matrix: sp.MutableDenseMatrix  # Added this line


@dataclass
class ModelParams:
    J_val: float
    g_val: float
    cavity_freq: float
    w_y: float
    gamma_vec: np.ndarray
    drive_vector: np.ndarray
    readout_vector: np.ndarray
    phi_val: float


def setup_symbolic_equations() -> ModelSymbolics:
    """
    Sets up the symbolic steady-state equations for the two-cavity system.
    Returns the symbolic variables and the steady-state equations.
    """
    # Symbolic variables
    J, w_f, w_y, gam_y, g = sp.symbols('J w_f w_y gam_y g', real=True)
    w_c1, w_c2 = sp.symbols('w_c1 w_c2', real=True)
    gam_1, gam_2 = sp.symbols('gamma_1 gamma_2', real=True)
    w0 = sp.Matrix([w_c1, w_c2])
    gamma = sp.Matrix([gam_1, gam_2])
    F1, F2 = sp.symbols('F1 F2')
    F = sp.Matrix([F1, F2])

    # Define the adjacency matrix with phase factor
    phi_val = sp.symbols('phi_val', real=True)
    cavity_adj_matrix = sp.Matrix([
        [0, sp.exp(1j * phi_val) * J],
        [J, 0]
    ])

    # Driving frequency vector
    num_cavities = cavity_adj_matrix.shape[0]
    wf = w_f * sp.ones(num_cavities, 1)

    # Define the cavity dynamics matrix
    cavity_dynamics_matrix = sp.zeros(num_cavities)
    cavity_dynamics_matrix[0, 0] = (cavity_adj_matrix[0, 0] * 1j
                                    - gamma[0] / 2
                                    - 1j * (w0[0] - wf[0]))
    cavity_dynamics_matrix[0, 1] = cavity_adj_matrix[0, 1] * 1j
    cavity_dynamics_matrix[1, 0] = cavity_adj_matrix[1, 0] * 1j
    cavity_dynamics_matrix[1, 1] = (cavity_adj_matrix[1, 1] * 1j
                                    - gamma[1] / 2
                                    - 1j * (w0[1] - wf[1]))

    # Steady-state equations
    steady_state_eqns = cavity_dynamics_matrix.inv() * F
    steady_state_eqns_simplified = sp.simplify(steady_state_eqns)

    # Return the updated ModelSymbolics with cavity_dynamics_matrix
    return ModelSymbolics(
        J, w_f, w_y, gam_y, g, w0, gamma, F, phi_val,
        steady_state_eqns_simplified, cavity_dynamics_matrix
    )


def get_steady_state_response_transmission_hybrid(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    FOR TRANSMISSION, ALL PARAMETERS MUST BE FILLED. The only unfilled parameter is w_f
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for transmission case
    substitutions = {
        w0[0]: params.cavity_freq,
        w0[1]: params.w_y,
        symbols_dict.J: params.J_val,
        symbols_dict.g: params.g_val,
        F[0]: params.drive_vector[0],
        F[1]: params.drive_vector[1],
        gamma[0]: params.gamma_vec[0],
        gamma[1]: params.gamma_vec[1],
        symbols_dict.phi_val: params.phi_val
    }

    ss_eqns_instantiated = steady_state_eqns.subs(substitutions)
    # ss_eqn = (params.readout_vector[0] * ss_eqns_instantiated[0] +
    #           params.readout_vector[1] * ss_eqns_instantiated[1])

    # Lambdify with w_f as variable
    return sp.lambdify(symbols_dict.w_f, ss_eqns_instantiated, 'numpy')


def compute_photon_numbers_transmission_hybrid(ss_response_func, w_f_vals):
    """
    Computes the photon numbers for the transmission case.
    ss_response_func: steady-state response function from get_steady_state_response_transmission
    w_f_vals: array of LO frequencies
    Returns an array of photon numbers.
    """
    photon_numbers_complex = ss_response_func(w_f_vals)
    photon_numbers_real = np.abs(photon_numbers_complex) ** 2
    return photon_numbers_real


def get_steady_state_response_transmission(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    FOR TRANSMISSION, ALL PARAMETERS MUST BE FILLED. The only unfilled parameter is w_f
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for transmission case
    substitutions = {
        w0[0]: params.cavity_freq,
        w0[1]: params.w_y,
        symbols_dict.J: params.J_val,
        symbols_dict.g: params.g_val,
        F[0]: params.drive_vector[0],
        F[1]: params.drive_vector[1],
        gamma[0]: params.gamma_vec[0],
        gamma[1]: params.gamma_vec[1],
        symbols_dict.phi_val: params.phi_val
    }

    ss_eqns_instantiated = steady_state_eqns.subs(substitutions)
    ss_eqn = (params.readout_vector[0] * ss_eqns_instantiated[0] +
              params.readout_vector[1] * ss_eqns_instantiated[1])

    # Lambdify with w_f as variable
    return sp.lambdify(symbols_dict.w_f, ss_eqn, 'numpy')


def compute_photon_numbers_transmission(ss_response_func, w_f_vals):
    """
    Computes the photon numbers for the transmission case.
    ss_response_func: steady-state response function from get_steady_state_response_transmission
    w_f_vals: array of LO frequencies
    Returns an array of photon numbers.
    """
    photon_numbers_complex = ss_response_func(w_f_vals)
    photon_numbers_real = np.abs(photon_numbers_complex) ** 2
    return photon_numbers_real


def get_cavity_dynamics_eigenvalues_numeric(symbols_dict: ModelSymbolics, params: ModelParams):
    """
    Computes the eigenvalues of the cavity dynamics matrix with given parameters.
    """
    # Unpack symbols
    cavity_dynamics_matrix = symbols_dict.cavity_dynamics_matrix
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    phi_val = symbols_dict.phi_val
    w_f = symbols_dict.w_f

    # Substitutions
    substitutions = {
        w0[0]: params.cavity_freq,
        w0[1]: params.w_y,
        symbols_dict.J: params.J_val,
        gamma[0]: params.gamma_vec[0],
        gamma[1]: params.gamma_vec[1],
        phi_val: params.phi_val,
        w_f: params.cavity_freq  # Assuming w_f equals cavity frequency
    }

    # Substitute into cavity_dynamics_matrix
    cavity_dynamics_matrix_sub = cavity_dynamics_matrix.subs(substitutions)

    # Convert to numpy array
    cavity_dynamics_matrix_num = np.array(cavity_dynamics_matrix_sub.evalf(), dtype=complex)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(cavity_dynamics_matrix_num)

    # Return eigenvalues
    return eigenvalues


def calculate_theoretical_peak_splittings(symbols_dict, params, K_values):
    """
    Calculate the peak splitting for a range of \( K \) values.

    Parameters:
        symbols_dict: Symbolic equations and variables.
        params: Model parameters.
        K_values: Array of \( K \) values to sweep.

    Returns:
        kappa_scaled: \( K / J \).
        splittings_scaled: Peak splitting scaled by \( J \).
    """
    splittings = []

    for K in K_values:
        # Update gamma_vec with the current \( K \) value
        params.gamma_vec[1] = params.gamma_vec[0] - K

        # Get steady-state response for transmission
        ss_response = get_steady_state_response_transmission(symbols_dict, params)

        # Define LO frequencies around the cavity resonance
        lo_freqs = np.linspace(params.cavity_freq - 0.5, params.cavity_freq + 0.5, 1000)

        # Compute photon numbers
        photon_numbers = compute_photon_numbers_transmission(ss_response, lo_freqs)

        # Find peaks in the photon number spectrum
        peaks, _ = find_peaks(photon_numbers)

        # If two peaks are found, calculate their splitting
        if len(peaks) >= 2:
            peak_splitting = abs(lo_freqs[peaks[1]] - lo_freqs[peaks[0]])
            splittings.append(peak_splitting)
        else:
            splittings.append(0)  # No splitting detected

    # Scale K and splittings by J
    K_scaled = K_values / params.J_val
    splittings_scaled = np.array(splittings) / params.J_val

    return K_scaled, splittings_scaled


if __name__ == "__main__":
    # Setup symbolic equations
    symbols_dict = setup_symbolic_equations()

    # Define parameters
    J_value = 0.1  # GHz
    params = ModelParams(
        J_val=J_value,
        g_val=0,  # Difference between gamma values
        cavity_freq=6.0,  # GHz
        w_y=6.0,  # GHz
        gamma_vec=np.array([0.025, 0.04]),  # Initial gamma values
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([1, 0]),
        phi_val=0,  # Phase difference
    )

    # Define \( K \) range scaled by J
    K_range = np.linspace(-4 * J_value, 0, 100)  # GHz

    # Calculate theoretical peak splittings
    K_scaled, splittings_scaled = calculate_theoretical_peak_splittings(symbols_dict, params, K_range)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(K_scaled, splittings_scaled, label="Theoretical Splitting")
    plt.xlabel("K / J")
    plt.ylabel("Peak Splitting / J")
    plt.title("Theoretical Peak Splitting vs K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("theoretical_peak_splitting_vs_K.png", dpi=400)
    plt.show()
