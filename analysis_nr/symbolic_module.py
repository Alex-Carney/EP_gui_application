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

    def __str__(self):
        return (f"ModelParams(J={self.J_val}, cavity_freq={self.cavity_freq}, "
                f"omega_y={self.w_y}, gamma_vec={self.gamma_vec})")


def effective_hamiltonian_eigenvalues(delta_kappa, delta_f, coupling, phi):
    matrix = np.array([
        [((-1j * delta_f) / 2) - (delta_kappa / 2), -1j * coupling],
        [-1j * coupling * np.exp(1j * phi), ((1j * delta_f) / 2) + (delta_kappa / 2)]
    ])
    return np.linalg.eigvals(matrix)


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


def compute_photon_numbers_NR(ss_response_func, w_y_vals, w_f_vals):
    """
    Computes the photon numbers for the non-PT symmetric case.
    ss_response_func: steady-state response function from get_steady_state_response_non_PT
    w_y_vals: array of YIG frequencies
    w_f_vals: array of LO frequencies
    Returns a 2D array of photon numbers.
    """
    W_Y, W_F = np.meshgrid(w_y_vals, w_f_vals, indexing='ij')
    photon_numbers_complex = ss_response_func(W_Y, W_F)
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


def get_steady_state_response_NR(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    Returns a function that computes the steady-state response for the non-PT symmetric case.
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for non-PT symmetric case
    substitutions = {
        symbols_dict.w0[0]: params.cavity_freq,
        symbols_dict.w0[1]: symbols_dict.w_y,  # Keep w_y symbolic
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

    # Lambdify with w_y and w_f as variables
    return sp.lambdify((symbols_dict.w_y, symbols_dict.w_f), ss_eqn, 'numpy')


def get_steady_state_response_PT(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    Returns a function that computes the steady-state response for the PT symmetric case.
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for PT symmetric case
    substitutions = {
        w0[0]: params.cavity_freq,
        w0[1]: params.w_y,
        symbols_dict.J: params.J_val,
        symbols_dict.g: params.g_val,
        F[0]: params.drive_vector[0],
        F[1]: params.drive_vector[1],
        gamma[0]: params.gamma_vec[0],
        gamma[1]: symbols_dict.gam_y,  # Keep gam_y symbolic
        symbols_dict.phi_val: params.phi_val
    }

    ss_eqns_instantiated = steady_state_eqns.subs(substitutions)
    ss_eqn = (params.readout_vector[0] * ss_eqns_instantiated[0] +
              params.readout_vector[1] * ss_eqns_instantiated[1])

    # Lambdify with gam_y and w_f as variables
    return sp.lambdify((symbols_dict.gam_y, symbols_dict.w_f), ss_eqn, 'numpy')


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


def setup_fast_transmission_function(drive=(1, 0), readout=(0, 1)):
    """
    Builds a single symbolic expression for the transmitted field from a
    2-cavity system, then lambdifies it as a function of:
       (J, w_c, w_y, gamma_c, gamma_y, phi, w_f)

    Returns
    -------
    fast_func : callable
        fast_func(J_val, w_c_val, w_y_val, gamma_c_val, gamma_y_val, phi_val, w_f_array)
        -> complex array representing the transmitted amplitude vs w_f_array
    """
    # Define symbolic variables
    J_sym = sp.Symbol('J', real=True)
    w_c_sym = sp.Symbol('w_c', real=True)  # cavity freq
    w_y_sym = sp.Symbol('w_y', real=True)  # YIG freq
    gamma_c_sym = sp.Symbol('gamma_c', real=True)
    gamma_y_sym = sp.Symbol('gamma_y', real=True)
    phi_sym = sp.Symbol('phi_val', real=True)
    w_f_sym = sp.Symbol('w_f', real=True)  # Will pass array in "numpy" mode

    # We'll fix drive and readout as symbolic constants for clarity
    F1, F2 = sp.symbols('F1 F2', complex=True)
    r1, r2 = sp.symbols('r1 r2', complex=True)

    # Build the adjacency matrix
    #   [ 0,   e^(i phi)*J_sym ]
    #   [ J_sym,         0     ]
    adjacency_matrix = sp.Matrix([
        [0, sp.exp(sp.I * phi_sym) * J_sym],
        [J_sym, 0]
    ])

    # We have 2 cavities, define w0 = [w_c, w_y] and gamma = [gamma_c, gamma_y]
    # but we won't do subs; we incorporate them as direct symbols.
    w0_c1 = w_c_sym
    w0_c2 = w_y_sym
    gamma_1 = gamma_c_sym
    gamma_2 = gamma_y_sym

    # The driving frequency for each cavity is w_f_sym (the same LO freq).
    # We'll represent it as a 2-element column for convenience:
    wf_mat = sp.Matrix([w_f_sym, w_f_sym])

    # Build the cavity-dynamics matrix:
    # M = [ (0 i - gamma1/2 - i(w0_c1 - w_f_sym)),   adjacency_matrix[0,1]*i ]
    #     [ adjacency_matrix[1,0]*i,                (0 i - gamma2/2 - i(w0_c2 - w_f_sym)) ]
    #
    # or more systematically:
    M = sp.zeros(2)
    M[0, 0] = adjacency_matrix[0, 0] * sp.I - gamma_1 / sp.Integer(2) - sp.I * (w0_c1 - w_f_sym)
    M[0, 1] = adjacency_matrix[0, 1] * sp.I
    M[1, 0] = adjacency_matrix[1, 0] * sp.I
    M[1, 1] = adjacency_matrix[1, 1] * sp.I - gamma_2 / sp.Integer(2) - sp.I * (w0_c2 - w_f_sym)

    # Build the drive vector F:
    F_mat = sp.Matrix([F1, F2])

    # Solve steady-state:
    #   steady_state = M.inv() * F
    # Then the "transmitted" amplitude is readout dot steady_state
    steady_state = M.inv() * F_mat
    # readout vector r = (r1, r2)
    transmitted_expr = r1 * steady_state[0] + r2 * steady_state[1]

    # Now we lambdify this in terms of all relevant symbolic variables
    # We'll fix F1,F2,r1,r2 to the user-provided drive/readout if you want.
    # For example, if drive=(1,0), readout=(0,1), then F1=1,F2=0, r1=0,r2=1.
    # That means we have 7 real parameters + the w_f array as input.
    expr_simpl = sp.simplify(transmitted_expr)

    # We'll define the argument order carefully:
    # (J, w_c, w_y, gamma_c, gamma_y, phi, w_f)
    # and we internally fix F1=drive[0], F2=drive[1], r1=readout[0], r2=readout[1].
    fast_expr = expr_simpl.subs({
        F1: drive[0],
        F2: drive[1],
        r1: readout[0],
        r2: readout[1],
    })

    fast_func = sp.lambdify(
        (J_sym, w_c_sym, w_y_sym, gamma_c_sym, gamma_y_sym, phi_sym, w_f_sym),
        fast_expr, 'numpy'
    )

    return fast_func


def compute_photon_numbers_fast(fast_func,
                                J_val, w_c_val, w_y_val,
                                gamma_c_val, gamma_y_val,
                                phi_val, w_f_array, ):
    """
    Convenience function that calls the lambdified fast_func and returns
    |transmitted|^2 (photon number).
    """
    complex_ampl = fast_func(J_val, w_c_val, w_y_val,
                             gamma_c_val, gamma_y_val,
                             phi_val, w_f_array)
    return np.abs(complex_ampl) ** 2


if __name__ == "__main__":
    # Setup symbolic equations
    symbols_dict = setup_symbolic_equations()

    # Define NREP Params
    J_value = .0085  # GHz
    params = ModelParams(
        J_val=J_value,
        g_val=0,  # Difference between gamma values
        cavity_freq=6.0063363531479474,  # GHz
        w_y=5.98905,  # GHz
        gamma_vec=np.array([0.00096165, 0.00093425]),  # Initial gamma values
        drive_vector=np.array([1, 0]),
        readout_vector=np.array([0, 1]),
        phi_val=np.pi,  # Phase difference
    )

    nr_ep_ss_eqn = get_steady_state_response_transmission(symbols_dict, params)

    # Define LO frequencies around the cavity resonance
    mean_freq = (params.cavity_freq + params.w_y) / 2
    lo_freqs = np.linspace(mean_freq - 0.006, mean_freq + 0.006, 1000)

    # Compute photon numbers
    photon_numbers = compute_photon_numbers_transmission(nr_ep_ss_eqn, lo_freqs)

    photon_numbers = np.log10(photon_numbers)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(lo_freqs, photon_numbers)
    plt.xlabel('LO Frequency (GHz)')
    plt.ylabel('Photon Number')
    plt.title(f'Photon Number vs LO Frequency at Detuning = {params.cavity_freq - params.w_y} GHz')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("photon_number_vs_LO_frequency.png", dpi=400)

# if __name__ == "__main__":
#     # Setup symbolic equations
#     symbols_dict = setup_symbolic_equations()
#
#     # Define parameters
#     J_value = 0.1  # GHz
#     params = ModelParams(
#         J_val=J_value,
#         g_val=0,  # Difference between gamma values
#         cavity_freq=6.0,  # GHz
#         w_y=6.0,  # GHz
#         gamma_vec=np.array([0.025, 0.04]),  # Initial gamma values
#         drive_vector=np.array([1, 0]),
#         readout_vector=np.array([1, 0]),
#         phi_val=0,  # Phase difference
#     )
#
#     # Define \( K \) range scaled by J
#     K_range = np.linspace(-4 * J_value, 0, 100)  # GHz
#
#     # Calculate theoretical peak splittings
#     K_scaled, splittings_scaled = calculate_theoretical_peak_splittings(symbols_dict, params, K_range)
#
#     # Plot the results
#     plt.figure(figsize=(8, 6))
#     plt.plot(K_scaled, splittings_scaled, label="Theoretical Splitting")
#     plt.xlabel("K / J")
#     plt.ylabel("Peak Splitting / J")
#     plt.title("Theoretical Peak Splitting vs K")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("theoretical_peak_splitting_vs_K.png", dpi=400)
#     plt.show()
