import numpy as np


def eigenvalues(gain_loss, detuning, coupling, phi, kappa_c):
    """
    Calculate eigenvalues of the given 2x2 matrix system.
    gain_loss: K/J (dimensionless)
    detuning: Δ/J (dimensionless)
    coupling: J (we can set this to 1 for simplicity)
    phi: phase parameter
    kappa_c: Cavity decay rate (dimensionless or scaled appropriately)
    """
    # Compute kappa_tilde/2
    kappa_tilde_offset = (kappa_c - gain_loss / 2) / 2

    # Define the 2x2 matrix with the offset added to the diagonal
    matrix = np.array([
        [((detuning * -1j) / 2) - (gain_loss / 2) + kappa_tilde_offset, -1j * coupling * np.exp(1j * phi)],
        [-1j * coupling, ((detuning * 1j) / 2) + (gain_loss / 2) + kappa_tilde_offset]
    ])
    eigenvals = np.linalg.eigvals(matrix)
    return eigenvals


def get_imag_diff_trace(K_min, K_max, N=100, phi=0.0, delta=0.0, coupling=1.0, kappa_c=0.1):
    """
    Compute the imaginary-part difference of the eigenvalues as a function of K/J.

    Parameters:
        K_min (float): Minimum value of K/J.
        K_max (float): Maximum value of K/J.
        N (int): Number of points in the range.
        phi (float): Phase parameter (default 0).
        delta (float): Δ/J (default 0).
        coupling (float): J (default 1.0).
        kappa_c (float): Cavity decay rate.

    Returns:
        K_values (ndarray): Array of K/J values.
        imag_diff (ndarray): Array of the imaginary-part differences of eigenvalues at each K/J.
    """
    K_values = np.linspace(K_min, K_max, N)
    imag_diff = np.zeros_like(K_values)

    for i, K in enumerate(K_values):
        evs = eigenvalues(K, delta, coupling, phi, kappa_c)
        # Imaginary part difference
        imag_diff[i] = np.abs(evs[0].imag - evs[1].imag)
    return K_values, imag_diff


def main():
    # Make a plot of the eigenvalues
    import matplotlib.pyplot as plt

    # Set cavity decay rate (kappa_c)
    kappa_c = 1.5

    # Compute the eigenvalue trace
    K_vals, imag_diff = get_imag_diff_trace(0, 2, N=100, phi=0.0, delta=0.0, coupling=1.0, kappa_c=kappa_c)

    # Plot the results
    plt.plot(K_vals, imag_diff, label=f'κ_c={kappa_c}')
    plt.xlabel('K/J')
    plt.ylabel('Imaginary part difference')
    plt.title('Imaginary part difference of eigenvalues')
    plt.legend()
    plt.savefig('eigenvalues.png')
    plt.close()


if __name__ == '__main__':
    main()
