import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_fit_results(csv_file):
    """
    Load the Lorentzian fit results CSV which has columns:
    voltage, readout_type, omega, omega_unc, kappa, kappa_unc
    """
    df = pd.read_csv(csv_file)
    return df


def compute_capital_kappa_and_capital_delta(cavity_df, yig_df):
    """
    Merge cavity and yig data on voltage and compute Kappa = kappa_c - kappa_y.
    Compute Kappa_unc by quadratic sum of uncertainties.
    """
    # Rename columns to distinguish
    cavity_df = cavity_df.rename(columns={'omega': 'omega_c', 'omega_unc': 'omega_c_unc',
                                          'kappa': 'kappa_c', 'kappa_unc': 'kappa_c_unc'})
    yig_df = yig_df.rename(columns={'omega': 'omega_y', 'omega_unc': 'omega_y_unc',
                                    'kappa': 'kappa_y', 'kappa_unc': 'kappa_y_unc'})

    # Merge on 'voltage'
    merged_df = pd.merge(cavity_df[['voltage', 'omega_c', 'omega_c_unc', 'kappa_c', 'kappa_c_unc']],
                         yig_df[['voltage', 'omega_y', 'omega_y_unc', 'kappa_y', 'kappa_y_unc']],
                         on='voltage', suffixes=('_c', '_y'))

    # Compute capital Kappa
    merged_df['Kappa'] = merged_df['kappa_c'] - merged_df['kappa_y']
    # Compute uncertainty in Kappa
    # Assuming independent uncertainties:
    merged_df['Kappa_unc'] = np.sqrt(merged_df['kappa_c_unc'] ** 2 + merged_df['kappa_y_unc'] ** 2)

    # Compute capital Delta
    merged_df['Delta'] = merged_df['omega_c'] - merged_df['omega_y']
    # Compute uncertainty in Delta
    # Assuming independent uncertainties:
    merged_df['Delta_unc'] = np.sqrt(merged_df['omega_c_unc'] ** 2 + merged_df['omega_y_unc'] ** 2)

    return merged_df


def plot_voltage_vs_delta(merged_df):
    """
    Plot Voltage vs. Delta with vertical error bars from Delta_unc.
    The y-axis is converted to MHz instead of GHz.
    A horizontal line is drawn at y = 0.
    """
    # Convert Delta and Delta_unc from GHz to MHz
    merged_df['Delta_mhz'] = merged_df['Delta'] * 1e3
    merged_df['Delta_unc_mhz'] = merged_df['Delta_unc'] * 1e3

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        merged_df['voltage'], merged_df['Delta_mhz'], yerr=merged_df['Delta_unc_mhz'],
        fmt='o', ecolor='red', capsize=4, label='Delta = omega_c - omega_y', markersize=2
    )

    # Add a horizontal line at y = 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Update axis labels
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Delta (MHz)')
    ax.set_title('Voltage vs. Delta Difference')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig('voltage_vs_delta.png', dpi=300)
    plt.close(fig)
    print("Saved plot: voltage_vs_delta.png")


def plot_voltage_vs_kappa(merged_df):
    """
    Plot Voltage vs. Kappa with vertical error bars from Kappa_unc.
    The y-axis is converted to MHz instead of GHz.
    """
    # Convert Kappa and Kappa_unc from GHz to MHz
    merged_df['Kappa_mhz'] = merged_df['Kappa'] * 1e3
    merged_df['Kappa_unc_mhz'] = merged_df['Kappa_unc'] * 1e3

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        merged_df['voltage'], merged_df['Kappa_mhz'], yerr=merged_df['Kappa_unc_mhz'],
        fmt='o', ecolor='red', capsize=4, label='Kappa = kappa_c - kappa_y', markersize=2
    )

    # Update axis labels
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Kappa (MHz)')
    ax.set_title('Voltage vs. Kappa Difference')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig('voltage_vs_kappa.png', dpi=300)
    plt.close(fig)
    print("Saved plot: voltage_vs_kappa.png")


def plot_kappa_to_delta_ratio(merged_df):
    """
    Plot the ratio Kappa / Delta vs. Voltage to compare the magnitudes.
    """
    # Calculate ratio (convert Delta to MHz if it's in GHz)
    merged_df['Ratio'] = np.abs(merged_df['Kappa_mhz'] / merged_df['Delta_mhz'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(merged_df['voltage'], merged_df['Ratio'], 'o-', label='Kappa / Delta', markersize=4)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Kappa / Delta')
    ax.set_title('Comparison of Kappa and Delta (Ratio)')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal Kappa and Delta')
    ax.grid(True)
    ax.legend()

    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('kappa_to_delta_ratio.png', dpi=300)
    plt.close(fig)
    print("Saved plot: kappa_to_delta_ratio.png")


if __name__ == "__main__":
    # Example usage
    # Load cavity and yig CSV files created by the analysis code
    cavity_csv = r'C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\analysis\csv\7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc_cavity\lorentzian_fit_results_7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc_cavity.csv'
    yig_csv = r'C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\analysis\csv\7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc_yig\lorentzian_fit_results_7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc_yig.csv'

    cavity_df = load_fit_results(cavity_csv)
    yig_df = load_fit_results(yig_csv)

    merged_df = compute_capital_kappa_and_capital_delta(cavity_df, yig_df)

    plot_voltage_vs_kappa(merged_df)
    plot_voltage_vs_delta(merged_df)
    plot_kappa_to_delta_ratio(merged_df)
