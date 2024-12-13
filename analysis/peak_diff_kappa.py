import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAVE_DPI = 400

def plot_peak_differences_vs_kappa(csv_file, output_folder):
    """
    Plot the difference between the two Lorentzian peaks vs. Kappa, scaled by J.

    Parameters:
        csv_file (str): Path to the CSV file containing the peak difference data.
        output_folder (str): Path to save the plot.
    """
    # Read the CSV file
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    peak_diff_df = pd.read_csv(csv_file)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate J (half the max peak splitting)
    max_peak_diff = peak_diff_df['Peak Difference (GHz)'].max()
    J = max_peak_diff / 2

    # Scale the data by J
    peak_diff_df['Scaled Kappa'] = peak_diff_df['Kappa'] / J
    peak_diff_df['Scaled Peak Difference'] = peak_diff_df['Peak Difference (GHz)'] / J
    peak_diff_df['Scaled Kappa_unc'] = peak_diff_df['Kappa_unc'] / J
    peak_diff_df['Scaled Peak Difference_unc'] = peak_diff_df['Peak Difference_unc (GHz)'] / J

    # Plot scaled peak difference vs scaled Kappa with error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        peak_diff_df['Scaled Kappa'], peak_diff_df['Scaled Peak Difference'],
        xerr=peak_diff_df['Scaled Kappa_unc'], yerr=peak_diff_df['Scaled Peak Difference_unc'],
        fmt='o', ecolor='red', capsize=4, label='Scaled Peak Splitting', markersize=4
    )

    # Draw a horizontal line at the scaled maximum value (theoretical EP line)
    ax.axvline(x=-2, color='green', linestyle='--', label='Theoretical EP Line')

    # Axis labels and title
    ax.set_xlabel('$K / J$', fontsize=14)
    ax.set_ylabel('Splitting / J', fontsize=14)
    ax.set_title('Scaled Double Lorentzian Peak Splitting vs Scaled Kappa', fontsize=16)
    ax.grid(True)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "scaled_peak_differences_vs_kappa_plot.png")
    plt.savefig(plot_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved scaled peak differences vs. Kappa plot to {plot_path}")

# Example usage
if __name__ == "__main__":
    csv_file = r'C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\analysis\csv\7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc_peak_differences\peak_differences_vs_kappa_7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc.csv'
    output_folder = "."  # Update this path
    plot_peak_differences_vs_kappa(csv_file, output_folder)