# load the first set of data from vna_data_normal_operation_0.000V.csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df1 = pd.read_csv("vna_data_normal_operation_0.000V.csv")
df2 = pd.read_csv("vna_data_nr_mode_0.000V.csv")

# Extract frequency and power data
freq1 = df1['frequency_hz'].values
power1 = df1['power_dBm'].values

freq2 = df2['frequency_hz'].values
power2 = df2['power_dBm'].values

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(freq1, power1, label='Normal Operation')
plt.plot(freq2, power2, label='NR Mode')
plt.plot(freq1, power1 + power2, label='Total Photon Signal', linestyle='--')
plt.xlabel('Frequency (Hz)', fontsize=14)
plt.ylabel('Power (dBm)', fontsize=14)
plt.title('Total Photon Signal Comparison', fontsize=16)
plt.legend()
plt.grid(True)

plt.savefig("total_photon_signal_comparison.png")
