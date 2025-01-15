import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
csv_path = r'C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\analysis\csv\96187cb2-5d3d-40eb-9b3a-6d1d69b7ac0a_peak_differences\peak_differences_vs_kappa_96187cb2-5d3d-40eb-9b3a-6d1d69b7ac0a.csv'
df = pd.read_csv(csv_path)

# Define the filtering range for Kappa
xlim_min, xlim_max = -0.0005, 0

# Filter the data based on the xlim range
filtered_df = df[(df['Kappa'] >= xlim_min) & (df['Kappa'] <= xlim_max)]

# Display the filtered DataFrame
filtered_df.head()

xf = filtered_df['Kappa']
yf = filtered_df['Peak Difference (GHz)']

# Plot the filtered data
plt.plot(xf, yf)

print(yf.max() / 2)
print(yf.min() / 2)