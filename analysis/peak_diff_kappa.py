import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv/7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc_peak_differences/peak_differences_vs_kappa_7ed9f8f6-fd12-4fcd-b14f-57b08dae27fc.csv")
plt.plot(df['Kappa'], df['Peak Difference (GHz)'], 'o-')
plt.xlabel('Kappa (GHz)')
plt.ylabel('Peak Difference (GHz)')
plt.title('Peak Difference vs. Kappa')
plt.grid(True)
plt.savefig('peak_diff_vs_kappa.png', dpi=300)
plt.close()
