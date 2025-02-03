# HOW TO USE `colorplots_delta_x_axis.py`

This script analyzes data from the database + experiment ID, and does the following analysis:

1. Analyzes the YIG-only and Cavity-only data, in order to grab Kappa_C, Kappa_Y, Omega_Y, Omega_C at each value 
of current. This is done by fitting the data to an lmfit LorentzianModel, and extracting the parameters.
2. Calculates Delta and Kappa (but Delta is more important for the NREP) as a function of current.
3. Creates the necessary colorplots - Using Current as the X axis for the "raw" colorplot (which is the same as the external)
file that does that as well, **and also** the Detuning based colorplot
4. Uses a double-lorentzian model to fit the NR data, in order to extract the peak locations (and linewidhts) of the hybridized data. This
raw peak data is plotted as its own "colorplot" (although there are no colors, it's just the peak locations)


# How to actually use it

1. Get the experiment ID from the database using:
```sql
select distinct(expr.experiment_id) from expr
```
Choose the experiment ID that is wanted, and use that to fill out:

```python 
db_path = "../databases/NR_SUN_NITE_PPP_FINE.db"  # adjust as needed
experiment_id = "294962de-dd80-49c6-81b5-394ae97b5838"
```

2. Run the script, USING THE FOLLOWING SETTINGS FOR FREQUENCY BOUNDS:

```python 
    # Frequency limits (Hz)
    colorplot_freq_min = 1e9
    colorplot_freq_max = 99e9

    cavity_freq_min = 1e9
    cavity_freq_max = 99e9

    yig_freq_min = 1e9
    yig_freq_max = 99e9
```

Also, ensure that `DEBUG_MODE = True` for this run, otherwise you'll be in trouble

We do this for an important reason - You need to look at the FULL data (with debug mode on), to see where
the peaks actually are. This is because if you have too much data **around the peaks, the lorentzian model will fail.**

So, you need to choose an adequate window for the cavity, yig, and full hybridization, such that you can get good fits!

3. Run the script, look at all the outputs

Now, you should have an actual output. 