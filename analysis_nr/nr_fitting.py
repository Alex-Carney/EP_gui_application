# fitting.py
import numpy as np
from scipy.signal import find_peaks, peak_widths
from lmfit.models import LorentzianModel
from typing import TypedDict, Union, Optional, Any
from stages import Fitter

OVERFITTING_AMPLITUDE_DIFFERENCE_THRESHOLD = 5  # dB


class FitTraceResult(TypedDict):
    current: float
    omega: float
    omega_unc: float
    kappa: float
    kappa_unc: float
    fit_type: str
    fit_result: Any
    x_fit: Any


def single_lorentzian_fit(x, y, center_guess):
    model = LorentzianModel(prefix="lz_")
    pars = model.make_params()

    # Slightly perturb the initial guess to avoid perfect convergence
    sigma_guess = 0.001 + np.random.uniform(-1e-6, 1e-6)  # Tiny randomness
    center_guess += np.random.uniform(-1e-6, 1e-6)  # Tiny randomness
    amp_guess = y.max() * np.pi * sigma_guess

    pars["lz_center"].set(value=center_guess, min=x.min(), max=x.max())
    pars["lz_amplitude"].set(value=amp_guess, min=0)
    pars["lz_sigma"].set(value=sigma_guess, min=1e-6)

    try:
        # Force covariance matrix calculation
        result = model.fit(y, pars, x=x, calc_covar=True)

        # Check if stderr is missing and manually set it
        if result.covar is None:
            print(f"\n\n\n WARNING: Covariance matrix is None for center={center_guess:.6f} \n\n\n")
            result.params["lz_center"].stderr = -1  # Mark as invalid
            result.params["lz_sigma"].stderr = -1

    except Exception as e:
        print("Single Lorentzian fit error:", e)
        return None

    return result


def fit_trace(current_value, frequencies, power_dbm) -> FitTraceResult:
    freqs_ghz = frequencies / 1e9
    peaks, _ = find_peaks(power_dbm, prominence=0.1)
    if len(peaks) == 0:
        print('\n WARNING - No peaks found in power trace!!')
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": None}
    peak_idx = peaks[0]
    center_guess = freqs_ghz[peak_idx]
    power_linear = 10 ** (power_dbm / 10)
    widths, _, _, _ = peak_widths(power_linear, [peak_idx], rel_height=0.5)
    if len(widths) == 0:
        print('\n WARNING - No peaks WIDTHS found in power trace!!')
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": None}
    freq_step = (freqs_ghz[-1] - freqs_ghz[0]) / (len(freqs_ghz) - 1)
    fwhm_guess = widths[0] * freq_step
    fit_range = 5 * fwhm_guess
    mask = (freqs_ghz >= center_guess - fit_range) & (freqs_ghz <= center_guess + fit_range)
    if mask.sum() < 5:
        print('\n WARNING - Not enough points in fit range!!')
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": None}
    x_fit = freqs_ghz[mask]
    y_fit = 10 ** (power_dbm[mask] / 10)
    fit_result = single_lorentzian_fit(x_fit, y_fit, center_guess)
    if fit_result is None or "lz_center" not in fit_result.params or "lz_sigma" not in fit_result.params:
        print('\n WARNING - Fit result is None -- UNABLE TO FIT SINGLE LORENTZIAN!!')
        return {"current": current_value, "omega": np.nan, "omega_unc": np.nan,
                "kappa": np.nan, "kappa_unc": np.nan, "fit_type": "single", "fit_result": None, "x_fit": x_fit}
    center = fit_result.params["lz_center"].value
    center_unc = fit_result.params["lz_center"].stderr if fit_result.params["lz_center"].stderr is not None else np.nan
    sigma = fit_result.params["lz_sigma"].value
    sigma_unc = fit_result.params["lz_sigma"].stderr if fit_result.params["lz_sigma"].stderr is not None else np.nan
    fwhm = 2 * sigma
    fwhm_unc = 2 * sigma_unc if not np.isnan(sigma_unc) else np.nan
    return {"current": current_value, "omega": center, "omega_unc": center_unc,
            "kappa": fwhm, "kappa_unc": fwhm_unc, "fit_type": "single", "fit_result": fit_result, "x_fit": x_fit}


def process_all_traces(power_grid, currents, frequencies):
    results = []
    for i, current in enumerate(currents):
        power_dbm = power_grid[i, :]
        res = fit_trace(current, frequencies, power_dbm)
        results.append(res)
    return results


def compute_delta(cavity_df, yig_df):
    # Assume cavity_df and yig_df are DataFrames with keys "omega", etc.
    cavity_df = cavity_df.rename(columns={"omega": "omega_c", "omega_unc": "omega_c_unc",
                                          "kappa": "kappa_c", "kappa_unc": "kappa_c_unc"})
    yig_df = yig_df.rename(columns={"omega": "omega_y", "omega_unc": "omega_y_unc",
                                    "kappa": "kappa_y", "kappa_unc": "kappa_y_unc"})
    merged = cavity_df.merge(yig_df, on="current", how="inner")
    merged["Delta"] = merged["omega_c"] - merged["omega_y"]
    merged["Delta_unc"] = np.sqrt(merged["omega_c_unc"] ** 2 + merged["omega_y_unc"] ** 2)
    return merged


# --- NR Fitting Functions ---
def double_lorentzian_fit_NR(x, y, guess1, guess2):
    from lmfit.models import LorentzianModel
    lz1 = LorentzianModel(prefix="lz1_")
    lz2 = LorentzianModel(prefix="lz2_")
    mod = lz1 + lz2
    sigma_guess = 0.001
    amp_guess = y.max() * np.pi * sigma_guess
    pars = mod.make_params()
    if guess1 > guess2:
        guess1, guess2 = guess2, guess1
    pars["lz1_center"].set(value=guess1, min=x.min(), max=guess2)
    pars["lz2_center"].set(value=guess2, min=guess2, max=x.max())
    pars["lz1_amplitude"].set(value=amp_guess, min=0)
    pars["lz1_sigma"].set(value=sigma_guess, min=1e-6)
    pars["lz2_amplitude"].set(value=amp_guess, min=0)
    pars["lz2_sigma"].set(value=sigma_guess, min=1e-6)
    try:
        out = mod.fit(y, pars, x=x)
    except Exception as e:
        print("Double Lorentzian fit failed:", e)
        return None
    return out


def theory_supported_NR_fit(current_value, frequencies, power_dbm, sim_trace, sim_peaks_idx):
    # ------------------ FIND PEAKS IN SIMULATED DATA ------------------
    freqs_ghz = frequencies / 1e9
    sim_peaks_freq = np.sort(freqs_ghz[sim_peaks_idx])
    power_linear = 10 ** (power_dbm / 10)
    # ------------------ FIT SINGLE AND/OR DOUBLE LORENTZIANS ------------------
    single_fit = None
    if len(sim_peaks_idx) == 1:
        # If 1 peak is found, then we try BOTH a Single and Double fit for the Expr data
        single_fit = single_lorentzian_fit(freqs_ghz, power_linear, sim_peaks_freq[0])
    guess1 = sim_peaks_freq[0]
    guess2 = sim_peaks_freq[1] if len(sim_peaks_idx) > 1 else guess1 + 0.001
    double_fit = double_lorentzian_fit_NR(freqs_ghz, power_linear, guess1, guess2)

    # Decide which fit to choose
    if double_fit is None or single_fit is None:
        chosen = single_fit if single_fit is not None else double_fit
        chosen_type = "single" if single_fit is not None else "double"
    else:
        if hasattr(double_fit, "redchi") and hasattr(single_fit, "redchi"):
            # First, compare redchi values
            if double_fit.redchi < single_fit.redchi:
                # For double fit to be accepted, check that the amplitudes are similar.
                # Get the amplitudes from the double fit parameters.
                amp1 = double_fit.params["lz1_amplitude"].value
                amp2 = double_fit.params["lz2_amplitude"].value
                # Convert amplitudes to dB (make sure amplitudes are > 0 to avoid log10 issues)
                amp1_db = 10 * np.log10(amp1) if amp1 > 0 else -np.inf
                amp2_db = 10 * np.log10(amp2) if amp2 > 0 else -np.inf
                if abs(amp1_db - amp2_db) <= OVERFITTING_AMPLITUDE_DIFFERENCE_THRESHOLD:

                    # The amplitudes are similar (within 10 dB) → accept the double fit.
                    chosen = double_fit
                    chosen_type = "double"
                else:
                    print(f"Amplitudes: {amp1_db:.2f} dB, {amp2_db:.2f} dB (diff: {abs(amp1_db - amp2_db):.2f} dB) → "
                          f"overfitting detected on current {current_value}")
                    # Amplitudes differ by more than 10 dB → overfitting, so choose the single fit.
                    chosen = single_fit
                    chosen_type = "single"
            else:
                chosen = single_fit
                chosen_type = "single"
        else:
            chosen = single_fit
            chosen_type = "single"

    if chosen_type == "single":
        center = chosen.params["lz_center"].value
        center_unc = chosen.params["lz_center"].stderr if chosen.params["lz_center"].stderr is not None else np.nan
        result = {"current": current_value, "fit_type": "single", "omega": center, "omega_unc": center_unc,
                  "fit_result": chosen, "x_fit": freqs_ghz}
    else:
        peak1 = chosen.params["lz1_center"].value
        peak1_unc = chosen.params["lz1_center"].stderr if chosen.params["lz1_center"].stderr is not None else np.nan
        peak2 = chosen.params["lz2_center"].value
        peak2_unc = chosen.params["lz2_center"].stderr if chosen.params["lz2_center"].stderr is not None else np.nan
        result = {"current": current_value, "fit_type": "double",
                  "peak1": peak1, "peak1_unc": peak1_unc,
                  "peak2": peak2, "peak2_unc": peak2_unc,
                  "fit_result": chosen, "x_fit": freqs_ghz}
    return result


def iterative_NR_fit(current_value, frequencies, power_dbm, initial_guesses=None):
    """
    For one NR trace, perform both a single-Lorentzian and a double-Lorentzian fit.
    If initial_guesses (a two-element list [guess1, guess2] in GHz) is provided, use it;
    otherwise, use basic peak finding.
    Compare fits via redchi (if available) and return the one with lower redchi.
    """
    freqs_ghz = frequencies / 1e9
    power_linear = 10 ** (power_dbm / 10)
    if initial_guesses is not None:
        guess1, guess2 = initial_guesses
    else:
        peaks, props = find_peaks(power_dbm, height=-30, prominence=0.025, distance=25)
        if len(peaks) == 0:
            return {"current": current_value, "fit_type": None, "fit_result": None, "x_fit": None,
                    "initial_guesses": None}
        elif len(peaks) == 1:
            guess1 = freqs_ghz[peaks[0]]
            guess2 = guess1 + 0.001
        else:
            prominences = props["prominences"]
            sorted_idx = np.argsort(prominences)[::-1]
            best_two = peaks[sorted_idx[:2]]
            best_two = np.sort(best_two)
            guess1 = freqs_ghz[best_two[0]]
            guess2 = freqs_ghz[best_two[1]]
    single_fit = single_lorentzian_fit(freqs_ghz, power_linear, guess1)
    double_fit = double_lorentzian_fit_NR(freqs_ghz, power_linear, guess1, guess2)
    if double_fit is None or single_fit is None:
        chosen = single_fit if single_fit is not None else double_fit
        chosen_type = "single" if single_fit is not None else "double"
    else:
        if hasattr(double_fit, "redchi") and hasattr(single_fit, "redchi"):
            if double_fit.redchi < single_fit.redchi:
                chosen = double_fit
                chosen_type = "double"
            else:
                chosen = single_fit
                chosen_type = "single"
        else:
            chosen = single_fit
            chosen_type = "single"
    if chosen_type == "single":
        center = chosen.params["lz_center"].value
        center_unc = chosen.params["lz_center"].stderr if chosen.params["lz_center"].stderr is not None else np.nan
        result = {"current": current_value, "fit_type": "single", "omega": center, "omega_unc": center_unc,
                  "fit_result": chosen, "x_fit": freqs_ghz, "initial_guesses": [guess1, guess2]}
    else:
        peak1 = chosen.params["lz1_center"].value
        peak1_unc = chosen.params["lz1_center"].stderr if chosen.params["lz1_center"].stderr is not None else np.nan
        peak2 = chosen.params["lz2_center"].value
        peak2_unc = chosen.params["lz2_center"].stderr if chosen.params["lz2_center"].stderr is not None else np.nan
        result = {"current": current_value, "fit_type": "double",
                  "peak1": peak1, "peak1_unc": peak1_unc,
                  "peak2": peak2, "peak2_unc": peak2_unc,
                  "fit_result": chosen, "x_fit": freqs_ghz, "initial_guesses": [guess1, guess2]}
    return result


def fit_NR_trace(current_value, frequencies, power_dbm):
    """Wrapper for non-iterative NR fit (for overlay use)."""
    return iterative_NR_fit(current_value, frequencies, power_dbm, initial_guesses=None)
