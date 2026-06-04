import numpy as np
from scipy.stats import linregress

def detrended_fluctuation(series):
    """Compute Hurst exponent using DFA."""
    n = len(series)
    if n < 20:
        return 0.5
    # Cumulative sum
    y = np.cumsum(series - np.mean(series))
    # Box sizes
    scales = np.logspace(np.log10(4), np.log10(n//4), 10).astype(int)
    scales = np.unique(scales)
    fluct = []
    for scale in scales:
        n_seg = n // scale
        rms = []
        for i in range(n_seg):
            seg = y[i*scale:(i+1)*scale]
            x = np.arange(len(seg))
            slope, intercept, _, _, _ = linregress(x, seg)
            trend = slope * x + intercept
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        fluct.append(np.mean(rms))
    # Fit log-log
    if len(scales) < 2:
        return 0.5
    coeffs = np.polyfit(np.log(scales), np.log(fluct), 1)
    H = coeffs[0]
    return H

def fou_mean_reversion(returns, macro_series):
    """
    Estimate fOU parameters: mean-reversion speed θ, volatility σ, and macro-dependent mean.
    Returns current macro-conditional mean (score).
    """
    # Align lengths
    min_len = min(len(returns), len(macro_series))
    returns = returns[:min_len]
    macro_series = macro_series[:min_len]
    if len(returns) < 5:
        return 0.0
    # Compute Hurst exponent
    H = detrended_fluctuation(returns)
    # If H is > 0.5 (persistent) or < 0.5 (anti-persistent), but we treat all as rough if H < threshold
    # Roughness is not critical for score; we'll use standard OU regression.
    # Estimate theta and sigma via linear regression of increments on level.
    # Discretised fOU: X_{t+1} - X_t = -θ (X_t - μ) Δt + σ ΔW^H
    # We approximate ΔW^H as random but ignore fractional correlation.
    # Instead, we directly regress returns on lagged deviations: return_t = -θ * (X_{t-1} - μ) + noise.
    # However μ is macro-dependent: μ = β0 + β1 * macro_t
    # We can solve jointly: return_t = -θ * (X_{t-1} - (β0 + β1*macro_{t-1})) + ε_t
    # Rearranged: return_t = -θ X_{t-1} + θβ0 + θβ1 * macro_{t-1} + ε_t
    # So we can regress returns on lagged price X and lagged macro.
    # X here is the log price (cumulative log return). We need log price level.
    # We'll compute cumulative log price starting at 0.
    X = np.cumsum(returns)
    X_lag = X[:-1]
    macro_lag = macro_series[:-1]
    y = returns[1:]
    # Regression: y = alpha1 * X_lag + alpha2 * macro_lag + alpha3? Actually we have intercept.
    # But we have three terms: constant, X_lag, macro_lag.
    A = np.column_stack([np.ones(len(X_lag)), X_lag, macro_lag])
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except:
        return 0.0
    intercept, theta_x, theta_macro = coeff
    # theta = -theta_x, because return_t = -θ (X_{t-1} - μ) = -θ X_{t-1} + θ μ
    theta = -theta_x if theta_x != 0 else 0.01
    # μ = (intercept + θ_macro * macro) / θ
    # For current macro (last value), compute μ_today
    current_macro = macro_series[-1]
    mu_today = (intercept + theta_macro * current_macro) / theta if theta != 0 else 0.0
    return float(mu_today)

def rough_fou_score(returns, macro_series):
    """Return the macro-conditional mean level (expected return drift)."""
    return fou_mean_reversion(returns, macro_series)
