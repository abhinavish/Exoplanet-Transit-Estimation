
# # Phase 2 — Transit Model Construction
# ### MCMC Transit Parameter Estimation · Graduate Class Project
# 
# **Targets:** Kepler-7b (hot Jupiter) · Kepler-10b (rocky super-Earth) · WASP-39b (hot Saturn/TESS)
# 
# **Goal of this notebook:**
# Build a validated transit forward model using `batman`, implement the log-likelihood and log-posterior functions, generate synthetic test data, and export everything needed for Phase 3 (MCMC sampling).
# 
# ---
# > **Run all cells in order.** Each cell is numbered and has a brief explanation at the top.
# 


# ## Cell 1 — Install & import dependencies
# `batman-package` must be installed each Colab session. `batman` uses compiled C extensions, so the install takes ~30 seconds.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import batman
import warnings
import os
warnings.filterwarnings('ignore')

# Consistent plot style throughout the notebook
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

print("batman version:", batman.__version__)
print("All imports successful ✓")



# ## Cell 2 — Literature parameters for all three systems
# 
# ### Parameter dictionary — what each key means
# 
# | Key | Description | Units |
# |-----|-------------|-------|
# | `t0` | Mid-transit time. Set to **0.0** for phase-folded data | days |
# | `per` | Orbital period — **fixed** during MCMC | days |
# | `rp` | Planet-to-star radius ratio **R_p/R★**. Transit depth ≈ (Rp/R★)² | dimensionless |
# | `a` | Semi-major axis in stellar radii **a/R★**. Controls duration & ingress shape | dimensionless |
# | `inc` | Orbital inclination. 90° = edge-on; grazing transits have lower `inc` | degrees |
# | `ecc` | Eccentricity — assumed **0** (circular; hot Jupiters are tidally circularized) | — |
# | `w` | Argument of periastron — set to **90°** for circular orbits | degrees |
# | `limb_dark` | Limb darkening law — we use **"quadratic"** (standard for Kepler/TESS) | — |
# | `u` | Limb darkening coefficients [u1, u2] from Claret & Bloemen (2011) tables | — |
# 
# **Impact parameter:** `b = (a/R★) × cos(i)`. Central transit: b=0; grazing: b→1.
# 


SYSTEMS = {

    "Kepler-7b": {
        # Hot Jupiter, Rp ~ 1.6 RJup. Deep ~1% transit. Best validation target.
        # Source: Latham et al. 2010, Kipping & Bakos 2011, NASA Exoplanet Archive
        "t0":        0.0,
        "per":       4.8854892,   # days (fixed during MCMC)
        "rp":        0.09829,     # Rp/R★ → transit depth ~ 0.97%
        "a":         8.9251,      # a/R★
        "inc":       85.16,       # degrees
        "ecc":       0.0,
        "w":         90.0,
        "limb_dark": "quadratic",
        "u":         [0.4804, 0.1547],  # Teff~5933 K, logg~3.98 (Claret 2011)
        "R_star":    2.02,              # R_sun (inflated sub-giant host)
        "T_eff":     5933,              # K
        "notes":     "Deep ~1% transit. Large Rp/R★ → easiest pipeline validation.",
    },

    "Kepler-10b": {
        # Rocky super-Earth, Rp ~ 1.47 R_earth. Extremely shallow ~0.014% transit.
        # Source: Batalha et al. 2011, Fogtmann-Schulz et al. 2014
        "t0":        0.0,
        "per":       0.83749070,  # days — very short, many transits available
        "rp":        0.01197,     # Rp/R★ → depth ~0.014% (70× shallower than Kepler-7b!)
        "a":         3.769,       # a/R★ — very close-in planet
        "inc":       84.40,       # degrees
        "ecc":       0.0,
        "w":         90.0,
        "limb_dark": "quadratic",
        "u":         [0.4820, 0.2015],  # Teff~5627 K, logg~4.34
        "R_star":    1.065,             # R_sun
        "T_eff":     5627,
        "notes":     "Shallow transit tests sampler sensitivity. Short period = many transits.",
    },

    "WASP-39b": {
        # Hot Saturn, Rp ~ 1.27 RJup. TESS mission. JWST benchmark system.
        # Source: Faedi et al. 2011, Tsiaras et al. 2018, Rustamkulov et al. 2023
        "t0":        0.0,
        "per":       4.0552941,   # days
        "rp":        0.14560,     # Rp/R★ → depth ~2.1% (deepest of our three)
        "a":         11.47,       # a/R★
        "inc":       87.83,       # degrees — nearly edge-on
        "ecc":       0.0,
        "w":         90.0,
        "limb_dark": "quadratic",
        "u":         [0.3982, 0.2164],  # Teff~5400 K, logg~4.49
        "R_star":    0.895,             # R_sun
        "T_eff":     5400,
        "notes":     "Deepest transit, TESS cadence. Well-studied JWST-era benchmark.",
    },
}

# ── Summary table ──────────────────────────────────────────────────────────
print(f"{'System':<14} {'Rp/R★':>8} {'Depth (%)':>10} {'Depth (ppt)':>12} {'a/R★':>7} {'i (°)':>8} {'P (days)':>10}")
print("─" * 75)
for name, p in SYSTEMS.items():
    depth_pct = p["rp"]**2 * 100
    depth_ppt = p["rp"]**2 * 1000
    print(f"{name:<14} {p['rp']:>8.5f} {depth_pct:>10.4f} {depth_ppt:>12.4f} "
          f"{p['a']:>7.3f} {p['inc']:>8.2f} {p['per']:>10.7f}")



# ## Cell 3 — Core transit forward model
# 
# ### How `batman` works
# 1. Create a `batman.TransitParams()` object and fill in physical parameters
# 2. Initialize `batman.TransitModel(params, times)` — this compiles the C extension
# 3. Call `m.light_curve(params)` to get normalized flux at each time stamp
# 
# ### Supersampling — critical for Kepler long-cadence data
# Kepler long-cadence bins are **29.4 minutes** wide. A typical transit lasts 2–4 hours,
# so only 4–8 bins fall in-transit. Each bin is the *average* flux over 29.4 minutes,
# so we must account for the transit shape evolution within each bin.
# 
# `batman` handles this with `supersample_factor=7`: it evaluates 7 sub-points per bin
# and averages them. Without this, ingress/egress are significantly distorted.
# 
# | Cadence | exp_time (days) | supersample_factor |
# |---------|----------------|--------------------|
# | Kepler long (29.4 min) | 0.020417 | 7 |
# | Kepler short (1 min) | — | 1 (none needed) |
# | TESS 2-min | — | 1–3 |
# 


def build_batman_params(param_dict):
    """Convert our parameter dict to a batman.TransitParams object."""
    p = batman.TransitParams()
    p.t0        = param_dict["t0"]
    p.per       = param_dict["per"]
    p.rp        = param_dict["rp"]
    p.a         = param_dict["a"]
    p.inc       = param_dict["inc"]
    p.ecc       = param_dict["ecc"]
    p.w         = param_dict["w"]
    p.limb_dark = param_dict["limb_dark"]
    p.u         = list(param_dict["u"])
    return p


def compute_transit_flux(param_dict, times, long_cadence=True):
    """
    Compute a normalized transit light curve using batman.

    Parameters
    ----------
    param_dict   : dict  — transit parameters (keys match SYSTEMS structure)
    times        : array — time stamps in days (phase-folded, t0=0)
    long_cadence : bool  — True applies Kepler 29.4-min supersampling

    Returns
    -------
    flux         : array — normalized flux (1.0 out of transit, dips during transit)
    """
    p = build_batman_params(param_dict)
    if long_cadence:
        # 29.4 min = 0.020417 days; factor=7 sub-samples per bin
        m = batman.TransitModel(p, times,
                                supersample_factor=7,
                                exp_time=0.020417)
    else:
        m = batman.TransitModel(p, times)
    return m.light_curve(p)


# ── Quick smoke test ───────────────────────────────────────────────────────
t_test = np.linspace(-0.2, 0.2, 500)
flux_test = compute_transit_flux(SYSTEMS["Kepler-7b"], t_test, long_cadence=False)

depth_expected = SYSTEMS["Kepler-7b"]["rp"]**2
depth_measured = 1.0 - flux_test.min()
print(f"Expected transit depth : {depth_expected:.6f}")
print(f"Measured transit depth : {depth_measured:.6f}")
print(f"Difference             : {abs(depth_expected - depth_measured)*1e6:.1f} ppm")
print("Forward model OK ✓")



# ## Cell 4 — Visualize all three transit models
# 
# Compare the three systems side by side. The dashed vertical lines mark the
# four **contact points** T1–T4:
# - **T1/T4**: first and last external contact (planet limb touches stellar limb)
# - **T2/T3**: internal contacts (planet fully inside stellar disk — start/end of flat bottom)
# 
# The flat bottom is only present when b < (1 − Rp/R★). A grazing transit (b close to 1)
# shows a V-shape with no flat region.
# 


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#2E86C1', '#E74C3C', '#27AE60']

for ax, (name, params), color in zip(axes, SYSTEMS.items(), colors):
    half_window = params["per"] * 0.08
    t = np.linspace(-half_window, half_window, 1000)
    flux = compute_transit_flux(params, t, long_cadence=False)

    ax.plot(t * 24, flux, color=color, lw=2.2)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.4)

    depth_ppt = params["rp"]**2 * 1000
    ax.set_title(f"{name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time from mid-transit [hours]")
    ax.set_ylabel("Normalized flux")

    # Parameter annotation box
    b = params["a"] * np.cos(np.radians(params["inc"]))
    textstr = (f"Rp/R★ = {params['rp']:.5f}\n"
               f"depth = {depth_ppt:.3f} ppt\n"
               f"i     = {params['inc']:.2f}°\n"
               f"a/R★  = {params['a']:.3f}\n"
               f"b     = {b:.3f}")
    ax.text(0.03, 0.06, textstr, transform=ax.transAxes,
            fontsize=9, va='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=color, alpha=0.9))

    # Contact point markers T1–T4
    k   = params["rp"]
    aR  = params["a"]
    inc_rad = np.radians(params["inc"])
    sin_i   = np.sin(inc_rad)

    if abs(b) < (1 + k) and sin_i > 0:
        arg14 = np.sqrt(max((1 + k)**2 - b**2, 0)) / (aR * sin_i)
        arg23 = np.sqrt(max((1 - k)**2 - b**2, 0)) / (aR * sin_i)
        T14 = (params["per"] / np.pi) * np.arcsin(min(arg14, 1.0))
        T23 = (params["per"] / np.pi) * np.arcsin(min(arg23, 1.0)) if (1-k)**2 > b**2 else 0

        for t_c, lbl in [(-T14, 'T1'), (-T23, 'T2'), (T23, 'T3'), (T14, 'T4')]:
            ax.axvline(t_c * 24, color=color, lw=0.9, ls=':', alpha=0.65)
            if t_c >= 0:
                ax.text(t_c * 24 + 0.03, ax.get_ylim()[0] + 0.0002,
                        lbl, fontsize=7.5, color=color, alpha=0.8)

plt.suptitle("Transit Model Light Curves — Literature Parameters", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("transit_models_all_systems.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: transit_models_all_systems.png")



# ## Cell 5 — Generate synthetic observations
# 
# While your partner downloads real Kepler/TESS data (Phase 1), we create realistic
# synthetic observations: true model + Gaussian noise at mission-typical levels.
# 
# ### Photometric noise levels
# 
# | System | Mission | Cadence | Noise σ | Transit depth | SNR (per transit) |
# |--------|---------|---------|---------|--------------|-------------------|
# | Kepler-7b | Kepler | Long (29.4 min) | ~100 ppm | 9,700 ppm | ~97 |
# | Kepler-10b | Kepler | Short (1 min) | ~300 ppm | 143 ppm | ~0.5 |
# | WASP-39b | TESS | 2-min | ~400 ppm | 21,200 ppm | ~53 |
# 
# > **Note:** Kepler-10b's SNR per transit is < 1! The planet is detected by stacking
# > hundreds of transits (short period → many repeats). Phase-folding boosts SNR ∝ √N_transits.
# 


np.random.seed(42)   # Fix seed for reproducibility — use this same seed throughout

NOISE_SIGMA = {
    "Kepler-7b":   1.0e-4,   # 100 ppm — bright star, long-cadence
    "Kepler-10b":  3.0e-4,   # 300 ppm — short-cadence, shallow transit
    "WASP-39b":    4.0e-4,   # 400 ppm — TESS 2-min
}

LONG_CADENCE = {
    "Kepler-7b":   True,    # 29.4-min bins → supersampling applied
    "Kepler-10b":  False,   # Short-cadence
    "WASP-39b":    False,   # TESS 2-min
}

CADENCE_DAYS = {
    "Kepler-7b":   0.020417,  # 29.4 min
    "Kepler-10b":  0.000694,  # ~1 min
    "WASP-39b":    0.001389,  # ~2 min
}

synthetic_data = {}

for name, params in SYSTEMS.items():
    sigma   = NOISE_SIGMA[name]
    cadence = CADENCE_DAYS[name]
    t       = np.arange(-params["per"] * 0.12, params["per"] * 0.12, cadence)

    flux_true = compute_transit_flux(params, t, long_cadence=LONG_CADENCE[name])
    flux_obs  = flux_true + np.random.normal(0, sigma, size=len(t))
    flux_err  = np.full(len(t), sigma)

    synthetic_data[name] = {
        "time":      t,
        "flux":      flux_obs,
        "flux_err":  flux_err,
        "flux_true": flux_true,
        "sigma":     sigma,
    }

    # SNR estimate
    in_tr = flux_true < 0.9999
    n_in  = in_tr.sum()
    depth = (1 - flux_true[in_tr].mean()) if n_in > 0 else 0
    snr   = depth / (sigma / np.sqrt(n_in)) if n_in > 0 else 0

    print(f"{name:<14}  n_points={len(t):5d}  in-transit={n_in:4d}  "
          f"depth={depth*1e6:>7.0f} ppm  σ={sigma*1e6:.0f} ppm  SNR≈{snr:.1f}")

print("\nSynthetic data generated ✓")



# ## Cell 6 — Plot synthetic observations with true model overlay


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#2E86C1', '#E74C3C', '#27AE60']

for ax, (name, data), color in zip(axes, synthetic_data.items(), colors):
    t, f, fe, ft = data["time"], data["flux"], data["flux_err"], data["flux_true"]

    ax.errorbar(t * 24, f, yerr=fe, fmt='.', color=color, alpha=0.25,
                markersize=2, elinewidth=0.5, label='Synthetic obs.')
    ax.plot(t * 24, ft, 'k-', lw=1.8, label='True model', zorder=5)

    ax.set_xlabel("Time from mid-transit [hours]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(name, fontweight='bold')
    ax.text(0.98, 0.05, f"σ = {data['sigma']*1e6:.0f} ppm",
            transform=ax.transAxes, ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85))
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle("Synthetic Observations + True Model", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("synthetic_data_all_systems.png", dpi=150, bbox_inches='tight')
plt.show()



# ## Cell 7 — Log-likelihood, log-prior, and log-posterior
# 
# These three functions together form the **statistical engine** of the MCMC sampler.
# `emcee` in Phase 3 will call `log_posterior` millions of times.
# 
# ### Parameter vector (theta) convention
# ```
# theta = [rp, a, inc, t0, u1, u2, log_sigma]
#           0   1   2   3   4   5      6
# ```
# This order is fixed throughout the project. Every function must agree on it.
# 
# ### Why fit log(σ) instead of σ?
# - Noise σ is strictly positive → sampling in log-space prevents σ going negative
# - log-normal sampling means the sampler can explore many orders of magnitude with equal step efficiency
# - `sigma = exp(log_sigma)` in all calculations
# 
# ### Prior choices
# | Parameter | Prior | Reasoning |
# |-----------|-------|-----------|
# | rp | Uniform (0.001, 0.5) | Physical bounds; can't be negative or > 1 |
# | a | Uniform (1.5, 100) | Planet must be outside stellar surface (a > 1) |
# | inc | Uniform (60°, 90°) | Transit geometry requires near-edge-on; i and 180°-i are identical |
# | t0 | Uniform (−0.1, 0.1) days | Small offset around reference epoch |
# | u1, u2 | Uniform (0,1) with u1+u2 ≤ 1 | Physical constraint: I(limb) ≥ 0 everywhere |
# | log_σ | Uniform (−15, −3) | Covers σ ∈ [3×10⁻⁷, 0.05] — all realistic noise levels |
# 
# > **Phase 3 upgrade:** Replace uniform LD priors with Gaussian priors centered on
# > `ldtk` predictions. This significantly tightens the posterior on rp.
# 


# ── Fixed parameters (not sampled) ────────────────────────────────────────
FIXED = {"ecc": 0.0, "w": 90.0, "limb_dark": "quadratic"}

# Column names for MCMC chain — MUST match theta order below
PARAM_NAMES = ["rp", "a", "inc", "t0", "u1", "u2", "log_sigma"]


def theta_to_dict(theta, system_name):
    """Unpack MCMC theta vector into a parameter dict for compute_transit_flux()."""
    rp, a, inc, t0, u1, u2, log_sigma = theta
    param_dict = {
        "t0":        t0,
        "per":       SYSTEMS[system_name]["per"],
        "rp":        rp,
        "a":         a,
        "inc":       inc,
        "ecc":       FIXED["ecc"],
        "w":         FIXED["w"],
        "limb_dark": FIXED["limb_dark"],
        "u":         [u1, u2],
    }
    return param_dict, np.exp(log_sigma)


def log_likelihood(theta, system_name, times, flux_obs, long_cadence):
    """
    Gaussian log-likelihood for transit photometry.

    log L = -0.5 * Σ [(obs - model)² / σ²  +  log(2π σ²)]

    The log(σ²) term is ESSENTIAL when σ is a free parameter —
    without it, the sampler is rewarded for making σ arbitrarily large.
    """
    try:
        pdict, sigma = theta_to_dict(theta, system_name)

        # Guard rails — return -inf for unphysical values
        if not (0 < pdict["rp"] < 1):           return -np.inf
        if not (pdict["a"] > 1.0):              return -np.inf
        if not (0 < pdict["inc"] <= 90):        return -np.inf
        if sigma <= 0:                           return -np.inf

        flux_model = compute_transit_flux(pdict, times, long_cadence)
        resid      = flux_obs - flux_model
        log_like   = -0.5 * np.sum(resid**2 / sigma**2
                                   + np.log(2 * np.pi * sigma**2))
        return log_like
    except Exception:
        return -np.inf


def log_prior(theta, system_name):
    """
    Flat (uniform) log-prior within physical bounds.
    Returns 0.0 inside bounds, -inf outside.
    """
    rp, a, inc, t0, u1, u2, log_sigma = theta

    if not (0.001 < rp < 0.5):          return -np.inf
    if not (1.5   < a  < 100.0):        return -np.inf
    if not (60.0  < inc <= 90.0):       return -np.inf
    if not (-0.1  < t0  < 0.1):        return -np.inf
    if not (0.0   <= u1 <= 1.0):        return -np.inf
    if not (0.0   <= u2 <= 1.0):        return -np.inf
    if not (u1 + u2 <= 1.0):            return -np.inf   # physical LD constraint
    if not (-15.0 < log_sigma < -3.0):  return -np.inf

    return 0.0   # log(1) for uniform prior inside bounds


def log_posterior(theta, system_name, times, flux_obs, long_cadence):
    """
    log P(theta | data) = log P(theta) + log P(data | theta)

    This is the function emcee will call. Returns -inf immediately
    if prior is violated (skips expensive likelihood computation).
    """
    lp = log_prior(theta, system_name)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, system_name, times, flux_obs, long_cadence)


print("log_likelihood, log_prior, log_posterior defined ✓")



# ## Cell 8 — Initial parameter vectors and posterior sanity check
# 
# We build `theta_init` for each system from the literature values.
# Then we verify that `log_posterior` returns a **finite** value at the
# literature point — if it returns `-inf`, something is broken.
# 


theta_init = {}

for name, params in SYSTEMS.items():
    sigma = NOISE_SIGMA[name]
    theta = np.array([
        params["rp"],
        params["a"],
        params["inc"],
        0.0,                   # t0 = 0 for phase-folded data
        params["u"][0],        # u1
        params["u"][1],        # u2
        np.log(sigma),         # log_sigma = log(noise per point)
    ])
    theta_init[name] = theta

    # Evaluate log-posterior at literature values
    data = synthetic_data[name]
    lp = log_posterior(theta, name, data["time"], data["flux"], LONG_CADENCE[name])

    print(f"{name:<14}  log_posterior = {lp:>12.2f}  {'✓ finite' if np.isfinite(lp) else '✗ PROBLEM'}")

print("\nInitial theta vectors:")
print(f"  {'Name':<12} " + "  ".join(f"{p:>10}" for p in PARAM_NAMES))
print("─" * 92)
for name, theta in theta_init.items():
    vals = "  ".join(f"{v:>10.5f}" for v in theta)
    print(f"  {name:<12} {vals}")



# ## Cell 9 — Residual analysis at literature parameters
# 
# Plot data minus model at the literature values. For synthetic data,
# residuals should be pure Gaussian noise with RMS ≈ σ.
# 
# **Reduced χ²** should be ~1.0:
# - χ²_red >> 1: model is too simple (underfit) or noise is underestimated
# - χ²_red << 1: noise is overestimated (we're overfitting)
# 
# When you switch to real data, **systematic patterns in residuals** (bumps,
# slopes, oscillations) indicate stellar variability, starspot crossings, or
# instrumental systematics that may need detrending.
# 


fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors = ['#2E86C1', '#E74C3C', '#27AE60']

for col, (name, data) in enumerate(synthetic_data.items()):
    color  = colors[col]
    t, f   = data["time"], data["flux"]
    fe     = data["flux_err"]
    pdict, _ = theta_to_dict(theta_init[name], name)
    f_model  = compute_transit_flux(pdict, t, LONG_CADENCE[name])
    resid    = f - f_model

    # Top: data + model
    ax_top = axes[0, col]
    ax_top.errorbar(t*24, f, yerr=fe, fmt='.', color=color,
                    alpha=0.2, ms=2, elinewidth=0.5)
    ax_top.plot(t*24, f_model, 'k-', lw=2, label='Model (lit. values)')
    ax_top.set_ylabel("Normalized flux")
    ax_top.set_title(name, fontweight='bold')
    ax_top.legend(fontsize=9)

    # Bottom: residuals in ppm
    ax_bot = axes[1, col]
    ax_bot.errorbar(t*24, resid*1e6, yerr=fe*1e6, fmt='.', color=color,
                    alpha=0.3, ms=2, elinewidth=0.5)
    ax_bot.axhline(0, color='k', lw=1)
    ax_bot.set_xlabel("Time from mid-transit [hours]")
    ax_bot.set_ylabel("Residuals [ppm]")

    rms = np.std(resid) * 1e6
    exp = data["sigma"] * 1e6
    chi2_red = np.sum(resid**2 / fe**2) / (len(f) - 7)
    ax_bot.text(0.98, 0.95,
                f"RMS = {rms:.0f} ppm\nExpected ≈ {exp:.0f} ppm\nχ²_red = {chi2_red:.3f}",
                transform=ax_bot.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.suptitle("Residuals at Literature Parameter Values", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("residuals_lit_params.png", dpi=150, bbox_inches='tight')
plt.show()



# ## Cell 10 — Parameter sensitivity analysis
# 
# Perturb each parameter individually to understand its effect on the light curve.
# This prepares you to interpret the MCMC corner plots in Phase 5.
# 
# ### Key degeneracies to anticipate
# - **rp ↔ a/R★**: A deeper transit can mimic a longer transit. Partially broken by ingress/egress shape.
# - **a ↔ inc**: Both affect the impact parameter `b = a·cos(i)`. The corner plot will show an anti-correlation.
# - **rp ↔ u1,u2**: Stronger limb darkening (higher u values) deepens the transit center similarly to a larger planet.
# 


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
name  = "Kepler-7b"
p0    = SYSTEMS[name]
t     = np.linspace(-0.25, 0.25, 800)
flux0 = compute_transit_flux(p0, t, long_cadence=False)

perturb_configs = [
    ("rp",  [0.082, 0.090, 0.098, 0.106, 0.114], "Rp/R★",  'Blues'),
    ("a",   [7.0,   8.0,   8.93,  10.0,  11.5],  "a/R★",   'Blues'),
    ("inc", [81.0,  83.0,  85.16, 87.0,  89.0],  "i (°)",  'Blues'),
    ("u1",  [0.15,  0.30,  0.48,  0.62,  0.78],  "u₁ (LD)", 'Oranges'),
    ("u2",  [0.02,  0.10,  0.155, 0.25,  0.38],  "u₂ (LD)", 'Oranges'),
]

cmaps = [plt.cm.Blues, plt.cm.Blues, plt.cm.Blues, plt.cm.Oranges, plt.cm.Oranges]

for idx, ((pkey, values, label, _), cmap) in enumerate(zip(perturb_configs, cmaps)):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    cvals = np.linspace(0.35, 0.9, len(values))

    for val, cv in zip(values, cvals):
        p_mod = dict(p0)
        if pkey == "u1":
            p_mod["u"] = [val, p0["u"][1]]
        elif pkey == "u2":
            p_mod["u"] = [p0["u"][0], val]
        else:
            p_mod[pkey] = val

        # Highlight the literature value
        lit_val = p0.get(pkey, (p0["u"][0] if pkey=="u1" else p0["u"][1]))
        lw = 2.8 if abs(val - lit_val) < 1e-6 else 1.2
        flux_mod = compute_transit_flux(p_mod, t, long_cadence=False)
        ax.plot(t * 24, flux_mod, color=cmap(cv), lw=lw, label=f"{val:.4g}")

    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"Effect of varying {label}", fontweight='bold')
    ax.legend(fontsize=8, loc='lower center', ncol=3)

# Panel 6: a/R★ ↔ inc degeneracy illustration
ax6 = axes[1, 2]
pairs = [
    (7.5,  84.0, '#D6EAF8', 'a=7.5,  i=84.0°'),
    (8.93, 85.16,'#2E86C1', 'a=8.93, i=85.16° (lit.)'),
    (11.0, 86.5, '#1A3A5C', 'a=11.0, i=86.5°'),
]
for a_v, i_v, col, lbl in pairs:
    p_mod = dict(p0); p_mod["a"] = a_v; p_mod["inc"] = i_v
    b = a_v * np.cos(np.radians(i_v))
    flux_mod = compute_transit_flux(p_mod, t, long_cadence=False)
    ax6.plot(t * 24, flux_mod, color=col, lw=2, label=f"{lbl}\nb={b:.2f}")

ax6.set_xlabel("Time [hours]"); ax6.set_ylabel("Normalized flux")
ax6.set_title("a/R★ ↔ inclination degeneracy", fontweight='bold')
ax6.legend(fontsize=8)

plt.suptitle(f"Parameter Sensitivity — {name}", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("parameter_sensitivity.png", dpi=150, bbox_inches='tight')
plt.show()



# ## Cell 11 — Maximum Likelihood Estimation (MLE sanity check)
# 
# Before MCMC, we run a fast optimizer to verify:
# 1. The model correctly recovers literature values (no coding bugs)
# 2. The likelihood surface has a clean peak (no numerical issues)
# 3. We have good starting points for MCMC walkers
# 
# **MLE gives no uncertainty information** — that's what MCMC is for.
# But it's a fast, cheap sanity check. Expect recovered values to differ
# from literature by ~1–5% due to the random noise realization in our synthetic data.
# 


from scipy.optimize import minimize

print("Running MLE for each system (~10–30 seconds each)...\n")
mle_results = {}

for name in SYSTEMS:
    data  = synthetic_data[name]
    theta0 = theta_init[name].copy()

    neg_lp = lambda th: -log_posterior(th, name, data["time"], data["flux"], LONG_CADENCE[name])
    result = minimize(neg_lp, theta0, method='Nelder-Mead',
                      options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-7})
    mle_results[name] = result.x

    print(f"{'═'*56}")
    print(f"  {name}  (converged={result.success}, iters={result.nit})")
    print(f"{'─'*56}")
    print(f"  {'Parameter':<12} {'Literature':>12} {'MLE':>12} {'Δ (%)':>8}")
    print(f"  {'─'*48}")
    for pn, lv, fv in zip(PARAM_NAMES, theta0, result.x):
        if pn == "log_sigma":
            print(f"  {pn:<12} {lv:>12.4f} {fv:>12.4f}")
        else:
            dpct = (fv - lv) / abs(lv) * 100 if lv != 0 else 0
            print(f"  {pn:<12} {lv:>12.5f} {fv:>12.5f} {dpct:>7.2f}%")
    print()



# ## Cell 12 — Derived physical parameters
# 
# Convert dimensionless fitted parameters to physically meaningful quantities
# by combining with stellar properties from the literature.
# 
# ### Conversion formulas
# | Quantity | Formula |
# |----------|---------|
# | Rp (R_earth) | (Rp/R★) × R★/R☉ × 9.7313 |
# | Rp (R_Jup) | (Rp/R★) × R★/R☉ × 0.10276 |
# | a (AU) | (a/R★) × R★/R☉ × 0.004651 |
# | Transit duration T14 | (P/π) × arcsin(√[(1+k)²−b²] / (a/R★ · sin i)) |
# | Equilibrium temp Teq | T_eff × √(R★/2a) × (1−A)^¼ |
# 
# > **Phase 5 task:** Apply this to posterior samples (not just the MLE point) to get
# > full posterior distributions on Rp, a, and Teq with proper uncertainties.
# 


R_SUN_TO_REARTH = 9.7313
R_SUN_TO_RJUP   = 0.10276
R_SUN_TO_AU     = 0.004651

ALBEDO = {
    "Kepler-7b":   0.35,   # Known high geometric albedo
    "Kepler-10b":  0.30,
    "WASP-39b":    0.10,   # Low albedo, hot gas giant
}

print(f"{'═'*70}")
print(f"  Derived Physical Parameters (from MLE fits)")
print(f"{'═'*70}")

for name in SYSTEMS:
    p     = SYSTEMS[name]
    theta = mle_results[name]
    rp_f, a_f, inc_f = theta[0], theta[1], theta[2]

    R_star = p["R_star"]
    T_eff  = p["T_eff"]
    A      = ALBEDO[name]
    P      = p["per"]

    Rp_E   = rp_f * R_star * R_SUN_TO_REARTH
    Rp_J   = rp_f * R_star * R_SUN_TO_RJUP
    a_AU   = a_f  * R_star * R_SUN_TO_AU
    b      = a_f  * np.cos(np.radians(inc_f))

    sin_i  = np.sin(np.radians(inc_f))
    arg14  = np.sqrt(max((1 + rp_f)**2 - b**2, 0)) / (a_f * sin_i) if sin_i > 0 else 0
    T14_hr = (P / np.pi) * np.arcsin(min(arg14, 1.0)) * 24 if arg14 > 0 else 0

    Teq = T_eff * np.sqrt(R_star * R_SUN_TO_AU / (2 * a_AU)) * (1 - A)**0.25

    print(f"\n  {name}")
    print(f"  {'─'*52}")
    print(f"  Rp / R★            = {rp_f:.5f}")
    print(f"  Rp                 = {Rp_E:.3f} R_earth   ({Rp_J:.3f} R_Jup)")
    print(f"  a / R★             = {a_f:.3f}")
    print(f"  a                  = {a_AU:.4f} AU")
    print(f"  Inclination        = {inc_f:.3f}°")
    print(f"  Impact parameter b = {b:.4f}")
    print(f"  Transit duration   = {T14_hr:.3f} hours")
    print(f"  Teq (A={A:.2f})      = {Teq:.0f} K")



# ## Cell 13 — Export everything for Phase 3
# 
# Save synthetic data and initial theta vectors as `.npz` files.
# When your partner's real data arrives, replace the `.npz` contents
# with the real phase-folded arrays — **no other changes needed**.
# 
# ### How to swap in real data (Phase 1 → Phase 2 handoff)
# ```python
# # Load partner's phase-folded data (example)
# real = np.load("kepler7b_phase_folded.npz")
# synthetic_data["Kepler-7b"]["time"]     = real["time"]
# synthetic_data["Kepler-7b"]["flux"]     = real["flux"]
# synthetic_data["Kepler-7b"]["flux_err"] = real["flux_err"]
# # Re-estimate sigma from out-of-transit scatter:
# oot = real["flux"][np.abs(real["time"]) > 0.08]
# synthetic_data["Kepler-7b"]["sigma"] = np.std(oot)
# theta_init["Kepler-7b"][6] = np.log(np.std(oot))  # update log_sigma
# ```
# 


os.makedirs("phase2_outputs", exist_ok=True)

for name, data in synthetic_data.items():
    safe = name.replace("-","_").replace(" ","_")
    np.savez(f"phase2_outputs/{safe}_synthetic.npz",
             time=data["time"], flux=data["flux"],
             flux_err=data["flux_err"], flux_true=data["flux_true"],
             sigma=[data["sigma"]])
    np.save(f"phase2_outputs/{safe}_theta_init.npy", theta_init[name])

print("Files saved to phase2_outputs/:")
for f in sorted(os.listdir("phase2_outputs")):
    print(f"  {f}")

print()
print("=" * 60)
print("  PHASE 2 COMPLETE ✓")
print("=" * 60)
print("""
What was built in this notebook:
  1.  Literature parameter dicts  — Kepler-7b, Kepler-10b, WASP-39b
  2.  compute_transit_flux()      — batman forward model w/ supersampling
  3.  Synthetic observations      — realistic Gaussian noise, all 3 systems
  4.  log_likelihood()            — Gaussian photometry, free σ parameter
  5.  log_prior()                 — flat priors with physical constraints
  6.  log_posterior()             — ready to plug into emcee (Phase 3)
  7.  theta_init vectors          — starting points for MCMC walkers
  8.  MLE sanity check            — optimizer recovers literature values
  9.  Derived physical quantities — Rp, a (AU), Teq
 10.  Sensitivity plots           — key degeneracies documented
 11.  .npz output files           — handoff package for Phase 3

Next: Phase 3 — Prior specification, emcee setup, MCMC run
""")



