
# # Phase 3 — MCMC Setup & Execution
# ### MCMC Transit Parameter Estimation
# 
# **Goal of this notebook:**
# Initialize the affine-invariant ensemble sampler (`emcee`), distribute walkers around our Phase 2 initial parameters, run the MCMC chains, and visualize the multi-dimensional posterior distributions using `corner`.
# 
# > **Note:** This notebook assumes it is running in the same environment/kernel as Phase 2, with `log_posterior`, `theta_init`, `synthetic_data`, and `SYSTEMS` already loaded in memory. If running separately, ensure you load the `.npz` and `.npy` files exported at the end of Phase 2.

# ## Cell 14 — Install & import Phase 3 dependencies
# We need `emcee` for the sampling and `corner` for the posterior visualizations.

import emcee
import corner
import time
import numpy as np
import matplotlib.pyplot as plt

print("emcee version:", emcee.__version__)
print("corner version:", corner.__version__)
print("Phase 3 imports successful ✓")


# ## Cell 15 — Initialize MCMC Walkers
# 
# `emcee` requires an ensemble of "walkers." We distribute these walkers in a tight Gaussian ball around our maximum likelihood/literature estimates (`theta_init`). 
# 
# - **Dimensions (`ndim`)**: 7 free parameters 
# - **Walkers (`nwalkers`)**: 50 (must be at least 2 × ndim, 50 is a good balance of speed and exploration)

ndim = len(PARAM_NAMES)
nwalkers = 50

# Dictionary to hold the initial positions for all walkers for each system
pos_init = {}

# We perturb the initial values by a tiny fraction (1e-4) to create the Gaussian ball
perturbation_scale = 1e-4

print(f"{'═'*40}")
print("  Initializing Walkers")
print(f"{'═'*40}")

for name in SYSTEMS:
    theta0 = theta_init[name]
    
    # Initialize a Gaussian ball around theta0
    # Add random noise scaled by the parameter's magnitude (or a small absolute value if 0)
    pos = theta0 + perturbation_scale * np.random.randn(nwalkers, ndim) * np.where(theta0 == 0, 1, np.abs(theta0))
    
    # Check bounds: If any walker starts outside the prior, it will immediately return -inf and break the chain.
    # We verify that log_prior is finite for all starting positions.
    valid_starts = 0
    for i in range(nwalkers):
        while not np.isfinite(log_prior(pos[i], name)):
            # Re-draw if outside prior bounds
            pos[i] = theta0 + perturbation_scale * np.random.randn(ndim) * np.where(theta0 == 0, 1, np.abs(theta0))
        valid_starts += 1
        
    pos_init[name] = pos
    print(f"{name:<14} {valid_starts}/{nwalkers} walkers initialized in valid prior space.")

print("Initialization complete ✓")


# ## Cell 16 — Run the MCMC Sampler
# 
# We run `emcee` for 5,000 steps per system. 
# *Note: For a full production run or if chains haven't converged (checked in Phase 4), you may need to increase this to 10,000+ steps.*


nsteps = 5000
samplers = {}

print(f"{'═'*60}")
print(f"  Running MCMC ({nsteps} steps per system)")
print(f"{'═'*60}")

for name in SYSTEMS:
    data = synthetic_data[name]
    pos = pos_init[name]
    
    # Arguments required by our log_posterior function
    args = (name, data["time"], data["flux"], LONG_CADENCE[name])
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=args)
    
    print(f"Starting {name} sampling...")
    t0_mcmc = time.time()
    
    # Run the MCMC chain
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    t1_mcmc = time.time()
    print(f"{name} completed in {(t1_mcmc - t0_mcmc)/60:.2f} minutes.\n")
    
    samplers[name] = sampler

print("All MCMC sampling complete ✓")


# ## Cell 17 — Discard Burn-in and Flatten Chains
# 
# The initial steps of the walkers retain memory of their starting positions and haven't yet reached the true posterior distribution. We discard the first 20% (1,000 steps) as "burn-in" and flatten the chains array to prepare it for `corner`.


burnin = 1000
thin = 15  # Thinning the chain reduces autocorrelation (takes every 15th step)

flat_samples = {}

for name, sampler in samplers.items():
    # sampler.get_chain() returns shape: (nsteps, nwalkers, ndim)
    # discard burn-in, thin the chains, and flatten to shape: (N_samples, ndim)
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_samples[name] = samples
    
    print(f"{name:<14} Retained {samples.shape[0]} valid independent samples.")


# ## Cell 18 — Posterior Visualization (Corner Plots)
# 
# The corner plot shows 1D marginalized posterior distributions on the diagonal, and 2D joint posterior distributions (parameter covariances) on the off-diagonals. 
# 
# Pay close attention to the `rp` vs `a` and `inc` vs `a` panels to observe the degeneracies we predicted in Phase 2.


# Formatted labels for the corner plots (using LaTeX for clean rendering in matplotlib)
CORNER_LABELS = [
    r"$R_p/R_*$", 
    r"$a/R_*$", 
    r"$i$ (deg)", 
    r"$t_0$", 
    r"$u_1$", 
    r"$u_2$", 
    r"$\log(\sigma)$"
]

for name in SYSTEMS:
    samples = flat_samples[name]
    true_vals = theta_init[name] # The true literature values we injected
    
    fig = corner.corner(
        samples, 
        labels=CORNER_LABELS, 
        truths=true_vals,
        truth_color="#E74C3C",
        quantiles=[0.16, 0.5, 0.84], # 1-sigma credible intervals
        show_titles=True, 
        title_kwargs={"fontsize": 11},
        label_kwargs={"fontsize": 12},
        color="#2E86C1",
        hist_kwargs={"linewidth": 1.5},
        title_fmt=".4f"
    )
    
    fig.suptitle(f"Posterior Distributions: {name}", fontsize=16, y=1.02, fontweight="bold")
    plt.savefig(f"corner_{name.replace('-', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()


# ## Cell 19 — Median Parameters and 1σ Uncertainties
# 
# Extract the 16th, 50th (median), and 84th percentiles to report our final fitted parameter values with their asymmetric uncertainties.

print(f"{'═'*80}")
print("  Final Recovered Parameters (Median ± 1σ)")
print(f"{'═'*80}")

for name in SYSTEMS:
    samples = flat_samples[name]
    print(f"\n{name}")
    print("-" * 50)
    
    for i, label in enumerate(PARAM_NAMES):
        mcmc_percentiles = np.percentile(samples[:, i], [16, 50, 84])
        q_m = mcmc_percentiles[1]
        q_minus = q_m - mcmc_percentiles[0]
        q_plus = mcmc_percentiles[2] - q_m
        
        # Compare to true injected value
        true_val = theta_init[name][i]
        
        if label == "log_sigma":
            # Convert back to standard ppm for readability
            med_sigma_ppm = np.exp(q_m) * 1e6
            true_sigma_ppm = np.exp(true_val) * 1e6
            print(f"  {label:<10} = {q_m:>8.4f}  (Noise: ~{med_sigma_ppm:.0f} ppm, True: {true_sigma_ppm:.0f} ppm)")
        else:
            print(f"  {label:<10} = {q_m:>8.5f} + {q_plus:.5f} / - {q_minus:.5f}  (True: {true_val:.5f})")

print("\nPhase 3 Complete ✓")