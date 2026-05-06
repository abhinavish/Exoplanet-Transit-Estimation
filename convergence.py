
# # Phase 4 — Convergence Diagnostics
# ### MCMC Transit Parameter Estimation
# 
# **Goal:** Rigorously verify that our MCMC chains have converged to the true posterior distribution using trace plots, autocorrelation times ($\tau$), effective sample size ($N_{eff}$), and the Gelman-Rubin $\hat{R}$ statistic.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner

# Ensure consistent styling
plt.rcParams.update({'figure.dpi': 120, 'font.size': 12})


# ## Cell 20 — Trace Plots
# Visual inspection of the chains. We want to see a "hairy caterpillar" indicating well-mixed walkers traversing the parameter space without getting stuck.

fig, axes = plt.subplots(ndim, 1, figsize=(10, 12), sharex=True)

# We'll use Kepler-7b as our primary visual example for trace plots
sample_system = "Kepler-7b"
chain = samplers[sample_system].get_chain()

for i in range(ndim):
    ax = axes[i]
    # Plot the paths of all 50 walkers
    ax.plot(chain[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(chain))
    ax.set_ylabel(PARAM_NAMES[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")
fig.suptitle(f"Walker Trace Plots: {sample_system}", fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# ## Cell 21 — Autocorrelation Time & Effective Sample Size
# The integrated autocorrelation time ($\tau$) estimates the number of steps needed to draw independent samples. Following the blueprint, we'll discard $3 \times \tau$ as burn-in and compute $N_{eff}$.


burnin_dict = {}
flat_samples_converged = {}

print(f"{'═'*70}")
print(f"  Autocorrelation & Effective Sample Size")
print(f"{'═'*70}")

for name, sampler in samplers.items():
    try:
        # tol=0 suppresses the emcee warning if chains are too short for a perfectly reliable tau
        tau = sampler.get_autocorr_time(tol=0)
        mean_tau = np.mean(tau)
        burnin = int(3 * mean_tau)
        
        # Calculate Effective Sample Size (N_eff = N_steps * N_walkers / tau)
        n_samples_total = sampler.iteration * nwalkers
        n_eff = int(n_samples_total / mean_tau)
        
        print(f"\n{name}:")
        print(f"  Mean Autocorrelation Time (τ) : {mean_tau:.1f} steps")
        print(f"  Recommended Burn-in (3×τ)   : {burnin} steps")
        print(f"  Effective Sample Size (N_eff) : {n_eff} independent samples")
        
        burnin_dict[name] = burnin
        
        # Update our flat samples using the calculated burn-in
        flat_samples_converged[name] = sampler.get_chain(discard=burnin, thin=15, flat=True)
        
    except emcee.autocorr.AutocorrError:
        print(f"{name}: Chains too short to reliably compute τ. Defaulting to 1000 step burn-in.")
        burnin_dict[name] = 1000
        flat_samples_converged[name] = sampler.get_chain(discard=1000, thin=15, flat=True)


# ## Cell 22 — Gelman-Rubin $\hat{R}$ Statistic
# Compares variance within individual chains to variance across all chains. Target: $\hat{R} < 1.01$.


def compute_gelman_rubin(chains):
    """Calculates Gelman-Rubin R-hat for an emcee chain of shape (nsteps, nwalkers, ndim)."""
    nsteps, nwalkers, ndim = chains.shape
    R_hat = np.zeros(ndim)
    
    for i in range(ndim):
        chain = chains[:, :, i]
        # Within-chain variance W
        W = np.mean(np.var(chain, axis=0, ddof=1))
        # Between-chain variance B
        chain_means = np.mean(chain, axis=0)
        mean_of_means = np.mean(chain_means)
        B = nsteps / (nwalkers - 1) * np.sum((chain_means - mean_of_means)**2)
        # Estimated marginal posterior variance V_hat
        V_hat = (nsteps - 1) / nsteps * W + B / nsteps
        R_hat[i] = np.sqrt(V_hat / W)
    return R_hat

print(f"{'═'*50}")
print(f"  Gelman-Rubin Convergence (Target < 1.01)")
print(f"{'═'*50}")

for name, sampler in samplers.items():
    # Discard burn-in before calculating R-hat
    chains_post_burnin = sampler.get_chain(discard=burnin_dict[name])
    r_hat_vals = compute_gelman_rubin(chains_post_burnin)
    
    print(f"\n{name}:")
    for param, r in zip(PARAM_NAMES, r_hat_vals):
        status = "✓" if r < 1.01 else "!"
        print(f"  {param:<10} R_hat = {r:.5f}  {status}")