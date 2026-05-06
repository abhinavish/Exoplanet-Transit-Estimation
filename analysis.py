
# # Phase 5 — Posterior Analysis
# **Goal:** Extract median values, 68% credible intervals, calculate derived physical parameters from the posteriors, and compare them.


# Constants for physical derivation
R_SUN_TO_REARTH = 9.7313
R_SUN_TO_RJUP = 0.10276
R_SUN_TO_AU = 0.004651
ALBEDO = {"Kepler-7b": 0.35, "Kepler-10b": 0.30, "WASP-39b": 0.10}


# ## Cell 23 — Deriving Physical Parameters from Posteriors
# We apply our physical equations to *every* sample in the posterior chain to naturally propagate the uncertainties.


derived_results = {}

for name in SYSTEMS:
    samples = flat_samples_converged[name]
    R_star = SYSTEMS[name]["R_star"]
    T_eff = SYSTEMS[name]["T_eff"]
    A = ALBEDO[name]
    
    # Extract rp/R* and a/R* columns
    rp_samples = samples[:, 0]
    a_samples = samples[:, 1]
    
    # Derive posteriors for physical parameters
    Rp_earth_samples = rp_samples * R_star * R_SUN_TO_REARTH
    Rp_jup_samples = rp_samples * R_star * R_SUN_TO_RJUP
    a_au_samples = a_samples * R_star * R_SUN_TO_AU
    Teq_samples = T_eff * np.sqrt((R_star * R_SUN_TO_AU) / (2 * a_au_samples)) * (1 - A)**0.25
    
    # Store median and 68% CIs
    res = {}
    for param, arr in zip(
        ['Rp/R*', 'a/R*', 'inc', 'Rp (R_earth)', 'Rp (R_Jup)', 'a (AU)', 'T_eq (K)'],
        [rp_samples, a_samples, samples[:, 2], Rp_earth_samples, Rp_jup_samples, a_au_samples, Teq_samples]
    ):
        q16, q50, q84 = np.percentile(arr, [16, 50, 84])
        res[param] = f"{q50:.4f} (+{q84-q50:.4f} / -{q50-q16:.4f})"
        
    derived_results[name] = res

# Compile into a clean pandas DataFrame for the final write-up
df_results = pd.DataFrame(derived_results).T
display(df_results)

# # Phase 6 — Comparison & Write-up
# **Goal:** Generate graduate-level visual cross-system comparisons analyzing degeneracies and how data quality influences parameter certainty.


# ## Cell 24 — $R_p/R_*$ vs Inclination Degeneracy
# Grazing transits (like Kepler-10b) suffer from a degeneracy where a larger, grazing planet creates the same transit depth as a smaller, central planet.


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, name in zip(axes, SYSTEMS):
    samples = flat_samples_converged[name]
    rp_samp = samples[:, 0]
    inc_samp = samples[:, 2]
    
    # Use 2D histograms to show density
    h = ax.hist2d(inc_samp, rp_samp, bins=40, cmap='Blues', density=True)
    
    # Overlay the literature truth value
    true_inc = theta_init[name][2]
    true_rp = theta_init[name][0]
    ax.plot(true_inc, true_rp, 'ro', markersize=6, label="Literature Truth")
    
    ax.set_title(f"{name} Degeneracy")
    ax.set_xlabel("Inclination (°)")
    ax.set_ylabel("$R_p/R_*$")
    ax.legend(loc='upper right')

plt.suptitle("$R_p/R_*$ vs Inclination Degeneracy Across Systems", fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("degeneracy_comparison.png", dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 25 — Posterior Width (Uncertainty) Comparison
# How does transit depth and data quality (TESS vs Kepler) propagate into parameter uncertainty? We compare the standard deviations of our posteriors.


# We calculate the coefficient of variation (sigma/mean) as a percentage to normalize across different scales
uncertainty_data = []

for name in SYSTEMS:
    samples = flat_samples_converged[name]
    
    # Extract percentage uncertainty for core parameters
    rp_unc = (np.std(samples[:, 0]) / np.mean(samples[:, 0])) * 100
    a_unc = (np.std(samples[:, 1]) / np.mean(samples[:, 1])) * 100
    inc_unc = (np.std(samples[:, 2]) / np.mean(samples[:, 2])) * 100
    
    uncertainty_data.append({
        "System": name,
        "Depth (ppt)": SYSTEMS[name]["rp"]**2 * 1000,
        "Rp/R* Unc (%)": rp_unc,
        "a/R* Unc (%)": a_unc,
        "Inc Unc (%)": inc_unc
    })

df_unc = pd.DataFrame(uncertainty_data)

# Visualize the uncertainty
fig, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(df_unc))
width = 0.25

ax1.bar(x - width, df_unc["Rp/R* Unc (%)"], width, label='$R_p/R_*$ Uncertainty', color='#2E86C1')
ax1.bar(x, df_unc["a/R* Unc (%)"], width, label='$a/R_*$ Uncertainty', color='#27AE60')
ax1.bar(x + width, df_unc["Inc Unc (%)"], width, label='Inclination Uncertainty', color='#E74C3C')

ax1.set_ylabel("Relative Uncertainty (%)")
ax1.set_title("Parameter Uncertainty Propagated by Transit Depth/SNR", fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df_unc["System"])
ax1.legend()

# Add a secondary axis to show the transit depth for context
ax2 = ax1.twinx()
ax2.plot(x, df_unc["Depth (ppt)"], 'k--o', linewidth=2, markersize=8, label='Transit Depth (ppt)')
ax2.set_ylabel("Transit Depth (Parts per Thousand)")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("uncertainty_comparison.png", dpi=150)
plt.show()

print("\nPhases 4, 5, and 6 Complete ✓")
print("Your cross-system analysis visuals and final dataframes are ready for the journal paper write-up.")