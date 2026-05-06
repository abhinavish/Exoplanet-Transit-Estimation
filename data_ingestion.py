# Cell 1
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set plotting style for clear visualization
plt.style.use('default')

# Cell 2
# Literature values for the 3 target systems [cite: 8, 9]
# Note: For your final Phase 5 validation, you may want to cross-reference 
# these exact T0 and Period values with the NASA Exoplanet Archive[cite: 80].

target_params = {
    "Kepler-7": {
        "search_name": "Kepler-7",
        "mission": "Kepler",
        "period": 4.8854892, 
        "t0": 133.8166 # BKJD
    },
    "Kepler-10": {
        "search_name": "Kepler-10",
        "mission": "Kepler",
        "period": 0.837491, 
        "t0": 2454964.576 - 2454833.0 # Converted roughly to BKJD
    },
    "WASP-39": {
        "search_name": "WASP-39",
        "mission": "TESS",
        "period": 4.0552941, 
        "t0": 2458269.837 - 2457000.0 # Converted roughly to BTJD
    }
}

# Cell 3
def acquire_and_process_lightcurve(target_name, params):
    print(f"Querying MAST archive for {target_name}...")
    
    # 1. Query MAST archive [cite: 39]
    author = "Kepler" if params['mission'] == "Kepler" else "SPOC"
    search_result = lk.search_lightcurve(params['search_name'], author=author)
    
    # 2. Download all appropriate quarters/sectors to maximize SNR 
    print(f"Downloading {len(search_result)} quarters/sectors...")
    lc_collection = search_result.download_all()
    
    # Stitch the collection into a single continuous light curve
    lc = lc_collection.stitch()
    
    # 3 & 4. Clean and Normalize [cite: 41, 42]
    # PDCSAP flux is used by default. We remove NaNs, normalize to baseline, 
    # and sigma-clip outliers (using a conservative 5-sigma to avoid clipping deep transits)
    lc_clean = lc.remove_nans().normalize().remove_outliers(sigma=5)
    
    # 5. Phase-fold 
    print(f"Phase-folding on period {params['period']} days...")
    folded_lc = lc_clean.fold(period=params['period'], epoch_time=params['t0'])
    
    return folded_lc

# Cell 4
# Dictionary to store the final output [cite: 44]
processed_lightcurves = {}

# Set up matplotlib figure for validation
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, (target, params) in enumerate(target_params.items()):
    # Execute pipeline
    folded_lc = acquire_and_process_lightcurve(target, params)
    processed_lightcurves[target] = folded_lc
    
    # Plot results
    ax = axes[i]
    folded_lc.scatter(ax=ax, s=2, alpha=0.5, color='black', label=f'{target} Data')
    
    # Zoom in on the transit window (-0.1 to 0.1 phase) for visual inspection
    ax.set_xlim(-0.1, 0.1) 
    ax.set_title(f"{target} Phase-Folded Light Curve")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized Flux")
    ax.legend()

plt.tight_layout()
plt.show()

# Cell 5
# Export the phase-folded light curves to CSV for Phase 2 / Phase 3 usage
import os

output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

for target, folded_lc in processed_lightcurves.items():
    filename = f"{output_dir}/{target}_folded.csv"
    
    # Extract time (phase), flux, and flux error
    df = pd.DataFrame({
        'phase': folded_lc.time.value,
        'flux': folded_lc.flux.value,
        'flux_err': folded_lc.flux_err.value
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved {target} to {filename}")