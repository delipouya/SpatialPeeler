import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squidpy as sq
import anndata as ad
from pathlib import Path

import NaiveDE
import SpatialDE as spatialde

root_path = os.path.abspath('./..')
sys.path.insert(0, root_path )
output_file = Path.home() / 'SpatialPeeler' / 'data_PSC' / 'PSC_varimax.h5ad'
adata_merged = ad.read_h5ad(output_file)

# Check if the varimax PCs are present
if 'X_vpca' not in adata_merged.obsm:
    raise ValueError("Varimax PCs not found in the merged adata object. Please run the varimax PCA calculation first.")
# Check the shape of the varimax PCs
print(f"Varimax PCs shape: {adata_merged.obsm['X_vpca'].shape}")
# Check the number of samples
sample_ids = adata_merged.obs['sample_id'].unique()
print(f"Number of samples: {len(sample_ids)}")

# --- Sample-wise split ---
sample_ids = adata_merged.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in sample_ids
}



import scipy# Monkey-patch NumPy functions into scipy for SpatialDE compatibility
scipy.argsort = np.argsort
scipy.zeros_like = np.zeros_like
scipy.ones = np.ones
scipy.sum = np.sum
scipy.mean = np.mean
scipy.median = np.median
scipy.isfinite = np.isfinite
scipy.sqrt = np.sqrt
scipy.amax = np.amax

results_dic = {}
for sid in sample_ids:
    ad = adata_by_sample[sid]
    
    # Create DataFrame for vPCA factors
    factor_names = [f'vpca_{i+1}' for i in range(30)]
    expr_df = pd.DataFrame(ad.obsm['X_vpca'], columns=factor_names, index=ad.obs_names)
    
    # Create DataFrame for spatial coordinates
    coords_df = pd.DataFrame(ad.obsm['spatial'], columns=['x', 'y'], index=ad.obs_names)
    
    # Combine into one DataFrame as required by SpatialDE
    combined_df = expr_df.copy()
    combined_df['x'] = coords_df['x']
    combined_df['y'] = coords_df['y']
    
    # Run SpatialDE    
    results = spatialde.run(combined_df[['x', 'y']].values, combined_df[factor_names])
    # Store results
    adata_by_sample[sid].uns['spatialde_results'] = results
    results_dic[sid] = results


 # Initialize a DataFrame to store q-values
qvals_df = pd.DataFrame(index=factor_names)

for sid in sample_ids:
    results = adata_by_sample[sid].uns['spatialde_results']
    qvals = results.set_index('g')['qval']
    qvals_df[sid] = qvals

# Separate normal and PSC samples
normal_samples = [sid for sid in sample_ids if 'normal' in sid]
psc_samples = [sid for sid in sample_ids if 'PSC' in sid]

# Calculate mean q-values for each group
qvals_df['normal_mean_qval'] = qvals_df[normal_samples].mean(axis=1)
qvals_df['psc_mean_qval'] = qvals_df[psc_samples].mean(axis=1)

# Calculate difference
qvals_df['qval_diff'] = qvals_df['psc_mean_qval'] - qvals_df['normal_mean_qval']

# Sort by difference
qvals_df_sorted = qvals_df.sort_values('qval_diff')



# Select top 4 factors with increased spatial variability in PSC
top_psc_factors = qvals_df_sorted.head(4).index.tolist()

# Select top 4 factors with increased spatial variability in normal
top_normal_factors = qvals_df_sorted.tail(4).index.tolist()

# Combine for visualization
top_factors = top_psc_factors + top_normal_factors

# Visualize
for factor in top_factors:
    idx = int(factor.split('_')[1]) - 1  # Extract index from factor name
    vmin = adata_merged.obsm['X_vpca'][:, idx].min()
    vmax = adata_merged.obsm['X_vpca'][:, idx].max()
    
    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axs = axs.flatten()
    
    for i, sid in enumerate(sample_ids):
        ad = adata_by_sample[sid]
        spatial = ad.obsm["spatial"]
        scores = ad.obsm["X_vpca"][:, idx]
        
        im = axs[i].scatter(
            spatial[:, 0], spatial[:, 1],
            c=scores, cmap="coolwarm", s=10, vmin=vmin, vmax=vmax
        )
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")
    
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(f"{factor} value", fontsize=12)
    
    plt.suptitle(f"{factor} across spatial coordinates", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()