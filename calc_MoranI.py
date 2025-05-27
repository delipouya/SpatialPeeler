
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

### compute Moran’s I for each of the first 30 varimax-rotated PCs using scanp for each samples seperately
### save the results in adata.uns['moranI'] 


moran_results = {}
for sid in sample_ids:
    ad = adata_by_sample[sid]
    
    # Ensure spatial coordinates are present
    if 'spatial' not in ad.obsm:
        raise ValueError(f"Spatial coordinates not found in the adata object for sample {sid}.")
    
    # Store each rotated factor as an observation-level column (adata.obs)
    for i in range(30):
        ad.obs[f'vpca_{i+1}'] = ad.obsm['X_vpca'][:, i]

    # Build spatial graph (if not already done)
    sq.gr.spatial_neighbors(ad, coord_type="generic", n_neighs=20)

    # Compute Moran's I for the rotated PC scores
    sq.gr.spatial_autocorr(
        ad,
        mode="moran",
        genes=[f'vpca_{i+1}' for i in range(30)],
        attr="obs"  # <- Tell Squidpy to use .obs instead of .X
    )
    # Extract and store the results in adata.uns
    moran_df = ad.uns['moranI'].copy()
    moran_df_sorted = moran_df.sort_values('I', ascending=False)
    moran_results[sid] = moran_df_sorted
    adata_merged.uns[f'moranI_{sid}'] = moran_df


#### identify the factors that have varuable moran's I across normal and PSC samples
### print top 10 factors for each sample
for sid, moran_df in moran_results.items():
    print(f"Top 10 Moran's I factors for sample {sid}:")
    print(moran_df.head(10).index.tolist())
    print("\n")

# for each factor, calculate the mean Moran's I across normal(normal_A tp D) and PSC (PSC_A to D)samples 
normal_samples = [sid for sid in sample_ids if 'normal' in sid]
psc_samples = [sid for sid in sample_ids if 'PSC' in sid]
moran_means = {}
for factor in moran_results[sample_ids[0]].index:
    normal_values = [moran_results[sid].loc[factor, 'I'] for sid in normal_samples if factor in moran_results[sid].index]
    psc_values = [moran_results[sid].loc[factor, 'I'] for sid in psc_samples if factor in moran_results[sid].index]
    
    moran_means[factor] = {
        'normal_mean': np.mean(normal_values) if normal_values else np.nan,
        'psc_mean': np.mean(psc_values) if psc_values else np.nan
    }
moran_means_df = pd.DataFrame(moran_means).T
moran_means_df['diff'] = moran_means_df['psc_mean'] - moran_means_df['normal_mean']
moran_means_df_sorted = moran_means_df.sort_values(by='diff', ascending=False)
print("Moran's I means for factors across normal and PSC samples:")
print(moran_means_df_sorted.head(10))


plt.figure(figsize=(12, 18))
sns.heatmap(moran_means_df_sorted[['normal_mean', 'psc_mean', 'diff']], annot=True, cmap='coolwarm', center=0)
plt.title("Moran's I Means for Factors Across Normal and PSC Samples")
plt.xlabel("Factors")
plt.ylabel("Sample Type")
plt.xticks(rotation=45)
plt.tight_layout()

## select the top 8 factors based on Moran's I difference (4 postive and 4 negative differences)
top_factors = moran_means_df_sorted.head(4).index.tolist()
bottom_factors = moran_means_df_sorted.tail(4).index.tolist()
top_bottom_factors = top_factors + bottom_factors

### define top10 indices based on moran_df_sorted indices
top_spatial_indices = [int(f[5:]) - 1 for f in top_bottom_factors] 

# Visualize each of these across spatial coordinates per sample
for idx in top_spatial_indices:
    factor_name = f"vPCA{idx + 1}"
    all_vals = adata_merged.obsm["X_vpca"][:, idx]
    vmin, vmax = np.quantile(all_vals, [0.01, 0.99])

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
            c=np.clip(scores, vmin, vmax),
            cmap="coolwarm", s=10, vmin=vmin, vmax=vmax
        )
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(f"{factor_name} value", fontsize=12)

    plt.suptitle(f"{factor_name} across spatial coordinates", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()



'''




'''