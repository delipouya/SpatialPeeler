
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

### compute Moran’s I for each of the first 30 varimax-rotated PCs using scanp

# 1. Store each rotated factor as an observation-level column (adata.obs)
for i in range(30):
    adata_merged.obs[f'vpca_{i+1}'] = adata_merged.obsm['X_vpca'][:, i]

# 2. Build spatial graph (if not already done)
sq.gr.spatial_neighbors(adata_merged, coord_type="generic", n_neighs=20)

# 2. Compute Moran's I for the rotated PC scores
sq.gr.spatial_autocorr(
    adata_merged,
    mode="moran",
    genes=[f'vpca_{i+1}' for i in range(30)],
    attr="obs"  # <- Tell Squidpy to use .obs instead of .X
)

# 3. Extract and sort the results
moran_df = adata_merged.uns['moranI'].copy()
moran_df_sorted = moran_df.sort_values('I', ascending=False)

# 4. Show top spatially patterned factors
print(moran_df_sorted.head(10))


moran_df = pd.DataFrame({
    'Factor': moran_df_sorted.index,
    'MoranI': moran_df_sorted['I'],
    'pval': moran_df_sorted['pval_norm'],
}).sort_values(by='MoranI', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Factor', y='MoranI', data=moran_df.head(15), palette="viridis")
plt.title("Top 15 Most spatially-variable vPCs")
plt.ylabel("MoranI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### define top10 indices based on moran_df_sorted indices
top10_spatial_vPCs = moran_df_sorted.head(10).index.tolist()
top10_spatial_indices = [int(f[5:]) - 1 for f in top10_spatial_vPCs] 

# Visualize each of these across spatial coordinates per sample
for idx in top10_spatial_indices:
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
