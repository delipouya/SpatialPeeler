import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path )
import functools
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import hiddensc
from hiddensc import utils, files, vis

import scanpy as sc
import scvi
import anndata
from sklearn.decomposition import NMF
import functools

import numpy as np
import numpy as np
from scipy.linalg import svd
from numpy.linalg import LinAlgError


#from scipy.stats import mannwhitneyu
RAND_SEED = 28
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

from hiddensc import models


# Files to load
file_names = [
    "normal_A", "normal_B", "normal_C", "normal_D",
    "PSC_A", "PSC_B", "PSC_C", "PSC_D"
]


adata_dict = {}
cell_type_sets = {}
# Load and extract unique cell types
for fname in file_names:
    
    PREFIX = fname.split('.')[0]  # Use the first file name as prefix
    at_data_dir = functools.partial(os.path.join, root_path, 'data_PSC')
    adata = sc.read(at_data_dir(f'{PREFIX}.h5ad'))
    adata_dict[fname] = adata

    adata.obs['binary_label'] = adata.obs['disease']!='normal' #'primary sclerosing cholangitis'

    hiddensc.datasets.preprocess_data(adata)
    hiddensc.datasets.normalize_and_log(adata)
    cell_types = adata.obs['cell_type'].unique()
    cell_type_sets[fname] = set(cell_types)


# Add batch info before merging
for fname, ad in adata_dict.items():
    ad.obs["batch"] = fname  # This will let HiDDEN know the origin

# Concatenate all datasets
adata_merged = anndata.concat(adata_dict.values(), 
                              label="sample_id", keys=adata_dict.keys(), merge="same")
adata_merged.obs['batch'] = adata_merged.obs['batch'].astype(str)
adata_merged.obs['sample_id'] = adata_merged.obs['sample_id'].astype(str)

### Apply NMF
n_components = 10
# ------------------------------------
# Step 1: Normalize and log-transform - I DIDN'T USE THIS
# ------------------------------------
sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)
adata_merged.layers["lognorm"] = adata_merged.X.copy()

# ------------------------------------
# Step 2: Apply NMF
# ------------------------------------
n_factors = 10  # or choose based on elbow plot, coherence, etc.
nmf_model = NMF(n_components=n_factors, init='nndsvda', random_state=RAND_SEED, max_iter=1000)

# Apply NMF to log-normalized expression matrix
# X must be dense; convert if sparse
import scipy.sparse as sp
X = adata_merged.X
if sp.issparse(X): 
    X = X.toarray()

W = nmf_model.fit_transform(X)  # cell × factor matrix
H = nmf_model.components_        # factor × gene matrix

# ------------------------------------
# Step 3: Save results in AnnData
# ------------------------------------
adata_merged.obsm["X_nmf"] = W
adata_merged.uns["nmf_components"] = H

def plot_spatial_nmf(adata, factor_idx, sample_id=None):
    spatial = adata.obsm["spatial"]
    values = adata.obsm["X_nmf"][:, factor_idx]

    # Clip extremes for visualization
    vmax = np.quantile(values, 0.99)
    vmin = np.quantile(values, 0.01)
    values = np.clip(values, vmin, vmax)

    plt.figure(figsize=(5, 5))
    sc_plot = plt.scatter(spatial[:, 0], spatial[:, 1], c=values, cmap="viridis", s=10)
    plt.axis("equal")
    title = f"NMF Factor {factor_idx + 1}"
    if sample_id:
        title += f" – {sample_id}"
    plt.title(title)
    plt.colorbar(sc_plot, label=f"NMF{factor_idx + 1}")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()

sample_ids = adata_merged.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in adata_merged.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())
# Plot for each sample and NMF factor 1–4
for sid in sample_ids:
    print(f"Plotting NMF for sample {sid}")
    for k in range(4):
        plot_spatial_nmf(adata_by_sample[sid], k, sample_id=sid)

adata = adata_merged
nmf_factors = adata.obsm['X_nmf'][:, :4]
nmf_df = pd.DataFrame(nmf_factors, columns=[f'NMF{i+1}' for i in range(4)])
nmf_df['sample_id'] = adata.obs['sample_id'].values

nmf_long = nmf_df.melt(id_vars='sample_id', var_name='Factor', value_name='Score')

plt.figure(figsize=(12, 6))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=nmf_long, inner="box", palette="Set2")
plt.title("Distribution of NMF1–NMF4 Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ------------------------------------
# Step 4: Logistic regression to find informative factors
# ------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

X = adata.obsm['X_nmf']  # shape: cells × NMFs
y_raw = adata.obs['disease'].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model.fit(X, y)

coefs = model.coef_[0]
nmf_labels = [f'NMF{i+1}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'Factor': nmf_labels,
    'Coefficient': coefs,
    'AbsCoefficient': np.abs(coefs)
})
importance_df_sorted = importance_df.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Factor', y='AbsCoefficient', data=importance_df_sorted.head(15), palette="viridis")
plt.title("Top 15 Most Informative NMF Factors for Predicting Disease")
plt.ylabel("Absolute Logistic Regression Coefficient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ------------------------------------
# Step 5: Top 5 NMF factors
# ------------------------------------

top5_nmf = importance_df_sorted['Factor'].head(8).tolist()
print("Top 5 NMF factors:", top5_nmf)

top5_indices = [int(f[3:]) - 1 for f in top5_nmf]  # e.g. 'NMF2' → 1
top5_df = pd.DataFrame(adata.obsm["X_nmf"][:, top5_indices], columns=top5_nmf)
top5_df["sample_id"] = adata.obs["sample_id"].values

top5_long = top5_df.melt(id_vars="sample_id", var_name="Factor", value_name="Score")

plt.figure(figsize=(16, 8))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=top5_long, inner="box", palette="Set2")
plt.title("Distribution of Top 5 Informative NMF Factors Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# ------------------------------------
# Step 6: Visualize NMF factors across samples
# ------------------------------------
for factor_idx in range(10):  # or up to 12 if you'd like
    factor_name = f"NMF{factor_idx + 1}"

    all_vals = adata.obsm["X_nmf"][:, factor_idx]
    vmin, vmax = np.quantile(all_vals, [0.01, 0.99])

    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axs = axs.flatten()

    for i, sid in enumerate(sample_ids):
        ad = adata_by_sample[sid]
        spatial = ad.obsm["spatial"]
        scores = ad.obsm["X_nmf"][:, factor_idx]
        im = axs[i].scatter(
            spatial[:, 0], spatial[:, 1],
            c=np.clip(scores, vmin, vmax),
            cmap="viridis", s=10, vmin=vmin, vmax=vmax
        )
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(f"{factor_name} value", fontsize=12)

    plt.suptitle(f"{factor_name} across spatial coordinates", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()



