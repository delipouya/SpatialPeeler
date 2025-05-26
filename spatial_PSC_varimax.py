import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path )
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import hiddensc
from hiddensc import utils, vis

import scanpy as sc
import scvi
import anndata
from scipy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import svd

####  python implementation of base R varimax function
def varimax(x, normalize=True, eps=1e-05):
    '''
    varimax rotation
    x: the factor loading matrix
    normalize: whether to normalize the factor loading matrix
    eps: the tolerance for convergence
    '''
    nc = x.shape[1]
    if nc < 2:
        return x
    
    if normalize:
        sc = np.sqrt(np.sum(x**2, axis=1))
        x = x / sc[:, np.newaxis]
    
    p = x.shape[0]
    TT = np.eye(nc)
    d = 0
    for i in range(1, 1001):
        z = np.dot(x, TT)
        B = np.dot(x.T, z**3 - np.dot(z, np.diag(np.sum(z**2, axis=0)) / p))
        u, sB, vh = svd(B, full_matrices=False)
        TT = np.dot(u, vh)
        dpast = d
        d = np.sum(sB)
        
        if d < dpast * (1 + eps):
            break
    
    z = np.dot(x, TT)
    
    if normalize:
        z = z * sc[:, np.newaxis]
    
    return {'rotloading': z, 'rotmat': TT}


def get_rotated_scores(factor_scores, rotmat):
    return np.dot(factor_scores, rotmat)


## method: scale(original pc scores) %*% rotmat
def get_rotated_scores(factor_scores, rotmat):
    '''
    calculate the rotated factor scores
    factor_scores: the factor scores matrix
    rotmat: the rotation matrix
    '''
    return np.dot(factor_scores, rotmat)

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


# --- Preprocessing ---
sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)
sc.pp.scale(adata_merged)
sc.tl.pca(adata_merged, n_comps=30)

# --- Apply varimax ---
pca_scores = adata_merged.obsm["X_pca"]
loadings = adata_merged.varm["PCs"][:, :30]
varimax_result = varimax(loadings)
rotmat = varimax_result['rotmat']
rotated_scores = get_rotated_scores(pca_scores[:, :30], rotmat)
adata_merged.obsm["X_vpca"] = rotated_scores

# --- Sample-wise split ---
sample_ids = adata_merged.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in sample_ids
}

# --- Visualization function ---
def plot_spatial_vpca(adata, factor_idx, sample_id=None):
    spatial = adata.obsm["spatial"]
    values = adata.obsm["X_vpca"][:, factor_idx]
    vmax, vmin = np.quantile(values, [0.99, 0.01])
    values = np.clip(values, vmin, vmax)
    plt.figure(figsize=(5, 5))
    plt.scatter(spatial[:, 0], spatial[:, 1], c=values, cmap="coolwarm", s=10)
    plt.axis("equal")
    title = f"vPCA Factor {factor_idx + 1}" + (f" – {sample_id}" if sample_id else "")
    plt.title(title)
    plt.colorbar(label=f"vPCA {factor_idx + 1}")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()

# --- Plot first 4 factors for each sample ---
for sid in sample_ids:
    print(f"Plotting varimax PCs for sample {sid}")
    for k in range(4):
        plot_spatial_vpca(adata_by_sample[sid], k, sample_id=sid)

# --- Violin plots of vPCA1–4 ---
vpca_factors = adata_merged.obsm['X_vpca'][:, :4]
vpca_df = pd.DataFrame(vpca_factors, columns=[f'vPCA{i+1}' for i in range(4)])
vpca_df['sample_id'] = adata_merged.obs['sample_id'].values
vpca_long = vpca_df.melt(id_vars='sample_id', var_name='Factor', value_name='Score')

plt.figure(figsize=(12, 6))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=vpca_long, inner="box", palette="Set2")
plt.title("Distribution of vPCA1–vPCA4 Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Logistic regression on varimax PCs ---
X = adata_merged.obsm['X_vpca']
y = LabelEncoder().fit_transform(adata_merged.obs['disease'].values)

model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model.fit(X, y)

coefs = model.coef_[0]
vpca_labels = [f'vPCA{i+1}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'Factor': vpca_labels,
    'Coefficient': coefs,
    'AbsCoefficient': np.abs(coefs)
}).sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Factor', y='AbsCoefficient', data=importance_df.head(15), palette="viridis")
plt.title("Top 15 Most Informative vPCs for Predicting Disease")
plt.ylabel("Absolute Logistic Regression Coefficient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Top 5 vPCA visualization ---
top5_vpca = importance_df['Factor'].head(5).tolist()
top5_indices = [int(f[4:]) - 1 for f in top5_vpca]
top5_df = pd.DataFrame(adata_merged.obsm["X_vpca"][:, top5_indices], columns=top5_vpca)
top5_df["sample_id"] = adata_merged.obs["sample_id"].values
top5_long = top5_df.melt(id_vars="sample_id", var_name="Factor", value_name="Score")

plt.figure(figsize=(16, 8))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=top5_long, inner="box", palette="Set2")
plt.title("Distribution of Top 5 Informative vPCA Factors Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



# Top 10 varimax PCs by absolute coefficient
top10_vpca = importance_df.sort_values("AbsCoefficient", ascending=False)['Factor'].head(10).tolist()
top10_indices = [int(f[4:]) - 1 for f in top10_vpca]  # e.g., 'vPCA3' → 2

# Visualize each of these across spatial coordinates per sample
for idx in top10_indices:
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




### save the merged adata object with varimax PCs
output_file = os.path.join(root_path, 'data_PSC', 'PSC_varimax.h5ad')
adata_merged.write(output_file)
# ------------------------------------

