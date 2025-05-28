import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
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

from hiddensc import models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit


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
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit


RAND_SEED = 28
CASE_COND = 1
def standalone_logistic(X, y):
    #clf = LogisticRegression(random_state=RAND_SEED, penalty='none').fit(X, y)
    clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X, y)
    predicted_label = clf.predict(X)
    predicted_prob = clf.predict_proba(X)
    return predicted_prob[:,1]


def single_factor_logistic_evaluation(adata, factor_key="X_nmf", max_factors=30):
    """
    Evaluate each individual factor using logistic regression to predict status.
    Performs KMeans on logit(p_hat) of case samples to identify subtypes.

    Parameters:
    - adata: AnnData object with .obsm[factor_key] storing factors
    - factor_key: str, key in .obsm where factor matrix is stored
    - max_factors: int, number of factors to evaluate (starting from index 0)

    Returns:
    - all_results: list of dicts with info for each factor
    """
    all_results = []
    X = adata.obsm[factor_key]
    y = adata.obs["status"].values
    sample_ids = adata.obs["sample_id"].values

    for i in range(min(max_factors, X.shape[1])):
        print(f"Evaluating factor {i+1}...")
        Xi = X[:, i].reshape(-1, 1)  # Single factor

        # Step 1: Fit logistic regression on single factor
        p_hat = standalone_logistic(Xi, y)
        logit_p_hat = logit(p_hat)

        # Step 2: Case-only clustering
        case_mask = (y == CASE_COND)
        kmeans_case = KMeans(n_clusters=2, random_state=0).fit(
            logit_p_hat[case_mask].reshape(-1, 1)
        )
        mean0 = p_hat[case_mask][kmeans_case.labels_ == 0].mean()
        mean1 = p_hat[case_mask][kmeans_case.labels_ == 1].mean()
        zero_lab_has_lower_mean = mean0 < mean1
        kmeans_labels = np.zeros_like(y)
        kmeans_labels[case_mask] = [
            1 if x == int(zero_lab_has_lower_mean) else 0
            for x in kmeans_case.labels_
        ]

        # Step 3: Store results
        result = {
            "factor_index": i,
            "p_hat": p_hat,
            "logit_p_hat": logit_p_hat,
            "kmeans_labels": kmeans_labels,
            "status": y,
            "sample_id": sample_ids,
        }
        all_results.append(result)

        # Step 4: Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(x=logit_p_hat, hue=y, ax=axes[0])
        axes[0].set_title(f'logit(p_hat), original labels (factor {i+1})')
        sns.histplot(x=logit_p_hat, hue=kmeans_labels, ax=axes[1])
        axes[1].set_title(f'logit(p_hat), KMeans labels (factor {i+1})')
        plt.tight_layout()
        plt.show()

    return all_results




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
    at_data_dir = functools.partial(os.path.join, root_path,'SpatialPeeler','data_PSC')
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




##################################################################
########### Running HiDDEN on the NMF ######################

#adata_merged.obsm["X_nmf"] 
#adata_merged.uns["nmf_components"]

adata = adata_merged.copy()
adata_merged.obsm["X_pca"] = adata_merged.obsm["X_nmf"]  # Use NMF factors as PCA scores

### save the adata object
adata_merged.write_h5ad(os.path.join(root_path, 'SpatialPeeler', 'data_PSC', 'PSC_NMF.h5ad'))





## run the model.py script within the hiddensc package
num_pcs, ks, ks_pval = determine_pcs_heuristic_ks_nmf(adata=adata, 
                                                                  orig_label="binary_label", 
                                                                  max_pcs=60)
optimal_num_pcs_ks = num_pcs[np.argmax(ks)]

plt.figure(figsize=(4, 2))
plt.scatter(num_pcs, ks, s=10, c='k', alpha=0.5)
plt.axvline(x = optimal_num_pcs_ks, color = 'r', alpha=0.2)
plt.scatter(optimal_num_pcs_ks, ks[np.argmax(ks)], s=20, c='r', marker='*', edgecolors='k')
plt.xticks(np.append(np.arange(0, 60, 15), optimal_num_pcs_ks), fontsize=18)
plt.xlabel('Number of PCs')
plt.ylabel('KS')
plt.show()

print(f"Optimal number of PCs: {optimal_num_pcs_ks}")
# Set up HiDDEN input
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['status'] = adata.obs['binary_label'].astype(int).values

# Run factor-wise HiDDEN-like analysis (logistic regression on single factors)
results = single_factor_logistic_evaluation(
    adata, factor_key="X_pca", max_factors=optimal_num_pcs_ks
)


# Plotting function
def plot_spatial_p_hat(a, sample_id):
    spatial = a.obsm["spatial"]
    p_hat = a.obs["p_hat"].values

    upper = np.quantile(p_hat, 0.99)
    p_hat = np.minimum(p_hat, upper)

    plt.figure(figsize=(5, 5))
    sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10)
    plt.axis("equal")
    plt.title(f"HiDDEN prediction – {sample_id}", fontsize=14)
    plt.colorbar(sc, label="p_hat")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()

# Store output for the best-performing factor (e.g., first one, or pick based on AUC)
counter = 1
for result in results:
    print(f"Factor {result['factor_index'] + 1}:")
    print(f"  p_hat mean: {result['p_hat'].mean():.4f}")
    print(f"  KMeans labels: {np.unique(result['kmeans_labels'])}")
    print(f"  Status distribution: {np.bincount(result['status'])}")


    adata.obs['p_hat'] = result['p_hat']
    adata.obs['new_label'] = result['kmeans_labels']
    adata.obs['new_label'] = adata.obs['new_label'].astype('category')
    adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')

    # Copy adata per sample for plotting
    sample_ids = adata.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }

    # Plot spatial maps for the first 8 samples
    #for i in range(min(8, len(sample_ids))):
    #    plot_spatial_p_hat(adata_by_sample[sample_ids[i]], sample_ids[i])

    ############# plotting all samples in a grid
    # Get global 99th percentile for consistent color scale
    p_hat_all = adata.obs["p_hat"].values
    vmax = np.quantile(p_hat_all, 0.99)

    # Plot 2x4 grid
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    for i, sid in enumerate(sample_ids):
        a = adata_by_sample[sid]
        spatial = a.obsm["spatial"]
        p_hat = a.obs["p_hat"].clip(upper=vmax)

        axs[i].scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10, vmin=0, vmax=vmax)
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label="p_hat")

    plt.suptitle("HiDDEN predictions (p_hat) across spatial coordinates for factor " 
                 + str(counter), fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()
    counter += 1



    df_violin = adata.obs[["sample_id", "p_hat"]].copy()
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        x="sample_id", y="p_hat", hue="sample_id", data=df_violin,
        palette="Set2", density_norm="width", inner=None, legend=False
    )
    sns.boxplot(
        x="sample_id", y="p_hat", data=df_violin,
        color="white", width=0.1, fliersize=0
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribution of p_hat per sample")
    plt.tight_layout()
    plt.show()




### Gives an average prediction confidence over factors.
# If your goal is to highlight subtle, distributed signals across factors in a robust way:
combined_p_hat_1 = np.mean([r['p_hat'] for r in results], axis=0)


#### Weight by a proxy for factor importance, e.g., variance, AUC, or KS score.
# Example: weight by factor variance
variances = [np.var(r['p_hat']) for r in results]
weights = np.array(variances) / np.sum(variances)
combined_p_hat_2 = np.average([r['p_hat'] for r in results], axis=0, weights=weights)


#### Fit a meta-model (e.g., logistic regression) on the matrix of all p_hat values.
# Stack all p_hat values into a feature matrix (n_cells x n_factors)
X_stack = np.column_stack([r['p_hat'] for r in results])
y = adata.obs['status'].values
# Fit logistic regression on top
clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X_stack, y)
combined_p_hat_3 = clf.predict_proba(X_stack)[:, 1]


# If instead you want to identify cells where any one factor is strongly perturbed (more conservative):
combined_p_hat_4 = np.max([r['p_hat'] for r in results], axis=0)

# Store the combined p_hat in adata
phat_combined = np.column_stack([
    combined_p_hat_1,  # or combined_p_hat_2, 3, or 4
    combined_p_hat_2,
    combined_p_hat_3,
    combined_p_hat_4
]) 

adata.obs['p_hat_combined_1'] = phat_combined[:, 0]
adata.obs['p_hat_combined_2'] = phat_combined[:, 1]
adata.obs['p_hat_combined_3'] = phat_combined[:, 2]
adata.obs['p_hat_combined_4'] = phat_combined[:, 3]

# Stack factor-wise p_hat values
p_hat_matrix = np.column_stack([r["p_hat"] for r in results])  # shape: (n_cells, n_factors)
W = adata.obsm["X_nmf"][:, :p_hat_matrix.shape[1]]  # shape: (n_cells, n_factors)
# Normalize weights row-wise so they sum to 1
W_normalized = W / W.sum(axis=1, keepdims=True)
# Compute combined p_hat per cell as weighted sum
combined_p_hat_spatial = np.sum(W_normalized * p_hat_matrix, axis=1)
adata.obs["p_hat_combined_5"] = combined_p_hat_spatial.astype(np.float32)

### calculate combined p_hat spatially - based on NMF weights

def plot_combined_p_hat(a, sample_id, p_hat_col):
        spatial = a.obsm["spatial"]
        p_hat = a.obs[p_hat_col].values

        upper = np.quantile(p_hat, 0.99)
        p_hat = np.minimum(p_hat, upper)

        plt.figure(figsize=(5, 5))
        sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10)
        plt.axis("equal")
        plt.title(f"Combined p_hat {p_hat_col} – {sample_id}", fontsize=14)
        plt.colorbar(sc, label="p_hat")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()

# Plot the combined p_hat spatially
for i in range(5):
    ############# plotting all samples in a grid
    
    sample_ids = adata.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }
    # Plot 2x4 grid for each combined p_hat
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    for j, sid in enumerate(sample_ids):
        a = adata_by_sample[sid]
        p_hat_col = f'p_hat_combined_{i+1}'
        spatial = a.obsm["spatial"]
        p_hat = a.obs[p_hat_col].clip(upper=np.quantile(adata.obs[p_hat_col], 0.99))

        axs[j].scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10)
        axs[j].set_title(sid, fontsize=10)
        axs[j].axis("off")
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="p_hat")
    plt.suptitle(f"Combined p_hat {p_hat_col} across spatial coordinates", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


# Save the updated AnnData object with combined p_hat
### make violin plots for all combined p_hat across samples
df_violin = adata.obs[["sample_id", 
                       "p_hat_combined_1", 
                        "p_hat_combined_2",
                        "p_hat_combined_3",
                        "p_hat_combined_4",
                        "p_hat_combined_5"
                       ]].copy()

for i in range(1, 6):
    df_violin = adata.obs[["sample_id", 
                           "p_hat_combined_"+str(i)]].copy()
    df_violin.columns = ["sample_id", "p_hat_combined_"+str(i)]
    
    # Plot single violin plot for each combined p_hat
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        x="sample_id",
        y="p_hat_combined_"+str(i),  # Change this to 2, 3, etc. for other combined p_hat
        hue="sample_id",
        data=df_violin,
        palette="Set2",
        density_norm="width",
        inner=None,
        legend=False
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribution of p_hat_combined_{i} per sample")
    plt.tight_layout()
    plt.show()


# Plot multi-violin plots
df_violin_long = df_violin.melt(
    id_vars="sample_id",
    var_name="Combination",
    value_name="p_hat"
)
plt.figure(figsize=(20, 8))
sns.violinplot(
    x="sample_id", y="p_hat", hue="Combination", data=df_violin_long,
    split=False, inner="box", palette="Set2", dodge=True
)
plt.xticks(rotation=45, ha="right")
plt.title("Comparison of Combined p_hat Scores Across Samples")
plt.xlabel("Sample ID")
plt.ylabel("p_hat")
plt.legend(title="Combination", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()





psc_mask = adata.obs["disease"] != "normal"
psc_adata = adata[psc_mask].copy()
psc_sample_ids = psc_adata.obs["sample_id"].unique()
from sklearn.cluster import KMeans

cluster_labels = pd.Series(index=psc_adata.obs_names, dtype="object")

for sid in psc_sample_ids:
    sample_mask = psc_adata.obs["sample_id"] == sid
    p_hat_vals = psc_adata.obs.loc[sample_mask, "p_hat_combined_1"].values.reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
    labels = kmeans.fit_predict(p_hat_vals)

    # Optional: force cluster 1 to be higher-mean cluster
    mean0, mean1 = p_hat_vals[labels == 0].mean(), p_hat_vals[labels == 1].mean()
    # If mean0 is greater, flip labels to ensure cluster 1 has higher mean
    if mean0 > mean1:
        labels = 1 - labels
        temp = mean0
        mean0 = mean1
        mean1 = temp
    # Print means for debugging
    print(f"Sample {sid}: mean0={mean0:.4f}, mean1={mean1:.4f}")

    cluster_labels.loc[sample_mask] = labels

psc_adata.obs["kmeans_p_hat_combined_1"] = cluster_labels.astype("category")


def plot_clusters_per_sample(adata_dict, col="kmeans_p_hat_combined_1", cmap="tab10"):
    n = len(adata_dict)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, (sid, a) in enumerate(adata_dict.items()):
        spatial = a.obsm["spatial"]
        clusters = a.obs[col].astype(int).values

        axs[i].scatter(spatial[:, 0], spatial[:, 1], c=clusters, cmap=cmap, s=10)
        axs[i].set_title(sid)
        axs[i].axis("off")

    for j in range(i+1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("KMeans Clustering of p_hat_combined_1 in PSC samples", fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage:
# Split adata by sample
psc_by_sample = {
    sid: psc_adata[psc_adata.obs["sample_id"] == sid].copy()
    for sid in psc_sample_ids
}

plot_clusters_per_sample(psc_by_sample, cmap="Paired")  # Or try "Paired", "Accent", "Dark2"

###