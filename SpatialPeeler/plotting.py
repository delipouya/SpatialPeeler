import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt
import hiddensc
from hiddensc import utils, vis
import scanpy as sc
import scvi
import anndata


#from scipy.stats import mannwhitneyu
RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.simplefilter("ignore", category=ConvergenceWarning)


def plot_gene_spatial(adata, gene_id, title, cmap="viridis"):
    """Plot the expression of a single gene over spatial coordinates."""
    if gene_id not in adata.var_names:
        print(f"Gene {gene_id} not found in adata.var_names.")
        return
    gene_idx = adata.var_names.get_loc(gene_id)
    expr = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, "toarray") else adata.X[:, gene_idx]
    coords = adata.obsm["spatial"]
    plt.figure(figsize=(5, 4))
    plt.scatter(coords[:, 0], coords[:, 1], c=expr, cmap=cmap, s=10)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(label="Expression")
    plt.tight_layout()
    plt.show()



def plot_spatial_nmf(adata, factor_idx, sample_id=None, figsize=(10, 10)):
    spatial = adata.obsm["spatial"]
    values = adata.obsm["X_nmf"][:, factor_idx]

    # Clip extremes for visualization
    vmax = np.quantile(values, 0.99)
    vmin = np.quantile(values, 0.01)
    values = np.clip(values, vmin, vmax)

    plt.figure(figsize=figsize)
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



def plot_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx):
    """
    Plot p_hat vs. NMF factor score for each sample in a grid.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare layout
    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 20))
    axs = axs.flatten()

    # Global scores
    nmf_scores_all = adata.obsm["X_nmf"][:, factor_idx]
    p_hat_all = results[factor_idx]['p_hat']
    obs_names = adata.obs_names

    for i, sid in enumerate(sample_ids):
        sample_mask = adata.obs["sample_id"] == sid
        idx = np.where(sample_mask)[0]

        nmf_scores = nmf_scores_all[idx]
        p_hat = p_hat_all[idx]

        axs[i].scatter(nmf_scores, p_hat, alpha=0.5, s=10)
        axs[i].set_title(sid, fontsize=18)
        axs[i].set_xlabel(f"NMF Factor {factor_idx + 1}", fontsize=20)
        axs[i].set_ylabel("p_hat", fontsize=20)
        axs[i].set_ylim(0, 1) ## fix the range
        axs[i].grid(True)

    # Turn off unused plots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle(f"p_hat vs NMF Factor {factor_idx + 1} by sample", fontsize=23)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def plot_logit_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx):
    """
    Plot logit(p_hat) vs. NMF factor score for each sample in a grid.
    """
    def safe_logit(p, eps=1e-5):
        # Avoid division by zero or log(0)
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    # Layout
    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 20))
    axs = axs.flatten()

    # Global arrays
    nmf_scores_all = adata.obsm["X_nmf"][:, factor_idx]
    p_hat_all = results[factor_idx]['p_hat']
    logit_p_hat_all = safe_logit(p_hat_all)

    for i, sid in enumerate(sample_ids):
        sample_mask = adata.obs["sample_id"] == sid
        idx = np.where(sample_mask)[0]

        nmf_scores = nmf_scores_all[idx]
        logit_p_hat = logit_p_hat_all[idx]

        axs[i].scatter(nmf_scores, logit_p_hat, alpha=0.7, s=10)
        axs[i].set_title(sid, fontsize=14)
        axs[i].set_xlabel(f"NMF Factor {factor_idx + 1}")
        axs[i].set_ylabel("logit(p̂)")
        axs[i].grid(True)

    # Turn off unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle(f"logit(p̂) vs NMF Factor {factor_idx + 1} by sample", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()



def plot_grid(adata_by_sample, sample_ids, key, title_prefix, counter, 
              from_obsm=False, factor_idx=None, figsize=(16, 8), fontsize=10):
    """
    Plots a spatial grid of actual (unclipped) values from .obs or .obsm.

    Parameters:
    - adata_by_sample: dict of AnnData objects per sample
    - sample_ids: list of sample IDs
    - key: string key in .obs or .obsm (e.g. 'p_hat', 'raw_residual') 
    - title_prefix: str for figure title
    - counter: factor index (for labeling)
    - from_obsm: if True, extract from .obsm using key and factor_idx
    - factor_idx: index of the factor (only needed for .obsm values)
    """
    # Collect all values across samples
    if from_obsm and factor_idx is not None:
        all_vals = np.concatenate([
            adata_by_sample[sid].obsm[key][:, factor_idx]
            for sid in sample_ids
        ])
    else:
        all_vals = np.concatenate([
            adata_by_sample[sid].obs[key].values
            for sid in sample_ids
        ])

    # Use global min and max for color scale, with no clipping
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    # Grid layout
    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    # Plot each sample
    for i, sid in enumerate(sample_ids):
        ad = adata_by_sample[sid]
        coords = ad.obsm["spatial"]

        if from_obsm and factor_idx is not None:
            values = ad.obsm[key][:, factor_idx]
        else:
            values = ad.obs[key].values

        im = axs[i].scatter(
            coords[:, 0], coords[:, 1],
            c=values,
            cmap="viridis",
            s=10,
            vmin=vmin,
            vmax=vmax
        )
        axs[i].set_title(sid, fontsize=fontsize)
        axs[i].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    ### increase fontsize of colorbar
    cbar_ax.tick_params(labelsize=fontsize)
    cb = fig.colorbar(im, cax=cbar_ax)
    # Set label with increased fontsize
    cb.set_label(key.replace("_", " ").title(), fontsize=fontsize+2)

    # Title
    plt.suptitle(f"{title_prefix} across spatial coordinates for factor {counter}", fontsize=fontsize+3)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()



def plot_grid_cliped(adata_by_sample, sample_ids, key, title_prefix, counter, 
              from_obsm=False, factor_idx=None):
    """
    Plots a spatial grid of values from .obs or .obsm with consistent NMF-style coloring.
    
    Parameters:
    - adata_by_sample: dict of AnnData objects per sample
    - sample_ids: list of sample IDs
    - key: string key in .obs or .obsm (e.g. 'p_hat', 'raw_residual') 
    - title_prefix: str for suptitle
    - counter: factor index (for labeling)
    - from_obsm: if True, extracts from .obsm using key and factor_idx
    - factor_idx: index of the factor (only needed for .obsm values)
    """
    # Collect all values across samples
    if from_obsm and factor_idx is not None:
        all_vals = np.concatenate([
            adata_by_sample[sid].obsm[key][:, factor_idx]
            for sid in sample_ids
        ])
    else:
        all_vals = np.concatenate([
            adata_by_sample[sid].obs[key].values
            for sid in sample_ids
        ])

    # Color scale using 1st and 99th percentiles
    vmin, vmax = np.quantile(all_vals, [0.01, 0.99])

    # Grid layout
    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axs = axs.flatten()

    # Plot each sample
    for i, sid in enumerate(sample_ids):
        ad = adata_by_sample[sid]
        coords = ad.obsm["spatial"]

        if from_obsm and factor_idx is not None:
            values = ad.obsm[key][:, factor_idx]
            display_values = np.clip(values, vmin, vmax)
        else:
            values = ad.obs[key].values
            display_values = np.clip(values, vmin, vmax)

        im = axs[i].scatter(
            coords[:, 0], coords[:, 1],
            c=display_values,
            cmap="viridis",
            s=10,
            vmin=vmin, vmax=vmax
        )
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(key.replace("_", " ").title(), fontsize=12)

    # Title
    plt.suptitle(f"{title_prefix} across spatial coordinates for factor {counter}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()
