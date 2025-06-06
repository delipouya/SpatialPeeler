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
import scipy.sparse as sp

import hiddensc
from hiddensc import utils, files, vis

import scanpy as sc
import scvi
import anndata
from sklearn.decomposition import NMF
import functools

from scipy.linalg import svd
from numpy.linalg import LinAlgError

from hiddensc import models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit

from sklearn.neighbors import NearestNeighbors

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

from hiddensc import models
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit




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
        # Added this line to ensure p_hat is within valid range
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
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

        raw_residual = Xi.flatten() - p_hat
        pearson_residual = raw_residual / np.sqrt(p_hat * (1 - p_hat))
        deviance_residual = np.sign(raw_residual) * np.sqrt(
            -2 * (y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
        )

        # Step 3: Store results
        result = {
            "factor_index": i,
            "p_hat": p_hat,
            "logit_p_hat": logit_p_hat,
            "kmeans_labels": kmeans_labels,
            "status": y,
            "sample_id": sample_ids,
            'raw_residual': raw_residual,
            # raw residual divided by the estimated standard deviation of a binomial distribution
            'pearson_residual': pearson_residual,
            # d <- sign(e)*sqrt(-2*(y*log(p_hat) + (1 - y)*log(1 - p_hat)))
            'deviance_residual': deviance_residual 
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
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
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
        axs[i].set_title(sid, fontsize=10)
        axs[i].set_xlabel(f"NMF Factor {factor_idx + 1}")
        axs[i].set_ylabel("p_hat")
        axs[i].grid(True)

    # Turn off unused plots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle(f"p_hat vs NMF Factor {factor_idx + 1} by sample", fontsize=14)
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
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
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

        axs[i].scatter(nmf_scores, logit_p_hat, alpha=0.5, s=10)
        axs[i].set_title(sid, fontsize=10)
        axs[i].set_xlabel(f"NMF Factor {factor_idx + 1}")
        axs[i].set_ylabel("logit(p̂)")
        axs[i].grid(True)

    # Turn off unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle(f"logit(p̂) vs NMF Factor {factor_idx + 1} by sample", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()



def plot_grid(adata_by_sample, sample_ids, key, title_prefix, counter, from_obsm=False, factor_idx=None):
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



from scipy.stats import pearsonr

def pearson_correlation_with_residuals(expr_matrix, residual_vector, gene_names):
    """
    Compute standard Pearson correlation between each gene and residuals.

    Parameters
    ----------
    expr_matrix : np.ndarray (n_cells, n_genes)
        Gene expression matrix.
    residual_vector : np.ndarray (n_cells,)
        Residuals to correlate with each gene (e.g., pearson or raw).
    gene_names : list or np.ndarray
        List of gene names corresponding to columns of expr_matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'gene' and 'correlation' columns.
    """
    correlations = []
    for g in range(expr_matrix.shape[1]):
        x = expr_matrix[:, g]
        if np.std(x) == 0 or np.std(residual_vector) == 0:
            correlations.append(np.nan)
        else:
            r, _ = pearsonr(x, residual_vector)
            correlations.append(r)

    return pd.DataFrame({
        "gene": gene_names,
        "correlation": correlations
    })



def compute_spatial_weights(coords, k=10, mode="geary"):
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    weights = np.zeros((n, n))
    d_max = np.max(distances)

    for i in range(n):
        for j_idx in range(1, k+1):  # skip self
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            if mode == "geary":
                weights[i, j] = 1.0 / (1 + d**2)
            elif mode == "moran":
                weights[i, j] = 1.0 - d / d_max
            else:
                raise ValueError("mode must be 'geary' or 'moran'")

    W = weights + weights.T ## symmertric
    W = W / W.sum()
    return W


def spatial_weighted_correlation_matrix_v1(expr_matrix, residual_vector, W):
    """
    Compute spatially weighted correlation between each gene and residuals.

    Parameters
    ----------
    expr_matrix : np.ndarray (n_cells, n_genes)
        Gene expression matrix.
    residual_vector : np.ndarray (n_cells,)
        Residuals to correlate with each gene (e.g., pearson or raw).
    mode : str
        "geary" or "moran" style weighting.

    Returns
    -------
    pd.Series
        Spatially weighted correlation per gene.
    """
    n = len(residual_vector)
    residual_centered = residual_vector - residual_vector.mean()
    # Correlation per gene
    correlations = []
    for g in range(expr_matrix.shape[1]):
        x = expr_matrix[:, g]
        x_centered = x - x.mean()
        cov = np.sum(W * np.outer(x_centered, residual_centered))
        var_x = np.sum(W * np.outer(x_centered, x_centered))
        var_y = np.sum(W * np.outer(residual_centered, residual_centered))
        if var_x == 0 or var_y == 0:
            correlations.append(np.nan)
        else:
            corr = cov / np.sqrt(var_x * var_y)
            correlations.append(corr)

    return pd.Series(correlations)


def spatial_weighted_correlation_matrix(expr_matrix, residual_vector, W, gene_names=None):
    """
    Compute spatially weighted correlation between each gene and residuals (vectorized).

    Parameters
    ----------
    expr_matrix : np.ndarray (n_cells, n_genes)
        Gene expression matrix (dense or sparse, should be float).
    residual_vector : np.ndarray (n_cells,)
        Residuals to correlate with each gene (e.g., raw or Pearson).
    W : np.ndarray (n_cells, n_cells)
        Symmetric normalized spatial weight matrix.

    Returns
    -------
    pd.Series
        Spatially weighted correlation per gene.
    """
    # Center residuals
    y = residual_vector - residual_vector.mean()
    # Center expression matrix (each gene)
    X = expr_matrix - expr_matrix.mean(axis=0)
    # Compute weighted covariance between each gene and residual
    cov_xy = X.T @ (W @ y)  # shape: (n_genes,)
    # Compute spatially weighted variance of residual
    Wy = W @ y
    var_y = np.dot(y, Wy)
    # Compute spatially weighted variance per gene
    WX = W @ X  # shape: (n_cells, n_genes)
    var_x = np.sum(X * WX, axis=0)  # shape: (n_genes,)
    # Compute correlations
    denom = np.sqrt(var_x * var_y)
    correlations = np.divide(cov_xy, denom, out=np.full_like(cov_xy, np.nan), where=denom > 0)

    return pd.DataFrame({
        "gene": gene_names,
        "correlation": correlations
    })



def weighted_mean(x, W):
    weights = W.sum()
    return (x.T @ W @ np.ones_like(x)) / weights

def weighted_cov(x, y, W):
    x = x.flatten()
    y = y.flatten()
    mu_x = weighted_mean(x, W)
    mu_y = weighted_mean(y, W)

    x_centered = x - mu_x
    y_centered = y - mu_y

    numerator = x_centered.T @ W @ y_centered
    denominator = W.sum()

    return numerator / denominator

def weighted_corr(x, y, W):
    cov_xy = weighted_cov(x, y, W)
    var_x = weighted_cov(x, x, W)
    var_y = weighted_cov(y, y, W)
    return cov_xy / np.sqrt(var_x * var_y)




import mygene
def map_ensembl_to_symbol(ensembl_ids):
    mg = mygene.MyGeneInfo()
    result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
    
    # Build mapping dict
    id_to_symbol = {r['query']: r.get('symbol', None) for r in result}
    return id_to_symbol



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
