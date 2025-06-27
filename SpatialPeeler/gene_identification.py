import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import pandas as pd
import numpy as np
import hiddensc
from hiddensc import utils, files, vis
import scanpy as sc
import scvi
import anndata
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed


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


from scipy.stats import pearsonr

from statsmodels.api import OLS, add_constant
from scipy.stats import zscore

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
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




def regression_with_residuals(expr_matrix, residual_vector, gene_names, scale=True):
    """
    Fit univariate linear regression of residual_vector ~ gene_expression for each gene.

    Parameters
    ----------
    expr_matrix : np.ndarray (n_cells, n_genes)
        Gene expression matrix.
    residual_vector : np.ndarray (n_cells,)
        Residuals (e.g., from logistic regression).
    gene_names : list or np.ndarray
        List of gene names corresponding to columns of expr_matrix.
    scale : bool
        Whether to z-score scale both gene expression and residuals before regression.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'gene', 'slope', 'pval', and 'r_squared' columns.
    """
    slopes = []
    pvals = []
    r_squared = []

    for g in range(expr_matrix.shape[1]):
        x = expr_matrix[:, g]
        if scale:
            # Scale both gene expression and residuals
            x = zscore(x)
            residual_vector = zscore(residual_vector)

        # Fit OLS regression
        
        if np.std(x) == 0 or np.std(residual_vector) == 0:
            slopes.append(np.nan)
            pvals.append(np.nan)
            r_squared.append(np.nan)
            continue

        X = add_constant(x)
        model = OLS(residual_vector, X).fit()

        slopes.append(model.params[1])        # β₁
        pvals.append(model.pvalues[1])        # p-value for β₁
        r_squared.append(model.rsquared)      # goodness of fit

    return pd.DataFrame({
        "gene": gene_names,
        "slope": slopes,
        "pval": pvals,
        "r_squared": r_squared
    })



def fit_gp_similarity_scores(expr_matrix, residual_vector, coords, gene_names, 
                             kernel=None):
    """
    Fit a GP to the residuals and a GP to each gene expression vector, compare spatial similarity.

    Parameters
    ----------
    expr_matrix : np.ndarray (n_cells, n_genes)
        Gene expression matrix.
    residual_vector : np.ndarray (n_cells,)
        Residuals from a model (e.g., Pearson residuals).
    coords : np.ndarray (n_cells, 2)
        Spatial coordinates (e.g., adata.obsm["spatial"]).
    gene_names : list or np.ndarray
        Gene names.
    kernel : sklearn GP kernel (optional)
        If None, uses RBF + WhiteKernel by default.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'gene' and 'similarity_to_residual_GP'.
    """
    if kernel is None:
        kernel = 1.0 * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1)

    # Fit GP to residuals
    gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    gp_r.fit(coords, residual_vector)
    residual_gp_mean = gp_r.predict(coords)

    # Fit GP to each gene and compute correlation with residual GP
    similarity_scores = []
    for g in range(expr_matrix.shape[1]):
        
        # Use hyperparameter optimization for each gene (optimizer='fmin_l_bfgs')
        gp_g = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)    
        gp_g.fit(coords, expr_matrix[:, g]) ### re-fit for each gene - O(n_cells^3)
        gene_gp_mean = gp_g.predict(coords)

        corr, _ = pearsonr(gene_gp_mean, residual_gp_mean)
        similarity_scores.append(corr)

    return pd.DataFrame({
        "gene": gene_names,
        "similarity_to_residual_GP": similarity_scores
    }).sort_values(by="similarity_to_residual_GP", ascending=False)


def fit_gp_similarity_scores_fastmode(expr_matrix, residual_vector, coords, gene_names, 
                             kernel=None, hyperopt=True, n_jobs=1):
    """
    Fit a GP to the residuals and a GP to each gene expression vector, compare spatial similarity.

    Parameters
    ----------
    expr_matrix : np.ndarray (n_cells, n_genes)
        Gene expression matrix.
    residual_vector : np.ndarray (n_cells,)
        Residuals from a model (e.g., Pearson residuals).
    coords : np.ndarray (n_cells, 2)
        Spatial coordinates (e.g., adata.obsm["spatial"]).
    gene_names : list or np.ndarray
        Gene names.
    kernel : sklearn GP kernel (optional)
        If None, uses RBF + WhiteKernel by default.
    hyperopt : bool
        Whether to optimize hyperparameters for each gene-specific GP.
    n_jobs : int
        Number of parallel jobs for per-gene GP fitting. Set >1 for parallelization.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'gene' and 'similarity_to_residual_GP'.
    """
    if kernel is None:
        kernel = 1.0 * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1)

    #  Fit GP to residuals 
    gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    gp_r.fit(coords, residual_vector)
    residual_gp_mean = gp_r.predict(coords)

    # If not hyperopting, extract the kernel after residual fit (re-using for all genes)
    fitted_kernel = gp_r.kernel_ if not hyperopt else kernel

    # Define function for per-gene GP 
    def fit_and_score_gene(g):
        y = expr_matrix[:, g]
        if hyperopt:
            gp_g = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
        else:
            gp_g = GaussianProcessRegressor(kernel=fitted_kernel, optimizer=None, 
                                            alpha=1e-4, normalize_y=True)
        gp_g.fit(coords, y)
        gene_gp_mean = gp_g.predict(coords)
        corr, _ = pearsonr(gene_gp_mean, residual_gp_mean)
        return gene_names[g], corr

    # Run in parallel 
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score_gene)(g) for g in range(expr_matrix.shape[1])
    )

    return pd.DataFrame(results, columns=["gene", "similarity_to_residual_GP"])\
             .sort_values(by="similarity_to_residual_GP", ascending=False)




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
