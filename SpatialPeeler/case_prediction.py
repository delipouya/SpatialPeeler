import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import hiddensc
from hiddensc import utils, vis

import scanpy as sc
import scvi
import anndata
from sklearn.decomposition import NMF

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.special import logit
import statsmodels.api as sm

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
from sklearn.cluster import KMeans
from scipy.special import logit



def standalone_logistic_v1(X, y):
    #clf = LogisticRegression(random_state=RAND_SEED, penalty='none').fit(X, y)
    clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X, y)
    predicted_label = clf.predict(X)
    predicted_prob = clf.predict_proba(X)
    return predicted_prob[:,1]


def standalone_logistic_v2(X, y):
    clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X, y)
    predicted_label = clf.predict(X)
    predicted_prob = clf.predict_proba(X)
    coef = clf.coef_       # shape: (1, n_features)
    intercept = clf.intercept_  # shape: (1,)
    return predicted_prob[:, 1], coef, intercept


def standalone_logistic(X, y):
    # Add intercept explicitly
    X_with_intercept = sm.add_constant(X)

    # Fit model using statsmodels for inference
    model = sm.Logit(y, X_with_intercept)
    result = model.fit(disp=False)

    predicted_prob = result.predict(X_with_intercept)  # returns P(y=1|x)

    coef = result.params          # includes intercept and weights
    stderr = result.bse           # standard errors for each beta
    pvals = result.pvalues        # p-values (Wald test)

    return predicted_prob, coef, stderr, pvals


def single_factor_logistic_evaluation(adata, factor_key="X_nmf", max_factors=30):
    all_results = []
    X = adata.obsm[factor_key]
    y = adata.obs["status"].values
    sample_ids = adata.obs["sample_id"].values

    for i in range(min(max_factors, X.shape[1])):
        print(f"Evaluating factor {i+1}...")
        Xi = X[:, i].reshape(-1, 1)  # single factor
        print(Xi)
        print(Xi.min(), Xi.max())

        # Logistic regression with full inference
        p_hat, coef, stderr, pvals = standalone_logistic(Xi, y)
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
        logit_p_hat = logit(p_hat)

        # Clustering in logit space for cases
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

        # Residuals
        raw_residual = Xi.flatten() - p_hat
        pearson_residual = raw_residual / np.sqrt(p_hat * (1 - p_hat))
        deviance_residual = np.sign(raw_residual) * np.sqrt(
            -2 * (y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
        )

        # Save the results
        result = {
            "factor_index": i,
            "coef": float(coef[1]),                    # β (slope)
            "intercept": float(coef[0]),               # β₀
            "std_err": float(stderr[1]),               # SE(β)
            "std_err_intercept": float(stderr[0]),     # SE(β₀)
            "pval": float(pvals[1]),                   # p(β ≠ 0)
            "pval_intercept": float(pvals[0]),         # p(β₀ ≠ 0)
            "p_hat": p_hat,
            "logit_p_hat": logit_p_hat,
            "kmeans_labels": kmeans_labels,
            "status": y,
            "sample_id": sample_ids,
            "raw_residual": raw_residual,
            "pearson_residual": pearson_residual,
            "deviance_residual": deviance_residual
            }
        
        all_results.append(result)

        # Optional visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(x=logit_p_hat, hue=y, ax=axes[0])
        axes[0].set_title(f"logit(p̂), original labels (factor {i+1})")
        sns.histplot(x=logit_p_hat, hue=kmeans_labels, ax=axes[1])
        axes[1].set_title(f"logit(p̂), KMeans labels (factor {i+1})")
        plt.tight_layout()
        plt.show()

    return all_results
