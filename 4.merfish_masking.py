import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import scanpy as sc
import scvi
import anndata
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
from scipy.sparse import issparse
from functools import reduce

from SpatialPeeler import helpers as hlps
from SpatialPeeler import case_prediction as cpred
from SpatialPeeler import plotting as plot
from SpatialPeeler import gene_identification as gid

import pickle




### kmeans clustering of p_hat_factor_case
from sklearn.cluster import KMeans


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
CASE_COND_NAME = 'infected'


with open('/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/results.pkl', 'rb') as f:
    results = pickle.load(f)
outp = "/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/merged_adata_infected_sham_allnorm_NMF30.h5ad"
adata = sc.read_h5ad(outp)

sample_ids = adata.obs['sample_id'].unique().tolist()


GOF_index = [6, 7, 29, 12, 23, 17]
LOF_index = [3, 9, 11, 21, 13, 4, 15, 10]

factor_idx = 29

for factor_idx in GOF_index:
    print(f"Factor {factor_idx+1}")
    result = results[factor_idx]
    p_hat_factor = result['p_hat']
    p_hat_factor_case = p_hat_factor[adata.obs['Condition'] == CASE_COND_NAME]
    p_hat_factor_case_df = pd.DataFrame(p_hat_factor_case, columns=['p_hat'])
    #plt.figure(figsize=(5, 5))
    #sns.histplot(p_hat_factor_case_df['p_hat'], bins=30, kde=True)
    #plt.title(f"Factor {factor_idx+1}")
    #plt.xlabel("p-hat for case samples")
    #plt.ylabel("Count")
    #plt.show()
    
    kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
    kmeans.fit(p_hat_factor_case.reshape(-1, 1))

    ### add cluster information to adata - to the case observations with the correct order
    mask_case = (adata.obs['Condition'].values == CASE_COND_NAME)

    # Sanity check: lengths must match
    print(mask_case.sum() == len(kmeans.labels_))

    # Remap labels so that 1 = higher p-hat, 0 = lower p-hat
    centers = kmeans.cluster_centers_.ravel()
    order = np.argsort(centers)               # [low_center_label, high_center_label]
    remap = {order[0]: 0, order[1]: 1}
    labels_remapped = np.vectorize(remap.get)(kmeans.labels_).astype(int)

    # Create an obs column name that encodes which factor you clustered (here factor 7)
    obs_col = 'phat_cluster_factor' + str(factor_idx+1)

    # Initialize with NaN for all, then fill case rows in-place to preserve order
    adata.obs[obs_col] = np.nan
    adata.obs.loc[mask_case, obs_col] = labels_remapped

    # make it categorical and store centers for reference
    adata.obs[obs_col] = pd.Categorical(adata.obs[obs_col], categories=[0, 1])
    ### convert to string
    adata.obs[obs_col] = adata.obs[obs_col].astype(str)
    
    adata.obs['phat_factor'+str(factor_idx+1)] = result['p_hat']
    

    ### calculate residual: Y-phat
    logit_phat = plot.safe_logit(adata.obs['phat_factor'+str(factor_idx+1)].values)
    adata.obs['Condition_binary'] = (adata.obs['Condition'] == CASE_COND_NAME).astype(int)
    Y = adata.obs['Condition_binary'].values
    response_residual = Y - logit_phat
    adata.obs['residual'+str(factor_idx+1)] = response_residual

    plt.figure(figsize=(5, 5))
    sns.histplot(adata.obs['residual'+str(factor_idx+1)], bins=30, kde=True)
    plt.title(f"Factor {factor_idx+1}")
    plt.xlabel("Residual for all samples")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(5, 5))
    sns.histplot(adata.obs['residual'+str(factor_idx+1)][adata.obs['Condition'] == CASE_COND_NAME], bins=30, kde=True)
    plt.title(f"Factor {factor_idx+1}")
    plt.xlabel("Residual for case samples")
    plt.ylabel("Count")
    plt.show()

    ### visualize p-hat scores over cluster 0 and 1 as violin plot
    plt.figure(figsize=(5, 5))
    sns.violinplot(x=obs_col, y='phat_factor'+str(factor_idx+1), data=adata.obs)
    plt.title(f"Violin plot of p-hat scores (factor {factor_idx+1})")
    plt.xlabel("Cluster")
    plt.ylabel("p-hat")
    plt.show()

    ### visualize p-hat scores over cluster 0 and 1 as violin plot
    plt.figure(figsize=(5, 5))
    sns.violinplot(x=obs_col, y='residual'+str(factor_idx+1), data=adata.obs)
    plt.title(f"Violin plot of residual scores (factor {factor_idx+1})")
    plt.xlabel("Cluster")
    plt.ylabel("Residual")
    plt.show()

    
    # Copy adata per sample for plotting
    sample_ids = adata.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }

    #plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=obs_col,
    #        title_prefix="Clusters (factor "+str(factor_idx+1)+")", from_obsm=False, discrete=True,
    #        palette={0: "#4C78A8", 1: "#F58518"}, dot_size=2, figsize=(25, 10))

    plot.plot_grid_upgrade(adata_by_sample, sample_ids, key='residual'+str(factor_idx+1),
                           title_prefix="Residuals (factor "+str(factor_idx+1)+")", 
                           from_obsm=False, discrete=False,
                            dot_size=2, figsize=(25, 10))
