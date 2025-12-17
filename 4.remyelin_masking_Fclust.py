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
from sklearn.cluster import KMeans

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
CASE_COND_NAME = 'LPC'


results_filename = 'remyelin_nmf30_hidden_logistic_Fclust_t3_7.pkl'

# Load
with open(results_filename, 'rb') as f:
    results = pickle.load(f)


outp = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7.h5ad'
#outp = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18.h5ad'
adata = sc.read_h5ad(outp)

adata.obs['binary_label'] = adata.obs['Condition'].apply(lambda x: 1 if x == 'LPC' else 0)
adata.obs['status'] = adata.obs['binary_label'].astype(int).values
sample_ids = adata.obs['sample_id'].unique().tolist()


cluster_mask_dict = {}
factor_idx = 0
print(f"Factor {factor_idx+1}")


def get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1):
        sig_de = de_results[
            (de_results['pvals_adj'] < fdr_threshold) &
            (np.abs(de_results['logfoldchanges']) > logfc_threshold)
        ]
        return sig_de.shape[0]


t3_7_gof = [9, 21, 18, 11, 1, 2, 5, 23]
t18_gof = [9, 2, 12, 18]
t18_lof = [3, 6, 19]
t7_gof = [0, 12, 20, 2]
t7_lof = [10, 18, 7]

factor_idx = t3_7_gof[0]

for factor_idx in t3_7_gof: #range(min(max_factors, X.shape[1])) ,3, 6, 19,
    print(f"Factor {factor_idx+1}")
    result = results[factor_idx]
    p_hat_factor = result['p_hat']
    high_cluster_indices = result['high_cluster_indices']

    
    adata_sub = adata[high_cluster_indices].copy()

    p_hat_factor_case = p_hat_factor[adata_sub.obs['Condition'] == CASE_COND_NAME]
    p_hat_factor_case_df = pd.DataFrame(p_hat_factor_case, columns=['p_hat'])
    
    plt.figure(figsize=(5, 5))
    sns.histplot(p_hat_factor_case_df['p_hat'], bins=30, kde=True)
    plt.title(f"Factor {factor_idx+1}")
    plt.xlabel("p-hat for case samples")
    plt.ylabel("Count")
    plt.show()
    
    kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
    kmeans.fit(p_hat_factor_case.reshape(-1, 1))

    ### add cluster information to adata - to the case observations with the correct order
    mask_case = (adata_sub.obs['Condition'].values == CASE_COND_NAME)
    

    # Sanity check: lengths must match
    print(mask_case.sum() == len(kmeans.labels_))

    # Remap labels so that 1 = higher p-hat, 0 = lower p-hat
    centers = kmeans.cluster_centers_.ravel()
    order = np.argsort(centers)               # [low_center_label, high_center_label]
    remap = {order[0]: 0, order[1]: 1}
    labels_remapped = np.vectorize(remap.get)(kmeans.labels_).astype(int)

    # Calculate the threshold as the midpoint between the centroids
    threshold = np.mean(centers)
    print(f"Centroids: {centers.flatten()}")
    print(f"Binarization Threshold: {threshold}")

    # Create an obs column name that encodes which factor you clustered (here factor 7)
    obs_col = 'phat_cluster_factor' + str(factor_idx+1)
    ### score the p-hat values for all cells
    adata_sub.obs['phat_factor'+str(factor_idx+1)] = result['p_hat']

    # Initialize with NaN for all, then fill case rows in-place to preserve order
    adata_sub.obs[obs_col] = np.nan
    adata_sub.obs.loc[mask_case, obs_col] = labels_remapped

    # make it categorical and store centers for reference
    adata_sub.obs[obs_col] = pd.Categorical(adata_sub.obs[obs_col], categories=[0, 1])
    ### convert to string
    adata_sub.obs[obs_col] = adata_sub.obs[obs_col].astype(str)
    ## add 'case' to the obs_col values
    adata_sub.obs[obs_col] = adata_sub.obs[obs_col].apply(lambda x: x if pd.isna(x) else 'case_' + str(x))
    ## replace case_nan with control
    adata_sub.obs[obs_col] = adata_sub.obs[obs_col].replace('case_nan', 'control')
    ### visualize p-hat scores over cluster 0 and 1 as violin plot
    plt.figure(figsize=(6, 5))
    sns.violinplot(x=obs_col, y='phat_factor'+str(factor_idx+1), 
                   data=adata_sub.obs,
                   order=['control','case_0', 'case_1'])
    plt.title(f"Violin plot of p-hat scores (factor {factor_idx+1})")
    plt.axhline(y=threshold, color='r', linestyle='--')
    
    plt.xlabel("Cluster")
    plt.ylabel("p-hat")
    plt.show()

    ### split the obs_col values equal to 'case_nan' into 'control_0' and 'control_1' based on the threshold derived from case samples
    def assign_control_label(row):
        if row[obs_col].startswith('case_0') or row[obs_col].startswith('case_1'):
            return row[obs_col]
        else:
            if row['phat_factor'+str(factor_idx+1)] >= threshold:
                return 'control_1'
            else:
                return 'control_0'


    adata_sub.obs[obs_col] = adata_sub.obs.apply(assign_control_label, axis=1)
    
    ## check if the obs_col values are string
    print(adata_sub.obs[obs_col].dtype)
    cluster_mask_dict[obs_col] = adata_sub.obs[obs_col]
    ### make it categorical
    adata_sub.obs[obs_col] = pd.Categorical(adata_sub.obs[obs_col], categories=['control_0', 'control_1', 'case_0', 'case_1'])
    adata_sub.obs[obs_col] = adata_sub.obs[obs_col].astype(str)
    ## check the stats of each cluster
    print(adata_sub.obs[obs_col].value_counts())

    ### visualize p-hat scores over cluster 0 and 1 as violin plot
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=obs_col, y='phat_factor'+str(factor_idx+1), 
                   data=adata_sub.obs,
                   order=['control_0', 'control_1', 'case_0', 'case_1'])
    plt.title(f"Violin plot of p-hat scores (factor {factor_idx+1})")
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.xlabel("Cluster")
    plt.ylabel("p-hat")
    plt.show()


    
    num_sig_DE = {}
    ###############################################################################
    ### perform DE between case 0 and 1
    case_1_mask = (adata_sub.obs[obs_col] == 'case_1').values
    case_0_mask = (adata_sub.obs[obs_col] == 'case_0').values
    # Subset to the two clusters
    keep = case_1_mask | case_0_mask
    ad = adata_sub[keep].copy()
    # Temporary 2-level group label
    grp_col = "_tmp_de_group"
    ad.obs[grp_col] = pd.Categorical(
        np.where(case_1_mask[keep], "case1", "case0"),
        categories=["case0", "case1"]
    )
    # Wilcoxon DE: case1 vs case0
    sc.tl.rank_genes_groups(
        ad,
        groupby=grp_col,
        groups=["case1"],
        reference="case0",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,   
        layer=None,      
        n_genes=ad.n_vars
    )
    de_results = sc.get.rank_genes_groups_df(ad, group="case1").rename(columns={"names": "gene"})
    # score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
    gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
    de_results['gene_name'] = de_results['gene'].map(gene_names)
    print(de_results.head(20))
    num_sig_DE['case1_vs_case0'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
    print(f"Number of significant DE genes (case1 vs case0): {num_sig_DE['case1_vs_case0']}")

    ###############################################################################
    ### perform DE between cluster-1 and control+cluster-0 
    case_1_mask = (adata_sub.obs[obs_col] == 'case_1').values
    case_0_mask = (adata_sub.obs[obs_col] == 'case_0').values
    control_mask = (adata_sub.obs['Condition'].values != CASE_COND_NAME)
    # Subset to the two clusters + control
    keep = case_1_mask | case_0_mask | control_mask
    ad = adata_sub[keep].copy()
    # Temporary 2-level group label
    grp_col = "_tmp_de_group"
    ad.obs[grp_col] = pd.Categorical(
        np.where(case_1_mask[keep], "case1", "other"),
        categories=["other", "case1"]
    )
    # Wilcoxon DE: case1 vs other
    sc.tl.rank_genes_groups(
        ad,
        groupby=grp_col,
        groups=["case1"],
        reference="other",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,   
        layer=None,      
        n_genes=ad.n_vars
    )
    de_results = sc.get.rank_genes_groups_df(ad, group="case1").rename(columns={"names": "gene"})
    # score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
    gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
    de_results['gene_name'] = de_results['gene'].map(gene_names)
    print(de_results.head(20))
    num_sig_DE['case1_vs_other'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
    print(f"Number of significant DE genes (case1 vs other): {num_sig_DE['case1_vs_other']}")

    ###############################################################################
    ### perform DE between cluster-1 and control_1
    case_1_mask = (adata_sub.obs[obs_col] == 'case_1').values
    control_1_mask = (adata_sub.obs[obs_col] == 'control_1').values
    # Subset to the two clusters
    keep = case_1_mask | control_1_mask
    ad = adata_sub[keep].copy()
    # Temporary 2-level group label
    grp_col = "_tmp_de_group"
    ad.obs[grp_col] = pd.Categorical(
        np.where(case_1_mask[keep], "case1", "control1"),
        categories=["control1", "case1"]
    )
    # Wilcoxon DE: case1 vs control1
    sc.tl.rank_genes_groups(
        ad,
        groupby=grp_col,
        groups=["case1"],
        reference="control1",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,
        layer=None,
        n_genes=ad.n_vars
    )
    de_results = sc.get.rank_genes_groups_df(ad, group="case1").rename(columns={"names": "gene"})
    # score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
    gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
    de_results['gene_name'] = de_results['gene'].map(gene_names)
    print('-----------------------------------')
    print('Case1 vs Control1')
    print(de_results.head(20))  
    print('-----------------------------------')
    num_sig_DE['case1_vs_control1'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
    print(f"Number of significant DE genes (case1 vs control1): {num_sig_DE['case1_vs_control1']}")


    ###############################################################################
    ### perform DE between case-0 and all control
    case_0_mask = (adata_sub.obs[obs_col] == 'case_0').values
    control_mask = (adata_sub.obs['Condition'].values != CASE_COND_NAME)
    # Subset to the two clusters + control
    keep = case_0_mask | control_mask
    ad = adata_sub[keep].copy()
    # Temporary 2-level group label
    grp_col = "_tmp_de_group"
    ad.obs[grp_col] = pd.Categorical(
        np.where(case_0_mask[keep], "case0", "control"),
        categories=["control", "case0"]
    )
    # Wilcoxon DE: case0 vs control
    sc.tl.rank_genes_groups(
        ad,
        groupby=grp_col,
        groups=["case0"],
        reference="control",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,   
        layer=None,      
        n_genes=ad.n_vars
    )
    de_results = sc.get.rank_genes_groups_df(ad, group="case0").rename(columns={"names": "gene"})
    # score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
    gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
    de_results['gene_name'] = de_results['gene'].map(gene_names)
    print(de_results.head(20))
    num_sig_DE['case0_vs_control'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
    print(f"Number of significant DE genes (case0 vs control): {num_sig_DE['case0_vs_control']}")


    ###############################################################################
    ### perform DE between case-0 and all control_0
    case_0_mask = (adata_sub.obs[obs_col] == 'case_0').values
    control_0_mask = (adata_sub.obs[obs_col] == 'control_0').values
    # Subset to the two clusters + control
    keep = case_0_mask | control_0_mask
    ad = adata_sub[keep].copy()
    # Temporary 2-level group label
    grp_col = "_tmp_de_group"
    ad.obs[grp_col] = pd.Categorical(
        np.where(case_0_mask[keep], "case0", "control0"),
        categories=["control0", "case0"]
    )
    # Wilcoxon DE: case0 vs control0
    sc.tl.rank_genes_groups(
        ad,
        groupby=grp_col,
        groups=["case0"],
        reference="control0",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,   
        layer=None,      
        n_genes=ad.n_vars
    )
    de_results = sc.get.rank_genes_groups_df(ad, group="case0").rename(columns={"names": "gene"})
    # score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
    gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
    de_results['gene_name'] = de_results['gene'].map(gene_names)
    print(de_results.head(20))
    num_sig_DE['case0_vs_control0'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
    print(f"Number of significant DE genes (case0 vs control0): {num_sig_DE['case0_vs_control0']}")
    
    ### calculate residual: Y-phat
    #logit_phat = plot.safe_logit(adata.obs['phat_factor'+str(factor_idx+1)].values)
    #adata.obs['Condition_binary'] = (adata.obs['Condition'] == CASE_COND_NAME).astype(int)
    #Y = adata.obs['Condition_binary'].values
    #response_residual = Y - logit_phat
    #adata.obs['residual'+str(factor_idx+1)] = response_residual

    
    
    ### I added this mapping to resolve the color issue in plotting the clusters
    ###############################################################################
    # Map string labels to numeric codes (ensure no NaNs remain)
    code_map = {'control_0': 0, 'control_1': 1, 'case_0': 2, 'case_1': 3}
    obs_col_num = obs_col + "_num"
    adata_sub.obs[obs_col_num] = adata_sub.obs[obs_col].map(code_map).astype(float)
    # Sanity check: if you see NaN here, some labels didn't match code_map exactly
    print("Unique numeric codes:", adata_sub.obs[obs_col_num].unique())
    assert not pd.isna(adata_sub.obs[obs_col_num]).any(), "Unmapped labels → extend code_map."
    # Numeric palette (keys must match the codes you just wrote)
    palette_num = {
        0: "#54A24B",  # control_0
        1: "#E45756",  # control_1
        2: "#4C78A8",  # case_0
        3: "#F58518",  # case_1
    }   
    ###############################################################################

    sample_ids = adata_sub.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata_sub[adata_sub.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }

    plot.plot_grid_upgrade(
        adata_by_sample, sample_ids, key=obs_col_num,
        title_prefix=f"Clusters (factor {factor_idx+1})",
        from_obsm=False, discrete=True,
        dot_size=7, figsize=(25, 10),
        palette=palette_num
    )

    #plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=obs_col,
    #        title_prefix="Clusters (factor "+str(factor_idx+1)+")", 
    #        from_obsm=False, discrete=True,
    #        dot_size=2, figsize=(25, 10),
    #        palette={'case_0': "#4C78A8", 'case_1': "#F58518", 
    #                 'control_0': "#54A24B", 'control_1': "#E45756"})
    ##

    plot.plot_grid_upgrade(adata_by_sample, sample_ids, key='phat_factor'+str(factor_idx+1),
                           title_prefix="p-hat (factor "+str(factor_idx+1)+")", 
                           from_obsm=False, discrete=False,
                            dot_size=2, figsize=(25, 10))
    

    ### visualize number of significant DE genes in different comparisons
    comparisons = list(num_sig_DE.keys())
    sig_de_counts = [num_sig_DE[comp] for comp in comparisons]  
    plt.figure(figsize=(8, 5))
    sns.barplot(x=comparisons, y=sig_de_counts, palette="viridis")
    plt.title(f"Number of significant Genes (factor {factor_idx+1})")
    plt.ylabel("#sig DE genes")
    plt.xlabel("Comparisons")
    plt.xticks(rotation=45)
    plt.show()




