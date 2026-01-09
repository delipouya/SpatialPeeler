import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import hiddensc
from hiddensc import utils, vis
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


from sklearn.cluster import KMeans

import statsmodels.api as sm
from scipy.special import logit
import pickle
RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()


# file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG_NMF10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_revLog_varScale_2000HVG_NMF10.h5ad'


adata = sc.read_h5ad(file_name)

sample_ids = adata.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())


sid=sample_ids[0]
plot.plot_spatial_nmf(adata_by_sample[sid], 3, sample_id=sid, figsize=(6, 5))



for i in range(len(sample_ids)): #len(sample_ids)
    sid = sample_ids[i]
    print(f"Plotting for sample: {sid}")
    plot.plot_spatial_nmf(adata_by_sample[sid], 0, sample_id=sid, figsize=(6, 5))


total_factors = adata.obsm["X_nmf"].shape[1]

num_factors = 10
nmf_factors = adata.obsm['X_nmf'][:, :num_factors]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])

nmf_df['sample_id'] = adata.obs['sample_id'].values
#nmf_df['sex'] = adata.obs['sex'].values
#nmf_df['donor_id'] = adata.obs['donor_id'].values
#nmf_df['disease'] = adata.obs['disease'].values

nmf_long = nmf_df.melt(id_vars='sample_id', 
                       var_name='Factor', 
                       value_name='Score')
plt.figure(figsize=(16, 6))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=nmf_long, 
               inner="box", palette="Set2")
plt.title("Distribution of NMF Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), 
           loc='upper left')
plt.tight_layout()
plt.show()

total_factors = adata.obsm["X_nmf"].shape[1]
nmf_factors = adata.obsm['X_nmf'][:, :8]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])
nmf_df['sample_id'] = adata.obs['sample_id'].values
nmf_long = nmf_df.melt(id_vars='sample_id', 
                       var_name='Factor', 
                       value_name='Score')


plt.figure(figsize=(16, 6))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=nmf_long, inner="box", palette="Set2")
plt.title("Distribution of NMF Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), 
           loc='upper left')
plt.tight_layout()
plt.show()


##################################################################
########### Running HiDDEN on the NMF ######################

#adata.obsm["X_nmf"] 
#adata.uns["nmf_components"]


optimal_num_pcs_ks = 30
print(f"Optimal number of PCs/KS: {optimal_num_pcs_ks}")
# Set up HiDDEN input
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['binary_label'] = adata.obs['Condition'].apply(lambda x: 1 if x == 'LPC' else 0)
adata.obs['status'] = adata.obs['binary_label'].astype(int).values



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



#def single_factor_logistic_evaluation_Fclust(adata, factor_key="X_nmf", max_factors=30):
factor_key = "X_nmf"
max_factors = 10
all_results = []
X = adata.obsm[factor_key]
y = adata.obs["disease"].values
sample_ids = adata.obs["sample_id"].values
i = 1

thresholding = 'zero'  # 'zero' or 'kmeans'
visualize_each_factor = False
exception_vis = False

for i in range(max_factors):#: #range(min(max_factors, X.shape[1])) ,3, 6, 19, range(max_factors)
    print(f"Evaluating factor {i+1}...")
    Xi = X[:, i].reshape(-1, 1)  # single factor
    #print("X i: ", Xi)
    #print(Xi.min(), Xi.max())

    # --- threshold-based filter instead of KMeans ---
    if thresholding == 'zero':
        threshold = 0.0
        high_mask = (Xi.ravel() > threshold)          # use >= if you want to include zeros
        high_cluster_indices = np.where(high_mask)[0]

        ## print the number of zeros and non-zeros
        num_high = np.sum(high_mask)
        num_low = len(high_mask) - num_high
        print(f"Number of samples with Xi > {threshold}: {num_high}")
        print(f"Number of samples with Xi <= {threshold}: {num_low}") 

        Xi_high = Xi[high_cluster_indices]
        y_high = y[high_cluster_indices]
        y_high = (np.asarray(y_high) == "LPC").astype(int)

        print(f"Using {len(high_cluster_indices)} samples with Xi > {threshold} for logistic regression")

        # Optional: sanity stats (rough analog of your min/max prints)
        if len(high_cluster_indices) == 0:
            print("Warning: no samples passed the threshold; skipping this factor.")
            continue  # or handle however you prefer

        low_mask = ~high_mask
        if low_mask.any():
            print("low (<=thr) min/max:", Xi.ravel()[low_mask].min(), Xi.ravel()[low_mask].max())
        print("high (>thr)  min/max:", Xi.ravel()[high_mask].min(), Xi.ravel()[high_mask].max())

        labels_remapped = high_mask.astype(int)


    # --- KMeans clustering to separate into two clusters ---
    elif thresholding == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
        kmeans.fit(Xi.reshape(-1, 1))
        centers = kmeans.cluster_centers_.ravel()
        print(centers)
        order = np.argsort(centers)               # [low_center_label, high_cluster_label]
        print(order)
        remap = {order[0]: 0, order[1]: 1}
        print(remap)
        labels_remapped = np.vectorize(remap.get)(kmeans.labels_).astype(int)
        
        
        min_cluster0 = Xi[labels_remapped == 0].min()
        max_cluster0 = Xi[labels_remapped == 0].max()
        min_cluster1 = Xi[labels_remapped == 1].min()
        max_cluster1 = Xi[labels_remapped == 1].max()

        print('original labels: ', kmeans.labels_)
        print("Labels remapped: ", labels_remapped)
        print('cluster-0 min/max:', min_cluster0, max_cluster0)
        print('cluster-1 min/max:', min_cluster1, max_cluster1)

        if min_cluster0 > min_cluster1 or max_cluster0 > max_cluster1:
            print("Error in clustering remapping!")
            break   

        # Calculate the threshold as the midpoint between the centroids
        threshold = np.mean(centers)
        #print(f"Centroids: {centers.flatten()}")
        #print(f"Binarization Threshold: {threshold}")

        ### only use Xi with high mean cluster for the logistic regression
        high_cluster_label = 1
        high_cluster_indices = np.where(labels_remapped == high_cluster_label)[0]
        Xi_high = Xi[high_cluster_indices]
        y_high = y[high_cluster_indices]
        y_high = (np.asarray(y_high) == 'LPC').astype(int)
        print(f"Using {len(high_cluster_indices)} samples from high-expression cluster for logistic regression")

    if visualize_each_factor:
        ### cluster the Xi into 2 clusters 
        plt.figure(figsize=(5, 5))
        sns.histplot(Xi, bins=30, kde=True)
        ### add vertical line for threshold
        plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
        plt.legend()
        plt.title(f"Factor {i+1}")
        plt.xlabel("Factor scores for all spots")
        plt.ylabel("Count")
        plt.show()

    
    ########################### VISUALIZATION ########################
    ## add the factor clustering results to adata object and vidualize all spatial samples
    adata.obs[f'Factor_{i+1}_cluster'] = labels_remapped
    adata.obs[f'Factor_{i+1}_score'] = Xi.astype('float32')


    sample_ids = adata.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }
    if visualize_each_factor:
        plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=f'Factor_{i+1}_cluster', 
               title_prefix=f"Factor {i+1} Clusters", 
               from_obsm=False, 
               figsize=(43, 20), fontsize=45, dot_size=60,
               palette_continuous='viridis') #figsize=(42, 30), fontsize=45 

        ### visualize the factor scores on spatial maps for each sample
        adata.obs[f'Factor_{i+1}_score'] = Xi.astype('float32')
        plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=f'Factor_{i+1}_score', 
               title_prefix=f" Factor {i+1} Scores", 
               from_obsm=False, figsize=(43, 30), fontsize=45,
                dot_size=60, palette_continuous='viridis_r') #figsize=(42, 30), fontsize=45
    #################################################################

    # Logistic regression with full inference
    p_hat, coef, stderr, pvals = standalone_logistic(Xi_high, y_high)
    logit_p_hat = logit(p_hat)

    # Save the results
    result = {
        "factor_index": i,
        "coef": float(coef[1]),                    # β (slope)
        "intercept": float(coef[0]),               # β0
        "std_err": float(stderr[1]),               # SE(β)
        "std_err_intercept": float(stderr[0]),     # SE(β0)
        "pval": float(pvals[1]),                   # p(β != 0)
        "pval_intercept": float(pvals[0]),         # p(β0 != 0)
        "p_hat": p_hat,
        "logit_p_hat": logit_p_hat,
        "status": y,
        "sample_id": sample_ids,
        'high_cluster_indices': high_cluster_indices,
        'Xi_high': Xi_high,
        'y_high': y_high
        }
    
    all_results.append(result)

    ## make a histogram of p_hat for the high-expression cluster
    if visualize_each_factor:
        plt.figure(figsize=(5, 5))
        sns.histplot(p_hat, bins=30, kde=True)
        plt.title(f"Factor {i+1} - p_hat Distribution (High-Exp Cluster)")
        plt.xlabel("p-hat for nmf-clust spots")
        plt.ylabel("Count")
        plt.show()
    
        ### visualize the p_hat distribution for all samples across conditions violin plot
        df_p_hat = pd.DataFrame({
            'disease': adata.obs['Condition'][high_cluster_indices],
            'sample_id': adata.obs['sample_id'][high_cluster_indices],
            'p_hat': p_hat
        })


        plt.figure(figsize=(10, 10))
        # Violin plot
        sns.violinplot(
            y="p_hat",
            x="disease",
            data=df_p_hat,
            inner="box",
            cut=0,
            density_norm="count",
            order=["Saline", "LPC"],
            palette={"Saline": "skyblue", "LPC": "salmon"}
        )

        if thresholding == 'kmeans':
            # Swarm plot (data points)
            sns.swarmplot(
                y="p_hat",
                x="disease",
                data=df_p_hat,
                order=["Saline", "LPC"],
                color="k",
                size=3,
                alpha=0.5,
                zorder=3
            )
        plt.title(f"Factor {i+1} - p_hat Distribution (High-Exp Cluster)")
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(15, 10))
        # Violin plot
        sns.violinplot(
            y="p_hat",
            x="sample_id",
            data=df_p_hat,
            inner="box",
            density_norm="count",
            cut=0,
            order=saline_samples + lpc_samples,
            palette={sid: "skyblue" for sid in saline_samples}
                    | {sid: "salmon" for sid in lpc_samples}
        )
        if thresholding == 'kmeans':
            # Swarm plot (data points)
            sns.swarmplot(
                y="p_hat",
                x="sample_id",
                data=df_p_hat,
                order=saline_samples + lpc_samples,
                color="k",
                size=5,
                alpha=0.5,
                zorder=3
            )
        plt.title(f"Factor {i+1} - p_hat Distribution (High-Exp Cluster)")
        plt.tight_layout()
        plt.show()


    if exception_vis:
        ### visualize p-hat values on spatial maps for each sample, for non-high-expression cluster points p_hat=NA
        adata.obs['p_hat'] = pd.NA
        adata.obs['p_hat'].iloc[high_cluster_indices] = p_hat.astype('float32')
        #adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
        # Copy adata per sample for plotting
        sample_ids = adata.obs['sample_id'].unique().tolist()
        adata_by_sample = {
            sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
            for sample_id in sample_ids
        }
        # Plot spatial maps for the first 8 samples
        plot.plot_grid_upgrade(adata_by_sample, sample_ids, key="p_hat", from_obsm=False, 
        title_prefix=f" Factor {i+1}- " + "p-hat predictions (high-exp cluster)", counter=i+1, 
        figsize=(43, 20), fontsize=45, dot_size=60) #figsize=(42, 30), fontsize=45

    
results = all_results
