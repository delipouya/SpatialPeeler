import os
import sys
from sklearn.cluster import KMeans
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
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

import statsmodels.api as sm
from scipy.special import logit
import pickle

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)

#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7_PreprocV2.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7_PreprocV2_samplewise.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t7_PreprocV2_samplewise.h5ad'


#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18_K10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t7.h5ad'

adata_cropped = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad')

adata = sc.read_h5ad(file_name)
### cropped data
adata.obs['Condition'].value_counts() 
# LPC       28582
# Saline     9522
adata.obs['Animal'].value_counts()
#['M140','M261','M257','M255','M259','EG102','EG98','EG100','M254','M258','EG103','EG99']
adata.obs['Timepoint'].value_counts() # [12, 18, 3, 7]

adata.obs['sample_id'] = adata.obs['puck_id'] #orig.ident
#adata.obs['sample_id'] = adata.obs['orig.ident']
spatial = {'x': adata.obs['x'].values.astype(float).tolist(), 
           'y': adata.obs['y'].values.astype(float).tolist()}
#adata.obsm["spatial"] = pd.DataFrame(spatial, index=adata.obs.index).values

adata.obs['cropped'] = adata.obs['barcode'].isin(adata_cropped.obs['bead'])


sample_ids = adata.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())

metadata = {'sample_id': adata.obs['sample_id'].values.tolist(),
            'Timepoint': adata.obs['Timepoint'].values.tolist(),
            'Animal': adata.obs['Animal'].values.tolist(),
            'Condition': adata.obs['Condition'].values.tolist()}
metadata_df = pd.DataFrame(metadata, index=adata.obs.index)
metadata_df = metadata_df.reset_index(drop=True)
duplicate_rows_mask = metadata_df.duplicated()
metadata_df = metadata_df[~duplicate_rows_mask]
print(metadata_df)
saline_samples = metadata_df[metadata_df['Condition'] == 'Saline']['sample_id'].tolist()
lpc_samples = metadata_df[metadata_df['Condition'] == 'LPC']['sample_id'].tolist()
print("Saline samples: ", saline_samples)
print("LPC samples: ", lpc_samples)

# Plot spatial maps for the first 8 samples
#for i in range(min(8, len(sample_ids))):
#    plot_spatial_p_hat(adata_by_sample[sample_ids[i]], sample_ids[i])

plot.plot_grid_upgrade(adata_by_sample, sample_ids, key="cropped", 
               title_prefix="Cropped regions", 
               from_obsm=False, figsize=(43, 30), fontsize=45, 
               dot_size=50) #figsize=(42, 30), fontsize=45

for i in range(0, 3): #len(sample_ids)
    sid = sample_ids[i]
    print(f"Plotting for sample: {sid}")
    plot.plot_spatial_nmf(adata_by_sample[sid], 0, sample_id=sid, figsize=(10, 10))
    plot.plot_spatial_nmf(adata_by_sample[sid], 1, sample_id=sid, figsize=(10, 10))
    plot.plot_spatial_nmf(adata_by_sample[sid], 2, sample_id=sid, figsize=(10, 10))

total_factors = adata.obsm["X_nmf"].shape[1]
num_factors = 10
nmf_factors = adata.obsm['X_nmf'][:, :num_factors]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])

nmf_df['sample_id'] = adata.obs['Animal'].values
nmf_df['sample_id'] = adata.obs['Timepoint'].values
nmf_df['sample_id'] = adata.obs['sample_id'].values
nmf_df['sample_id'] = adata.obs['Condition'].values

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

### make the same plot but for disease


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
max_factors = 30
all_results = []
X = adata.obsm[factor_key]
y = adata.obs["Condition"].values
sample_ids = adata.obs["sample_id"].values
i = 1

t3_7_gof = [9, 21, 18, 11, 1, 2, 5, 23]
t3_7_gof_v2 = [3, 4, 16, 8, 0, 1, 17, 15]

t7_gof_v2 = [14, 28, 11, 6, 0, 18, 21]


t18_gof = [9, 2, 12, 18]
t18_lof = [3, 6, 19]
t7_gof = [0, 12, 20, 2]
t7_lof = [10, 18, 7]
i = t3_7_gof_v2[0]
t3_gof_control12_18 = [8, 13, 16, 28, 0, 4, 21, 2, 11]


thresholding = 'zero'  # 'zero' or 'kmeans'
visualize_each_factor = True
exception_vis = True

for i in t7_gof_v2:#: #range(min(max_factors, X.shape[1])) ,3, 6, 19, range(max_factors)
    print(f"Evaluating factor {i+1}...")
    Xi = X[:, i].reshape(-1, 1)  # single factor
    #print("X i: ", Xi)
    #print(Xi.min(), Xi.max())

    # --- threshold-based filter instead of KMeans ---
    if thresholding == 'zero':
        threshold = 0.0
        high_mask = (Xi.ravel() > threshold)          # use >= if you want to include zeros
        high_cluster_indices = np.where(high_mask)[0]

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
               from_obsm=False, figsize=(43, 30), fontsize=45,
                dot_size=60, palette_continuous='viridis_r') #figsize=(42, 30), fontsize=45 

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

#results_filename = 'remyelin_nmf30_hidden_logistic_Fclust_t3_7_PreprocV2.pkl'
#results_filename = 'remyelin_nmf30_hidden_logistic_t3_7_PreprocV2_noFclust.pkl'
#results_filename = 'remyelin_nmf30_hidden_logistic_zeroThr_t3_gof_control12_18.pkl'
#results_filename = 'remyelin_nmf30_hidden_logistic_Fclust_t3_7_PreprocV2.pkl'

results_filename = 'remyelin_nmf30_hidden_logistic_zeroThr_Fclust_t7_PreprocV2.pkl'

### save the results using pickle
with open(results_filename, 'wb') as f:
    pickle.dump(results, f)

# Load
with open(results_filename, 'rb') as f:
    results = pickle.load(f)


### count the number of spots included high_cluster_indices for each factor within each sample_id
factor_sample_counts = []
for res in results:
    high_cluster_indices = res['high_cluster_indices']
    sample_ids_high = adata.obs['sample_id'].iloc[high_cluster_indices].values
    sample_counts = pd.Series(sample_ids_high).value_counts().to_dict()
    factor_sample_counts.append({
        'factor_index': res['factor_index'],
        'sample_counts': sample_counts
    })

for i in t3_gof_control12_18:
    ### create a barplot for the counts of spots in high-expression cluster for each sample_id
    factor_count_dict = factor_sample_counts[i]['sample_counts']
    samples = list(factor_count_dict.keys())
    counts = list(factor_count_dict.values())
    plt.figure(figsize=(6, 6))
    sns.barplot(x=samples, y=counts)
    plt.title(f"Factor {i+1} - High-Exp Cluster Spot Counts perSample", fontsize=16)
    plt.xlabel("Sample ID", fontsize=16)
    plt.ylabel("#Spots in high-Exp Cluster", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

# Extract full model stats for each factor
coef_list = [res['coef'] for res in results]
intercept_list = [res['intercept'] for res in results]
stderr_list = [res.get('std_err', None) for res in results]
stderr_intercept_list = [res.get('std_err_intercept', None) for res in results]
pval_list = [res.get('pval', None) for res in results]
pval_intercept_list = [res.get('pval_intercept', None) for res in results]
factor_index_list = [res['factor_index'] for res in results]

# Build summary DataFrame
coef_df = pd.DataFrame({
    'factor': [f'Factor_{i+1}' for i in factor_index_list],
    'coef': coef_list,
    'intercept': intercept_list,
    'std_err': stderr_list,
    'std_err_intercept': stderr_intercept_list,
    'pval': pval_list,
    'pval_intercept': pval_intercept_list
})
coef_df_sorted = coef_df.sort_values(by='coef', ascending=False)
print(coef_df_sorted)

### draw histogram of coefficients
plt.figure(figsize=(10, 6))
sns.histplot(coef_df['coef'], bins=30, kde=False, color='blue', stat='density')
plt.title("Distribution of Coefficients Across Factors")
plt.xlabel("Coefficient Value")
plt.ylabel("Density")
plt.legend()
plt.show()

sample_ids = adata.obs['sample_id'].unique().tolist()


### we have 0-110088 indices and a subset of them are used for each factor high-expression cluster
## extract the indices for each factor, and count how many times each index is used across all factors
index_usage = np.zeros(adata.n_obs, dtype=int)
for res in results:
    high_cluster_indices = res['high_cluster_indices']
    index_usage[high_cluster_indices] += 1  
print("Index usage counts across all factors:")
print(index_usage) ## length = total number of spots (110088)

### make a histogram of index usage, use counts not density
plt.figure(figsize=(10, 6))
sns.histplot(index_usage, bins=30, kde=False, color='green', stat='count')
plt.title("Histogram of Spot Index Usage Across All Factors")
plt.xlabel("Number of Factors Using Spot Index")
plt.ylabel("Count of Spot Indices")
plt.legend()
plt.show()


factor_index = 0  # Change this to the desired factor index (0-based)
PATTERN_COND = 'GOF'#'GOF'
sample_id_to_check = 1#1#12#6

t3_7_gof_v2 = [3, 4, 16, 8, 0, 1, 17, 15]
t3_7_gof = [9, 21, 18, 11, 1, 2, 5, 23]
t18_gof = [9, 2, 12, 18]
t18_lof = [3, 6, 19]
t7_gof = [0, 12, 20, 2]
t7_lof = [10, 18, 7]
#### uncropped - t3_7 indices - PreprocV2 without clustering factors
t3_7_gof_v2_noFclust = [4, 13, 16, 3, 17, 29, 26, 0, 23]

results_path = 'remyelin_nmf30_hidden_logistic_zeroThr_t3_gof_control12_18.pkl'
with open(results_path, 'rb') as f:
    results = pickle.load(f)

for factor_index in t3_gof_control12_18:
    if 'high_cluster_indices' in results[factor_index]:
        print(f"Factor {factor_index+1} - Number of spots in high-expression cluster: {len(results[factor_index]['high_cluster_indices'])}")
        adata_sub = adata[results[factor_index]['high_cluster_indices'], :].copy()
    else:
        adata_sub = adata.copy()
    print(adata_sub.shape)
    adata_sub.obs['p_hat'] = results[factor_index]['p_hat'].astype('float32')
    adata_sub.obs['1_p_hat'] = 1 - adata_sub.obs['p_hat'].astype('float32')
    adata_sub_by_sample = {
        sid: adata_sub[adata_sub.obs['sample_id'] == sid].copy()
        for sid in adata_sub.obs['sample_id'].unique()
    }
    sample_ids = list(adata_sub_by_sample.keys())
    
    #0:10 are diseased samples, 11:14 are normal samples 
    
    an_adata_sample = adata_sub_by_sample[sample_ids[sample_id_to_check]]
    expr_matrix = an_adata_sample.X.toarray() if issparse(an_adata_sample.X) else an_adata_sample.X  # shape: (n_spots, n_genes)
    p_hat_vector = an_adata_sample.obs['p_hat']  # shape: (n_spots,)

    neg_p_hat_vector = an_adata_sample.obs['1_p_hat']  # shape: (n_spots,)
    pattern_vector = p_hat_vector if PATTERN_COND == 'GOF' else neg_p_hat_vector

    #### removing genes with zero variance
    print(np.all(np.isfinite(expr_matrix)))
    gene_zero_std_index = np.std(expr_matrix, axis=0) == 0
    print(expr_matrix.shape)
    expr_matrix_sub = expr_matrix[:, ~gene_zero_std_index]  # Exclude genes with zero variance
    print(expr_matrix_sub.shape)
    print(np.var(expr_matrix[:, gene_zero_std_index], axis=0)  )
    print(np.var(expr_matrix_sub, axis=0)  )

    gene_names = an_adata_sample.var_names[~gene_zero_std_index]
    pearson_corr = gid.pearson_correlation_with_pattern(expr_matrix_sub, pattern_vector, 
                                                        gene_names=gene_names)
    symbols= pearson_corr['gene'].map(hlps.map_ensembl_to_symbol(pearson_corr['gene'].tolist(), species='mouse'))
    pearson_corr['symbols'] = symbols
    ### sort the pearson correlation dataframe
    pearson_corr.sort_values("correlation", ascending=False, inplace=True)
    print(pearson_corr.head(30))


    #NMF_idx_values = an_adata_sample.obsm["X_nmf"][:,factor_index]
    ### NMF gene df
    NMF_gene_df = pd.DataFrame({
        "gene": an_adata_sample.var_names,
        #"NMF_loading": an_adata_sample.uns['nmf_components'][factor_index,:]
        "NMF_loading": an_adata_sample.uns['nmf']['H'][factor_index,:]
    })

    #### merge the NMF_gene_df with pearson_corr based on gene column
    merged_df_genescores = pd.merge(pearson_corr, NMF_gene_df, on="gene", how="inner")
    ### plot NMF_loading vs correlation scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df_genescores['NMF_loading'], merged_df_genescores['correlation'])
    plt.xlabel('NMF Loading')
    plt.ylabel('Pearson Correlation')
    plt.title('Factor ' + str(factor_index+1) + ': NMF Loading vs Pearson Correlation')
    plt.show()
    

    from adjustText import adjust_text
    plt.figure(figsize=(10, 6))
    plt.scatter(
        merged_df_genescores['NMF_loading'],
        merged_df_genescores['correlation'],
        alpha=0.7, edgecolor='k'
    )
    # Select top 5 genes based on NMF_loading
    top5_cor = merged_df_genescores.nlargest(5, "correlation")
    top5_nmf = merged_df_genescores.nlargest(5, "NMF_loading")
    top5 = pd.concat([top5_cor, top5_nmf]).drop_duplicates().reset_index(drop=True)
    # Highlight top 5
    plt.scatter(top5['NMF_loading'], top5['correlation'], color='red', s=90)
    texts = []
    for _, row in top5.iterrows():
        texts.append(
            plt.text(
                row['NMF_loading'], row['correlation'],
                row['symbols'],
                fontsize=16, weight='bold'
            )
        )
    # Automatically adjust label positions to avoid overlaps
    adjust_text(
        texts,
        expand_points=(2, 2),   # push away from points
        arrowprops=dict(arrowstyle='-', lw=1, color='black', alpha=0.6)
    )
    plt.xlabel('NMF Loading', fontsize=16)
    plt.ylabel('Pearson Correlation', fontsize=16)
    plt.title(f'Factor {factor_index+1}: NMF Loading vs Pearson Correlation', fontsize=18)
    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(10, 6))
    plt.scatter(
        merged_df_genescores['NMF_loading'],
        merged_df_genescores['correlation'],
        alpha=0.7, edgecolor='k'
    )
    # Select top 5 genes based on NMF_loading
    top5 = merged_df_genescores.nlargest(5, "correlation")
    # Highlight top 5
    plt.scatter(top5['NMF_loading'], top5['correlation'], color='red', s=90)
    texts = []
    for _, row in top5.iterrows():
        texts.append(
            plt.text(
                row['NMF_loading'], row['correlation'],
                row['symbols'],
                fontsize=16, weight='bold'
            )
        )
    # Automatically adjust label positions to avoid overlaps
    adjust_text(
        texts,
        expand_points=(2, 2),   # push away from points
        arrowprops=dict(arrowstyle='-', lw=1, color='black', alpha=0.6)
    )
    plt.xscale("log")  # compress extreme loadings
    plt.xlabel('NMF Loading', fontsize=16)
    plt.ylabel('Pearson Correlation', fontsize=16)
    plt.title(f'Factor {factor_index+1}: NMF Loading vs Pearson Correlation', fontsize=18)
    plt.tight_layout()
    plt.show()


    

    



'''
regression_res = gid.regression_with_pattern(expr_matrix_sub, pattern_vector,
                                               gene_names=gene_names, 
                                               scale=True)
regression_corr = pd.DataFrame({
        "gene": regression_res["gene"],
        "correlation": regression_res["slope"]})
regression_corr.sort_values("correlation", ascending=False, inplace=True)

### make a histogram of the regression coefficients
plt.figure(figsize=(10, 6))
sns.histplot(regression_corr['correlation'], bins=30, 
             kde=False, color='blue', stat='density')
plt.title("Distribution of Regression Coefficients")
plt.xlabel("Regression Coefficient Value")
plt.ylabel("Density")
plt.legend()
plt.show()

corr_dict = {
    "Pearson": pearson_corr,
    "Regression": regression_corr,
}
'''


