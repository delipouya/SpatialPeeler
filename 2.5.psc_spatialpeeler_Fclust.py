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

CONDITION_TAG = 'condition' # 'disease'
CONTROL_TAG = 'CONTROL' #'normal'
CASE_TAG = 'PSC' #'primary sclerosing cholangitis'

# file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG_NMF10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_revLog_varScale_2000HVG_NMF10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_10_varScale_2000HVG_filtered.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG_filtered_RAW_COUNTS.h5ad'

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
adata.obs['binary_label'] = adata.obs[CONDITION_TAG].apply(lambda x: 1 if x == CASE_TAG else 0)
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


### for the normalized datasets
normal_samples = [sid for sid in sample_ids if 'normal' in sid.lower()]
psc_samples = [sid for sid in sample_ids if 'psc' in sid.lower()]
#### for the raw counts dataset
normal_samples = [sid for sid in sample_ids if 'C73' in sid.upper()]
psc_samples = [sid for sid in sample_ids if 'PSC' in sid.upper()]


#def single_factor_logistic_evaluation_Fclust(adata, factor_key="X_nmf", max_factors=30):
factor_key = "X_nmf"
max_factors = 30
all_results = []
X = adata.obsm[factor_key]
y = adata.obs[CONDITION_TAG].values
sample_ids = adata.obs["sample_id"].values
i = 1

thresholding = 'none'  # 'zero' or 'kmeans', 'none'
visualize_each_factor = True
exception_vis = True

#gof_indices = [26, 19, 29, 11, 20, 12, 18, 0]
#lof_indices = [5, 22, 3, 1, 26]

gof_indices = [6, 1, 11, 7, 8, 2, 16, 18, 13, 22]
for i in gof_indices:#: #range(min(max_factors, X.shape[1])) ,3, 6, 19, range(max_factors)
    print(f"Evaluating factor {i+1}...")
    Xi = X[:, i].reshape(-1, 1)  # single factor
    #print("X i: ", Xi)
    #print(Xi.min(), Xi.max())

    if thresholding == 'none':
        Xi_high = Xi
        y_high = y
        y_high = (np.asarray(y_high) == CASE_TAG).astype(int)
        print(f"Using all {len(Xi_high)} samples for logistic regression")
        labels_remapped = np.zeros(Xi.shape[0], dtype=int)  # all zeros, no clustering
        high_cluster_indices = np.arange(Xi.shape[0])

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
        y_high = (np.asarray(y_high) == CASE_TAG).astype(int)

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
        y_high = (np.asarray(y_high) == CASE_TAG).astype(int)
        print(f"Using {len(high_cluster_indices)} samples from high-expression cluster for logistic regression")

    if visualize_each_factor:
        ### cluster the Xi into 2 clusters 
        plt.figure(figsize=(5, 5))
        sns.histplot(Xi, bins=30, kde=True)
        ### add vertical line for threshold
        if thresholding != 'none':
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
        if thresholding != 'none':
            ### visualize the factor clusters on spatial maps for each sample
            plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=f'Factor_{i+1}_cluster', 
                title_prefix=f"Factor {i+1} Clusters", 
                from_obsm=False, 
                figsize=(43, 20), fontsize=45, dot_size=60,
                palette_continuous='viridis_r') #figsize=(42, 30), fontsize=45 

        ### visualize the factor scores on spatial maps for each sample
        adata.obs[f'Factor_{i+1}_score'] = Xi.astype('float32')
        plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=f'Factor_{i+1}_score', 
               title_prefix=f" Factor {i+1} Scores", 
               from_obsm=False, figsize=(43, 20), fontsize=45,
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
            'disease': adata.obs[CONDITION_TAG][high_cluster_indices],
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
            order=[CONTROL_TAG, CASE_TAG],
            palette={CONTROL_TAG: "skyblue", CASE_TAG: "salmon"}
        )

        if thresholding == 'kmeans':
            # Swarm plot (data points)
            sns.swarmplot(
                y="p_hat",
                x="disease",
                data=df_p_hat,
                order=[CONTROL_TAG, CASE_TAG],
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
            order=normal_samples + psc_samples,
            palette={sid: "skyblue" for sid in normal_samples}
                    | {sid: "salmon" for sid in psc_samples}
        )
        if thresholding == 'kmeans':
            # Swarm plot (data points)
            sns.swarmplot(
                y="p_hat",
                x="sample_id",
                data=df_p_hat,
                order=normal_samples + psc_samples,
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

#results_filename = 'PSC_NMF_10_varScale_2000HVG_filtered_results_factorwise.pkl'

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

for i in range(len(factor_sample_counts)):
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





################################ clustering the p-hat values ##########################

#results_filename = 'PSC_NMF_10_varScale_2000HVG_filtered_results_factorwise.pkl'
# Load
with open(results_filename, 'rb') as f:
    results = pickle.load(f)


cluster_mask_dict = {}

gof_indices = [1, 6, 5]
lof_indices = [0, 4, 2]
factor_idx = 0
print(f"Factor {factor_idx+1}")


def get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1):
        sig_de = de_results[
            (de_results['pvals_adj'] < fdr_threshold) &
            (np.abs(de_results['logfoldchanges']) > logfc_threshold)
        ]
        return sig_de.shape[0]


CASE_COND_NAME = 'primary sclerosing cholangitis'
factor_idx = gof_indices[0]

for factor_idx in lof_indices: #range(min(max_factors, X.shape[1])) ,3, 6, 19,
    print(f"Factor {factor_idx+1}")
    result = results[factor_idx]
    p_hat_factor = result['p_hat']
    if 'high_cluster_indices' in result:
        high_cluster_indices = result['high_cluster_indices']
    else:
        high_cluster_indices = np.arange(adata.n_obs)
    
    adata_sub = adata[high_cluster_indices].copy()

    p_hat_factor_case = p_hat_factor[adata_sub.obs['disease'] == CASE_COND_NAME]
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
    mask_case = (adata_sub.obs['disease'].values == CASE_COND_NAME)
    

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

    # Violin plot (lighter + count-scaled)
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        x=obs_col,
        y=f'phat_factor{factor_idx+1}',
        data=adata_sub.obs,
        order=['control', 'case_0', 'case_1'],
        inner="box",
        cut=0,
        density_norm="count",
        color="skyblue",
        alpha=0.4,
        linewidth=1
    )
    # Swarm plot (data points)
    '''
    sns.swarmplot(
        x=obs_col,
        y=f'phat_factor{factor_idx+1}',
        data=adata_sub.obs,
        order=['control', 'case_0', 'case_1'],
        color="k",
        size=3,
        alpha=0.5,
        zorder=3
    )
    '''
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title(f"Violin plot of p-hat scores (factor {factor_idx+1})", fontsize=19)
    plt.xlabel("Cluster", fontsize=18)
    plt.ylabel("p-hat", fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
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

    # Violin plot (lighter + count-scaled)
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x=obs_col,
        y=f'phat_factor{factor_idx+1}',
        data=adata_sub.obs,
        order=['control_0', 'control_1', 'case_0', 'case_1'],
        inner="box",
        cut=0,
        density_norm="count",
        color="skyblue",
        alpha=0.4,
        linewidth=1
    )
    '''
    # Swarm plot (data points)
    sns.swarmplot(
        x=obs_col,
        y=f'phat_factor{factor_idx+1}',
        data=adata_sub.obs,
        order=['control_0', 'control_1', 'case_0', 'case_1'],
        color="k",
        size=3,
        alpha=0.5,
        zorder=3
    )
    '''
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title(f"Violin plot of p-hat scores (factor {factor_idx+1})", fontsize=20)
    plt.xlabel("Cluster", fontsize=18)
    plt.ylabel("p-hat", fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

    
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
    ### put NA for the spots that are. not phat_Cluster_factorX == 'case_1' to only visualize case_1 spots
    adata_sub.obs['phat_factor'+str(factor_idx+1)+'_case1_only'] = np.nan
    adata_sub.obs['phat_factor'+str(factor_idx+1)+'_case1_only'] = adata_sub.obs.apply(
        lambda row: row['phat_factor'+str(factor_idx+1)] if row[obs_col] == 'case_1' else np.nan,
        axis=1
    )   
    
    adata_by_sample = {
        sample_id: adata_sub[adata_sub.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }

    plot.plot_grid_upgrade(
        adata_by_sample, sample_ids, key=obs_col_num,
        title_prefix=f"Clusters (factor {factor_idx+1})",
        from_obsm=False, discrete=True,
        figsize=(43, 20), fontsize=45, dot_size=60,
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
                           figsize=(43, 20), fontsize=45, dot_size=60)

    
    plot.plot_grid_upgrade(adata_by_sample, sample_ids, 
                           key='phat_factor'+str(factor_idx+1)+'_case1_only',
                           title_prefix="p-hat Case1 only (factor "+str(factor_idx+1)+")", 
                           from_obsm=False, discrete=False,
                            figsize=(43, 20), fontsize=45, dot_size=60)
    
    


