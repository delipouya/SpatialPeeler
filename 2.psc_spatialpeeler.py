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


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()


# file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG.h5ad'


adata_merged = sc.read_h5ad(file_name)

sample_ids = adata_merged.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in adata_merged.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())


sid=sample_ids[0]
plot.plot_spatial_nmf(adata_by_sample[sid], 3, sample_id=sid)

total_factors = adata_merged.obsm["X_nmf"].shape[1]
adata = adata_merged

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

#adata_merged.obsm["X_nmf"] 
#adata_merged.uns["nmf_components"]

adata = adata_merged.copy()
adata_merged.obsm["X_pca"] = adata_merged.obsm["X_nmf"]  # Use NMF factors as PCA scores

optimal_num_pcs_ks = total_factors
# Set up HiDDEN input
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['status'] = adata.obs['binary_label'].astype(int).values





# Run factor-wise HiDDEN-like analysis (logistic regression on single factors)
results = cpred.single_factor_logistic_evaluation(
    adata, factor_key="X_pca", max_factors=optimal_num_pcs_ks
)

# Run factor-wise HiDDEN-like analysis (logistic regression on single factors)
results_orig = cpred.single_factor_logistic_evaluation_original(
    adata, factor_key="X_pca", max_factors=optimal_num_pcs_ks
)


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
coef_df_sorted

### draw histogram of coefficients
plt.figure(figsize=(10, 6))
sns.histplot(coef_df['coef'], bins=30, kde=False, color='blue', stat='density')
plt.title("Distribution of Coefficients Across Factors")
plt.xlabel("Coefficient Value")
plt.ylabel("Density")
plt.axvline(x=200, color='red', linestyle='--', label='DAM - GOF')
plt.axvline(x=-200, color='green', linestyle='--', label='DAM - LOF')
plt.legend()
plt.show()


factor_id = 24
results[factor_id]['p_hat']  # p_hat for the first factor
results[factor_id]['scaled_p_hat']  # p_hat for the first factor
adata_merged.obs['disease']
### create a dataframe 
df_p_hat = pd.DataFrame({
    'disease': adata_merged.obs['disease'],
    'sample_id': adata_merged.obs['sample_id'],
    'p_hat': results[factor_id]['p_hat'],
    'scaled_p_hat': results[factor_id]['scaled_p_hat'],
    'pearson_residual': results[factor_id]['pearson_residual'],
    'raw_residual': results[factor_id]['raw_residual'],

})
plt.figure(figsize=(10, 10))
sns.violinplot(y="scaled_p_hat", x="disease",  
               data=df_p_hat, inner="box", palette="Set2")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


########################  VISUALIZATION  ########################
for i in range(14,optimal_num_pcs_ks): #optimal_num_pcs_ks
    plot.plot_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i)
    plot.plot_logit_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i)



################################################
################### Importing results from pickle and Anndata ##################
################################################
import pickle
# Save
#with open('/home/delaram/SpatialPeeler/Data/PSC_liver/results.pkl', 'wb') as f:
#    pickle.dump(results, f)
# Load
with open('/home/delaram/SpatialPeeler/Data/PSC_liver/results.pkl', 'rb') as f:
    results = pickle.load(f)

adata_merged = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad')
sample_ids = adata_merged.obs['sample_id'].unique().tolist()
adata = adata_merged.copy()
################################################

# Store output for the best-performing factor (e.g., first one, or pick based on AUC)
counter = 1
result = results[12] #24
for result in results[10:26]:
    print(f"Factor {result['factor_index'] + 1}:")
    print(f"  p_hat mean: {result['p_hat'].mean():.4f}")
    print(f"  Status distribution: {np.bincount(result['status'])}")

    adata.obs['p_hat'] = result['p_hat']
    adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
    adata.obs['scaled_p_hat'] = result['scaled_p_hat']
    adata.obs['scaled_z'] = result['scaled_z']
    adata.obs['random_construct'] = result['random_construct']
    adata.obs['raw_residual'] = result['raw_residual']
    adata.obs['pearson_residual'] = result['pearson_residual']
    adata.obs['deviance_residual'] = result['deviance_residual']
    

    # Copy adata per sample for plotting
    sample_ids = adata.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }
    # Plot spatial maps for the first 8 samples
    #for i in range(min(8, len(sample_ids))):
    #    plot_spatial_p_hat(adata_by_sample[sample_ids[i]], sample_ids[i])
    
    # For NMF factor  (from obsm)
    plot.plot_grid(adata_by_sample, sample_ids, key="X_nmf", 
    title_prefix="NMF Factor", counter=counter, from_obsm=True, factor_idx=3)
    # For p_hat
    plot.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix="HiDDEN predictions", counter=counter)

    # for scaled p_hat
    plot.plot_grid(adata_by_sample, sample_ids, key="scaled_p_hat",
    title_prefix="Scaled HiDDEN predictions", counter=counter)

    plot.plot_grid(adata_by_sample, sample_ids, key="scaled_z",
    title_prefix="Scaled Z", counter=counter)

    plot.plot_grid(adata_by_sample, sample_ids, key="random_construct",
    title_prefix="Random Construct", counter=counter)
    
    # For raw residuals
    plot.plot_grid(adata_by_sample, sample_ids, key="raw_residual", 
                   title_prefix="Raw Residual", counter=counter)
    # For Pearson residuals
    plot.plot_grid(adata_by_sample, sample_ids, key="pearson_residual", 
    title_prefix="Pearson Residual", counter=counter)


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



###### for all factors, plot p-hat vs nmf score as a scatter plot
for i in range(optimal_num_pcs_ks):
    nmf_scores = adata.obsm["X_nmf"][:, i]
    p_hat = results[i]['p_hat']  # p_hat for the i-th factor

    plt.figure(figsize=(8, 6))
    plt.scatter(nmf_scores, p_hat, alpha=0.5, s=10)
    plt.xlabel(f"NMF Factor {i+1} Score")
    plt.ylabel("HiDDEN p_hat")
    plt.title(f"p_hat vs NMF Factor {i+1} Score")
    plt.grid()
    plt.tight_layout()
    plt.show()


for i in range(optimal_num_pcs_ks):
    ### violin plot for nmf scores per sample
    nmf_scores = adata.obsm["X_nmf"][:, i]
    nmf_df = pd.DataFrame(nmf_scores,
                            columns=[f'NMF{i+1}'])
    nmf_df['sample_id'] = adata.obs['sample_id'].values
    nmf_long = nmf_df.melt(id_vars='sample_id',
                            var_name='Factor',
                            value_name='Score')
    plt.figure(figsize=(12, 5))
    sns.violinplot(x="sample_id", y="Score", hue="sample_id",
                     data=nmf_long, inner="box", palette="Set2")
    plt.title(f"Distribution of NMF Factor {i+1} Scores per Sample")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

