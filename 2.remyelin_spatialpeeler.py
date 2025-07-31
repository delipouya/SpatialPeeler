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

import pickle

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()

#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_SampleWiseNorm.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18.h5ad'

adata = sc.read_h5ad(file_name)

### cropped data
adata.obs['Condition'].value_counts() 
# LPC       28582
# Saline     9522
adata.obs['Animal'].value_counts()
#['M140','M261','M257','M255','M259','EG102','EG98','EG100','M254','M258','EG103','EG99']
adata.obs['Timepoint'].value_counts() # [12, 18, 3, 7]

adata.obs['sample_id'] = adata.obs['puck_id']
#adata.obs['sample_id'] = adata.obs['orig.ident']
spatial = {'x': adata.obs['x'].values.astype(float).tolist(), 
           'y': adata.obs['y'].values.astype(float).tolist()}
#adata.obsm["spatial"] = pd.DataFrame(spatial, index=adata.obs.index).values


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
metadata_df

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
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=nmf_long, inner="box", palette="Set2")
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


optimal_num_pcs_ks = total_factors
print(f"Optimal number of PCs/KS: {optimal_num_pcs_ks}")
# Set up HiDDEN input
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['binary_label'] = adata.obs['Condition'].apply(lambda x: 1 if x == 'LPC' else 0)
adata.obs['status'] = adata.obs['binary_label'].astype(int).values



# Run factor-wise HiDDEN-like analysis (logistic regression on single factors)
results = cpred.single_factor_logistic_evaluation(
    adata, factor_key="X_nmf", max_factors=optimal_num_pcs_ks
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
print(coef_df_sorted)

### draw histogram of coefficients
plt.figure(figsize=(10, 6))
sns.histplot(coef_df['coef'], bins=30, kde=False, color='blue', stat='density')
plt.title("Distribution of Coefficients Across Factors")
plt.xlabel("Coefficient Value")
plt.ylabel("Density")
plt.legend()
plt.show()


factor_id = 18
results[factor_id]['p_hat']  # p_hat for the first factor
adata.obs['Condition']
### create a dataframe 
df_p_hat = pd.DataFrame({
    'disease': adata.obs['Condition'],
    'sample_id': adata.obs['sample_id'],
    'p_hat': results[factor_id]['p_hat']

})
plt.figure(figsize=(10, 10))
sns.violinplot(y="p_hat", x="disease",  
               data=df_p_hat, inner="box", palette="Set2")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


########################  VISUALIZATION  ######################## #(10,20)
for i in range(0,optimal_num_pcs_ks): #optimal_num_pcs_ks
    plot.plot_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i, figsize=(16, 10)) #(18, 6)
    #plot.plot_logit_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i)



################################################
################### Importing results from pickle and Anndata ##################
################################################
# Save
#results_path = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/results_Remyelin.pkl'
#results_path = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/results_Remyelin_uncropped.pkl'
#results_path = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/results_Remyelin_uncropped_SampleWiseNorm.pkl'
results_path = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/results_Remyelin_uncropped_t3_7.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
# Load


with open(results_path, 'rb') as f:
    results = pickle.load(f)
#adata = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad')
#adata = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped.h5ad')
sample_ids = adata.obs['sample_id'].unique().tolist()



################################################
### cropped indices
GOF_index = [1, 14, 12, 22, 26, 0]
LOF_index = [2, 6, 13]

#### uncropped indices
GOF_index = [24, 20, 3, 2, 27, 17, 13, 9]
LOF_index = [11, 7, 23, 26, 29, 6, 12, 22]

#### uncropped - sample-wise norm indices
GOF_index = [0, 2, 5, 12, 6]
LOF_index = [16, 8, 29, 13]

#### uncropped - t3_7 indices
GOF_index = [21, 1, 9, 22, 24]
LOF_index = [0, 27, 12, 5, 19]


print(GOF_index)
results_LOF = [results[i] for i in LOF_index]
results_GOF = [results[i] for i in GOF_index]


for result in results:
    print(f"Factor {result['factor_index'] + 1}:")
    factor_number = result['factor_index'] + 1
    print(f"  p_hat mean: {result['p_hat'].mean():.4f}")
    print(f"  Status distribution: {np.bincount(result['status'])}")

### creat a heatmap of p_hat for each factor for correlation (30x30)
p_hat_matrix = np.array([res['p_hat'] for res in results])
p_hat_df = pd.DataFrame(p_hat_matrix.T,
                        columns=[f'Factor_{i+1}' for i in range(len(results))],
                        index=[f'Sample_{i+1}' for i in range(p_hat_matrix.shape[1])])

plt.figure(figsize=(30, 25))
sns.heatmap(p_hat_df.corr(), annot=True, cmap='coolwarm',
            xticklabels=p_hat_df.columns, yticklabels=p_hat_df.columns, 
            ### increase font size
            annot_kws={"size": 20}, fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap of p_hat Across Factors")
plt.tight_layout()
plt.show()

### creat a heatmap of NMF for each factor for correlation (30x30)
nmf_matrix = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
nmf_df = pd.DataFrame(nmf_matrix,
                        columns=[f'NMF{i+1}' for i in range(nmf_matrix.shape[1])],
                        index=[f'Sample_{i+1}' for i in range(nmf_matrix.shape[0])])
plt.figure(figsize=(30, 25))
sns.heatmap(nmf_df.corr(), annot=True, cmap='coolwarm',
            xticklabels=nmf_df.columns, yticklabels=nmf_df.columns,
            annot_kws={"size": 20}, fmt=".2f", linewidths=.5
            )
plt.title("Correlation Heatmap of NMF Scores Across Samples")
plt.tight_layout()
plt.show()

for i in range(optimal_num_pcs_ks):
    p_hat_i = results[i]['p_hat']
    NMF_i = adata.obsm["X_nmf"][:, i]

    ### plot p_hat vs NMF score for the first factor
    plt.figure(figsize=(8, 6))
    plt.scatter(NMF_i, p_hat_i, alpha=0.5, s=10)
    plt.xlabel(f"NMF Factor {i+1} Score")
    plt.ylabel("HiDDEN p_hat")
    ### set phat range to 0-1
    plt.ylim(0, 1)
    plt.title(f"p_hat vs NMF Factor {i+1} Score")
    plt.grid()
    plt.tight_layout()
    plt.show()

### create a heatmap correlation between p_hat and NMF scores for each factor
### 30x30 - NMF scores x p_hat heatmap matrix
n_factors = nmf_matrix.shape[1]
correlation_matrix = np.corrcoef(nmf_matrix.T, p_hat_matrix.T)[:n_factors, n_factors:]  # shape: (30, 30)
corr_df = pd.DataFrame(correlation_matrix,
                       index=[f'p_hat_{i+1}' for i in range(n_factors)],
                       columns=[f'NMF{i+1}' for i in range(n_factors)])

plt.figure(figsize=(20, 12))
sns.heatmap(corr_df, annot=True, cmap='coolwarm',
            annot_kws={"size": 8}, fmt=".2f", linewidths=.5)
plt.title("Correlation Between NMF Factors (x) and p_hat Factors (y)")
plt.xlabel("NMF Factors")
plt.ylabel("p_hat Factors")
plt.tight_layout()
plt.show()

for result in results_GOF: #results_GOF
    print(f"Factor {result['factor_index'] + 1}:")
    factor_number = result['factor_index'] + 1
    print(f"  p_hat mean: {result['p_hat'].mean():.4f}")
    print(f"  Status distribution: {np.bincount(result['status'])}")

    adata.obs['p_hat'] = result['p_hat']
    adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')

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
    title_prefix="NMF Factor", counter=factor_number, from_obsm=True, 
    factor_idx=factor_number-1, figsize=(43, 15), fontsize=45) #figsize=(45, 33), fontsize=45
    
    # For p_hat #plot.
    plot.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix="HiDDEN predictions", counter=factor_number, 
    figsize=(43, 15), fontsize=45) #figsize=(42, 30), fontsize=45


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

