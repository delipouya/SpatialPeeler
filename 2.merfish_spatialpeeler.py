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

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)


outp = "/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/merged_adata_infected_sham_allnorm_NMF30.h5ad"
adata = sc.read_h5ad(outp)
adata.obs['Condition'].value_counts()
adata.obs['sample_id'].value_counts()
print(adata.obsm)
sample_ids = adata.obs['sample_id'].unique().tolist()

# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())


plot.plot_grid(adata_by_sample, sample_ids, key="nCount_MERSCOPE", 
               title_prefix="nCount_MERSCOPE", 
               from_obsm=False, figsize=(70, 50), fontsize=45, 
               dot_size=35) #figsize=(42, 30), fontsize=45

for i in range(0, 3): #len(sample_ids)
    sid = sample_ids[i]
    for factor_idx in range(0, 3):
        print(f"Plotting for sample: {sid}, factor: {factor_idx}")
        plot.plot_spatial_nmf(adata_by_sample[sid], factor_idx, sample_id=sid, figsize=(15, 10))

total_factors = adata.obsm["X_nmf"].shape[1]
num_factors = 20
nmf_factors = adata.obsm['X_nmf'][:, :num_factors]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])


nmf_df['sample_id'] = adata.obs['Condition'].values
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

### make the same plot but for disease


##################################################################
########### Running HiDDEN on the NMF ######################

#adata.obsm["X_nmf"] 
#adata.uns["nmf_components"]


optimal_num_pcs_ks = 30
print(f"Optimal number of PCs/KS: {optimal_num_pcs_ks}")
# Set up HiDDEN input
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['binary_label'] = adata.obs['Condition'].apply(lambda x: 1 if x == 'infected' else 0)
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


factor_id = 6#18
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




GOF_index = [6, 7, 29, 12, 23, 17]
LOF_index = [3, 9, 11, 21, 13, 4, 15, 10]

########################  VISUALIZATION  ########################
for i in GOF_index+LOF_index: #optimal_num_pcs_ks
    #plot.plot_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i, figsize=(18, 10))
    plot.plot_logit_p_hat_vs_nmf_by_sample(adata, results, sample_ids, 
                                           factor_idx=i, figsize=(18, 10))



################################################
################### Importing results from pickle and Anndata ##################
################################################
# Save
#with open('/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/results.pkl', 'wb') as f:
#    pickle.dump(results, f)

# Load
with open('/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/results.pkl', 'rb') as f:
    results = pickle.load(f)

outp = "/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/merged_adata_infected_sham_allnorm_NMF30.h5ad"
adata = sc.read_h5ad(outp)

sample_ids = adata.obs['sample_id'].unique().tolist()



GOF_index = [6, 7, 29, 12, 23, 17]
LOF_index = [3, 9, 11, 21, 13, 4, 15, 10]

print(GOF_index)
results_LOF = [results[i] for i in LOF_index]
results_GOF = [results[i] for i in GOF_index]


for result in results_LOF[1:len(results_LOF)]: #results_GOF
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
    title_prefix=f" Factor {result['factor_index'] + 1}- " + "NMF ", counter=factor_number, from_obsm=True, 
    factor_idx=factor_number-1, 
    figsize=(70, 50), fontsize=45, dot_size=30)
    #figsize=(43, 20), fontsize=45, dot_size=30) #figsize=(45, 15), fontsize=45
      
    # For p_hat #plot.
    plot.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix=f" Factor {result['factor_index'] + 1}- " + "HiDDEN predictions", counter=factor_number, 
    figsize=(70, 50), fontsize=45, dot_size=30)
    #figsize=(43, 20), fontsize=45, dot_size=30) #figsize=(42, 30), fontsize=45



for result in results_LOF[1:len(results_LOF)]: #results_GOF
    print(f"Factor {result['factor_index'] + 1}:")

    df_violin = adata.obs[["sample_id", "p_hat"]].copy()
    plt.figure(figsize=(5, 7))
    sns.violinplot(
        x="sample_id", y="p_hat", hue="sample_id", data=df_violin,
        palette="Set2", density_norm="width", inner=None, legend=False
    )
    sns.boxplot(
        x="sample_id", y="p_hat", data=df_violin,
        color="white", width=0.1, fliersize=0
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f" Factor {result['factor_index'] + 1}- " + "Distribution of p_hat" , fontsize=13)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    


######################################################
###### for all factors, plot p-hat vs nmf score as a scatter plot
for i in range(optimal_num_pcs_ks):
    nmf_scores = adata.obsm["X_nmf"][:, i]
    p_hat = results[i]['p_hat']  # p_hat for the i-th factor

    #nmf_scores = nmf_scores[adata.obs['cropped'] == True]
    #p_hat = p_hat[adata.obs['cropped'] == True]

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



import dataframe_image as dfi
################ Identifying the top genes associated with factors


GOF_index = [6, 7, 29, 12, 23, 17]
LOF_index = [3, 9, 11, 21, 13, 4, 15, 10]


NMF_H = pd.DataFrame(adata.uns["nmf_components"]).T
NMF_H.columns = [f'NMF{i+1}' for i in range(NMF_H.shape[1])]
NMF_H['genes'] = adata.var_names
print(NMF_H.shape)
print(NMF_H.head(10))

for index in GOF_index + LOF_index:
    factor_name = f'NMF{index + 1}'
    print(f"Top genes for {factor_name}:")
    ### make a df of the gene symbol and the factor name and sort it
    gene_df = pd.DataFrame({
    'genes': NMF_H['genes'],
    factor_name: NMF_H[factor_name]
    })
    gene_df.sort_values(by=factor_name, ascending=False, inplace=True)
    gene_df.reset_index(drop=True, inplace=True)
    # Apply styling to set the background color to white
    styled_df = gene_df.head(35).style.set_properties(**{'background-color': 'white'})
    styled_df = styled_df.set_properties(**{'color': 'black'})
    styled_df
    # Export the styled DataFrame to a PNG file
    #outpath = f"/home/delaram/SpatialPeeler/Plots/remyelin_t3_7_NMF{index+1}_genes_df.png"
    outpath = f"/home/delaram/SpatialPeeler/Plots/merfish/merfish_NMF{index+1}_genes_df.png"

    dfi.export(
        styled_df,
        outpath,
        table_conversion="matplotlib",  
        dpi=300
    )