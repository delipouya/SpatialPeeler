
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
from scipy.sparse import issparse
from matplotlib_venn import venn3

from SpatialPeeler import helpers as hlps
from SpatialPeeler import plotting as plot
from SpatialPeeler import gene_identification as gid
import pickle
from gprofiler import GProfiler
import pickle
import dataframe_image as dfi


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()

################################################
################### Importing results from pickle and Anndata ##################
################################################
with open('/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/results.pkl', 'rb') as f:
    results = pickle.load(f)

outp = "/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/merged_adata_infected_sham_allnorm_NMF30.h5ad"
adata = sc.read_h5ad(outp)

sample_ids = adata.obs['sample_id'].unique().tolist()


### select columns Condition and sample_id and make a unique table of two columns
cond_sample_df = adata.obs[['Condition', 'sample_id']].drop_duplicates().reset_index(drop=True)
cond_sample_df['Condition'] = cond_sample_df['Condition'].astype(str)
cond_sample_df['sample_id'] = cond_sample_df['sample_id'].astype(str)
cond_sample_df['Condition'] = cond_sample_df['Condition'].str.replace(' ', '_')
#############################################
########### Gene-Based analysis #############
#############################################

GOF_index = [6, 7, 29, 12, 23, 17]
LOF_index = [3, 9, 11, 21, 13, 4, 15, 10]

################################################

i = 0
PATTERN_COND = 'GOF'#'LOF'  # 'GOF' or 
factor_idx = GOF_index[i]

if PATTERN_COND == 'GOF':
    print("Using GOF pattern")
    factor_idx = GOF_index[i]
    print(f"Factor index for GOF: {factor_idx}")
else:
    print("Using LOF pattern")
    factor_idx = LOF_index[i]
    print(f"Factor index for LOF: {factor_idx}")


result = results[factor_idx] 
adata.obs['p_hat'] = result['p_hat']
adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
adata.obs['1_p_hat'] = 1 - adata.obs['p_hat']


sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}

plot.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix="HiDDEN predictions", counter=factor_idx+1, 
    figsize=(70, 50), fontsize=45, dot_size=30)
    #figsize=(43, 20), fontsize=45) #figsize=(45, 33) (43, 15)

plot.plot_grid(adata_by_sample, sample_ids, key="1_p_hat", 
    title_prefix="HiDDEN predictions", counter=factor_idx+1, 
    figsize=(70, 50), fontsize=45, dot_size=30)
    #figsize=(43, 20), fontsize=45) #figsize=(45, 33),


### scatter plot of p_hat vs NMF
plt.figure(figsize=(8, 8))
for sample_id in sample_ids:
    an_adata_sample = adata_by_sample[sample_id]
    plt.scatter(an_adata_sample.obsm["X_nmf"][:, factor_idx], 
                an_adata_sample.obs['p_hat'], 
                alpha=0.5, label=sample_id, s=10)
plt.xlabel("NMF Factor")
plt.ylabel("p_hat")
plt.title("NMF Factor vs p_hat")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()  


#factor_idx = GOF_index[2]
PATTERN_COND = 'LOF'#'LOF'  # 'GOF' or 

for factor_idx in LOF_index:
    factor_name = f'NMF{factor_idx + 1}'
    print(f"Top genes for p-hat {factor_name}:")

    if PATTERN_COND == 'GOF':
        print("Using GOF pattern")
        print(f"Factor index for GOF: {factor_idx}")
    else:
        print("Using LOF pattern")
        print(f"Factor index for LOF: {factor_idx}")

    result = results[factor_idx] 
    adata.obs['p_hat'] = result['p_hat']
    adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
    adata.obs['1_p_hat'] = 1 - adata.obs['p_hat']

    #sample_id_to_check = 2#1#12#6
    #an_adata_sample = adata_by_sample[sample_ids[sample_id_to_check]]
    an_adata_sample = adata
    expr_matrix = an_adata_sample.X.toarray() if issparse(an_adata_sample.X) else an_adata_sample.X  # shape: (n_spots, n_genes)
    p_hat_vector = an_adata_sample.obs['p_hat']  # shape: (n_spots,)
    neg_p_hat_vector = an_adata_sample.obs['1_p_hat']  # shape: (n_spots,)
    pattern_vector = p_hat_vector if PATTERN_COND == 'GOF' else neg_p_hat_vector

    #### removing genes with zero variance
    gene_zero_std_index = np.std(expr_matrix, axis=0) == 0
    expr_matrix_sub = expr_matrix[:, ~gene_zero_std_index]  # Exclude genes with zero variance
    gene_names = an_adata_sample.var_names[~gene_zero_std_index]
    pearson_corr = gid.pearson_correlation_with_pattern(expr_matrix_sub, pattern_vector, 
                                                        gene_names=gene_names)

    pearson_corr = pearson_corr.sort_values("correlation", ascending=False)
    pearson_corr = pearson_corr.reset_index(drop=True)
    pearson_corr = pearson_corr[['gene', 'correlation']]
    pearson_corr.columns = ['Gene', 'phat_F'+str(factor_idx+1)]

    # Apply styling to set the background color to white
    styled_df = pearson_corr.head(35).style.set_properties(**{'background-color': 'white'})
    styled_df = styled_df.set_properties(**{'color': 'black'})
    styled_df
    # Export the styled DataFrame to a PNG file
    outpath = f"/home/delaram/SpatialPeeler/Plots/merfish/merfish_phat_{factor_idx+1}_genes_df.png"
    
    dfi.export(
        styled_df,
        outpath,
        table_conversion="matplotlib",  
        dpi=300
    )


#0:10 are diseased samples, 11:14 are normal samples 
sample_id_to_check = 3#1#12#6
an_adata_sample = adata_by_sample[sample_ids[sample_id_to_check]]


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

NMF_idx_values = an_adata_sample.obsm["X_nmf"][:,factor_idx]
weighted_pearson_corr = gid.weighted_pearson_correlation_with_pattern(expr_matrix_sub, pattern_vector, 
                                                                      W_vector=NMF_idx_values, 
                                                                      gene_names=gene_names, 
                                                                      scale=True)


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
    'weighted_pearson': weighted_pearson_corr
}

key = 'Pearson'
cor_df = corr_dict[key]

for key, cor_df in corr_dict.items():
    cor_df = cor_df.sort_values("correlation", ascending=False)    
    cor_df = cor_df.reset_index(drop=True)
    print(f"Top 10 genes for {key}:")
    print(cor_df[['gene', 'correlation']].head(20).to_string(index=False))
    corr_dict[key] = cor_df


num_genes_viz = 15
top_genes = {}
for key, cor_df in corr_dict.items():
    top_genes[key] = cor_df.sort_values("correlation", ascending=False).head(num_genes_viz)[['gene','symbol', 'correlation']]
top_genes

print(top_genes['Pearson'])


num_genes_viz = 4
for key, df in top_genes.items():
    for i in range(num_genes_viz):
        gene_symbol = df['symbol'].values[i]
        print(f"Top {i+1} gene for {key}: {gene_symbol}")
        # Plot the spatial distribution of the top genes
        plot.plot_gene_spatial(an_adata_sample, df['gene'].values[i], 
                             title=f"{key} - {gene_symbol}", cmap="viridis", figsize=(11, 9))


i = 0 ## number of top genes to visualize
key = 'Regression'  # or 'Pearson' or 'weighted_pearson'
#4:7 are PSC samples, 0:3 are normal samples - initial analysis on sample #5, 4
for sample_id_to_check in range(len(sample_ids)):
    an_adata_sample_2 = adata_by_sample[sample_ids[sample_id_to_check]]
    print(f"Sample {sample_ids[sample_id_to_check]}:")
    df = top_genes[key]
    gene_symbol = df['symbol'].values[i]
    print(f"Top {i+1} gene for {key}: {gene_symbol}")
    plot.plot_gene_spatial(an_adata_sample_2, df['gene'].values[i], 
                            title=f"{sample_ids[sample_id_to_check]} - {key} - {gene_symbol}", cmap="viridis",
                            figsize=(13, 11)) #figsize=(10, 8) (11, 9)

x_axis = 'weighted_pearson'#'weighted_pearson'
y_axis = 'Pearson' 
df_vis = pd.merge(corr_dict[x_axis], corr_dict[y_axis], on='symbol', how='inner')
plt.figure(figsize=(8, 8))
plt.scatter(df_vis["correlation_x"], 
            df_vis["correlation_y"], alpha=0.4, s=10)
plt.xlabel(x_axis + " Cor")
plt.ylabel(y_axis + " Cor")
### add a diagonal line
plt.plot([-1, 1], [-1, 1], color='red', linestyle='--', linewidth=1)
plt.title(f"{x_axis} vs {y_axis} Correlation")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.show()
