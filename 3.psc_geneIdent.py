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
from matplotlib_venn import venn3

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


adata_merged = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad')
sample_ids = adata_merged.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in adata_merged.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())
adata = adata_merged

with open('/home/delaram/SpatialPeeler/Data/PSC_liver/results.pkl', 'rb') as f:
    results = pickle.load(f)
################################################

########################################################################
######################## Gene-Based analysis
########################################################################
factor_idx = 12#8 #12

result = results[factor_idx] 
adata.obs['p_hat'] = result['p_hat']
adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
adata.obs['1_p_hat'] = 1 - adata.obs['p_hat']

# Copy adata per sample for plotting
sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}

print(f"Factor {result['factor_index'] + 1}:")
### sanity check
plot.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix="HiDDEN predictions", counter=factor_idx+1)

#4:7 are PSC samples, 0:3 are normal samples - initial analysis on sample #5, 4
sample_id_to_check = 5
#sample_id_to_check = 1
an_adata_sample = adata_by_sample[sample_ids[sample_id_to_check]]
# Compute spatial weights using squidpy
# Get expression matrix and coordinates

#### calculating spatial weights using squidpy
import squidpy as sq
sq.gr.spatial_neighbors(an_adata_sample, coord_type="generic", n_neighs=10) #### how many neighbors to use?
W1 = an_adata_sample.obsp["spatial_connectivities"]
W1_array = np.array(W1.todense())  # if W1 is sparse, convert to dense matrix

##### calculating spatial weights using custom function
coords = an_adata_sample.obsm['spatial']  # shape: (n_spots, 2)
W2_geary = gid.compute_spatial_weights(coords, k=10, mode="geary")
W2_moran = gid.compute_spatial_weights(coords, k=10, mode="moran")

print('squidpy-W mean: ', W1_array.mean(),
       'squidpy-W sd: ', W1_array.std())
print('Geary-W mean: ', W2_geary.mean(),
         'Geary-W sd: ', W2_geary.std())
print('Moran-W mean: ', W2_moran.mean(),
            'Moran-W sd: ', W2_moran.std())

expr_matrix = an_adata_sample.X.toarray() if issparse(an_adata_sample.X) else an_adata_sample.X  # shape: (n_spots, n_genes)
p_hat_vector = an_adata_sample.obs['p_hat']  # shape: (n_spots,)

spatial_corr_1 = gid.spatial_weighted_correlation_matrix(expr_matrix, p_hat_vector, 
                                                        W1, an_adata_sample.var_names)
spatial_corr_2 = gid.spatial_weighted_correlation_matrix(expr_matrix, p_hat_vector, 
                                                        W2_geary, an_adata_sample.var_names)
spatial_corr_3 = gid.spatial_weighted_correlation_matrix(expr_matrix, p_hat_vector, 
                                                        W2_moran, an_adata_sample.var_names)
pearson_corr = gid.pearson_correlation_with_pattern(expr_matrix, p_hat_vector, 
                                                     gene_names=an_adata_sample.var_names)

regression_res = gid.regression_with_pattern(expr_matrix, p_hat_vector,
                                               gene_names=an_adata_sample.var_names, 
                                               scale=True)
regression_corr = pd.DataFrame({
        "gene": regression_res["gene"],
        "correlation": regression_res["slope"]})

regression_corr.sort_values("correlation", ascending=False, inplace=True)
print(regression_corr.head(20))
raw_reg_thr = regression_corr['correlation'].head(200).head(1).values[0]

### make a histogram of the regression coefficients
plt.figure(figsize=(10, 6))
sns.histplot(regression_corr['correlation'], bins=30, 
             kde=False, color='blue', stat='density')
plt.title("Distribution of Regression Coefficients")
plt.xlabel("Regression Coefficient Value")
plt.ylabel("Density")
plt.legend()
plt.show()


# Example usage on real data:
import scipy.sparse
coords = an_adata_sample.obsm["spatial"]  # shape (n_cells, 2)
gene_names = an_adata_sample.var_names.to_numpy()
gp_results = gid.fit_gp_similarity_scores(expr_matrix, pattern_vector=p_hat_vector, coords, gene_names)

import time
import multiprocessing
n_cores = multiprocessing.cpu_count()
n_jobs = max(1, int(n_cores * 0.5))  # Use ~50% of logical cores
 #(O(n³) time and O(n²) memory per job)
start_time = time.time() ### TODO: add printing progress to the function later
gp_results = gid.fit_gp_similarity_scores_fastmode(expr_matrix, 
                                                  p_hat_vector, coords, gene_names, 
                                                  n_jobs=n_jobs, hyperopt=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total runtime: {elapsed_time/60:.2f} minutes")
### save GP results to a DataFrame csv file
gp_results_df = pd.DataFrame({
    "gene": gp_results["gene"],
    "gp_similarity": gp_results["similarity_to_residual_GP"],
})
gp_results_df.sort_values("gp_similarity", ascending=True, inplace=True)
#gp_results_df.to_csv(os.path.join(root_path, 'SpatialPeeler',
#                             'data_PSC', 'gp_similarity_scores.csv'), index=False)

gp_regression_corr = pd.DataFrame({
    "gene": gp_results["gene"],
    "correlation": gp_results["similarity_to_residual_GP"]
})
gp_regression_corr.sort_values("correlation", ascending=True, inplace=True)

gp_regression_corr['ensemble_id'] = gp_regression_corr['gene']
gp_regression_corr['symbol'] = gp_regression_corr['gene'].map(
    hlps.map_ensembl_to_symbol(gp_regression_corr['gene'].tolist()))



### make a histogram of the regression coefficients
plt.figure(figsize=(10, 6))
sns.histplot(gp_regression_corr['correlation'], bins=30, 
             kde=False, color='blue', stat='density')
plt.title("Distribution of Regression Coefficients")
plt.xlabel("GP Regression Correlation Value")
plt.ylabel("Density")
plt.axvline(x=0, color='red', linestyle='--', label='Zero Line')
plt.legend()
plt.show()




corr_dict = {
    "Spatial_W1": spatial_corr_1,
    "Spatial_W2_Geary": spatial_corr_2,
    "Spatial_W2_Moran": spatial_corr_3,
    "Pearson": pearson_corr,
    "Regression": regression_corr,
    #"GP_Regression": gp_regression_corr
}

for key, cor_df in corr_dict.items():
    # Ensure 'gene' column is present
    if 'gene' not in cor_df.columns:
        cor_df['gene'] = an_adata_sample.var_names
    
    cor_df = cor_df.sort_values("correlation", ascending=False)
    id_to_symbol = hlps.map_ensembl_to_symbol(cor_df['gene'].tolist())
    cor_df['symbol'] = cor_df['gene'].map(id_to_symbol)
    cor_df['ensemble_id'] = cor_df['gene']  # Keep the original gene ID
    print(f"Top 10 genes for {key}:")
    ### print top 10 genes
    print(cor_df[['symbol', 'correlation']].head(20).to_string(index=False))
    ### replace the item in the dictionary with the updated DataFrame
    corr_dict[key] = cor_df


num_genes_viz = 10
top_genes = {}
for key, cor_df in corr_dict.items():
    top_genes[key] = cor_df.sort_values("correlation", ascending=False).head(num_genes_viz)[['ensemble_id','symbol', 'correlation']]
top_genes


for key, df in top_genes.items():
    for i in range(num_genes_viz):
        gene_symbol = df['symbol'].values[i]
        print(f"Top {i+1} gene for {key}: {gene_symbol}")
        # Plot the spatial distribution of the top genes
        plot.plot_gene_spatial(an_adata_sample, df['ensemble_id'].values[i], 
                             title=f"{key} - {gene_symbol}", cmap="viridis")

# Calculate mean and standard deviation of correlations for each method
for name, df in corr_dict.items():
    cor_vals = df["correlation"].dropna()
    mean_val = cor_vals.mean()
    std_val = cor_vals.std()
    print(f"{name}: mean = {mean_val:.4f}, std = {std_val:.4f}")
    print(f"{name}: max = {cor_vals.max():.4f}, min = {cor_vals.min():.4f}")



x_axis = 'Regression'
y_axis = 'Pearson' #'Spatial_W1' # 'Spatial_W2_Moran' 'Pearson' 'Regression'
df_vis = pd.merge(corr_dict[x_axis], corr_dict[y_axis], on='symbol', how='inner')
plt.figure(figsize=(5, 5))
plt.scatter(df_vis["correlation_x"], 
            df_vis["correlation_y"], alpha=0.4, s=10)
plt.xlabel(x_axis + " Cor")
plt.ylabel(y_axis + " Cor")
plt.legend()
plt.show()



### scatter plot of Pearson vs Regression Correlation
pearson_series = corr_dict["Pearson"]["correlation"]
regression_series = regression_res["slope"]
## sort by value of reg and find the 200 value 
n_gene_pathway_thr = 600
r_pathway_thr_pearson = pearson_series[:n_gene_pathway_thr].tail(1)
r_pathway_thr_regression = raw_reg_thr



n_gene_pathway_thr = 20
method_name = 'Regression'  # or 'Spatial_W2_Geary', 'Spatial_W2_Moran' 'Pearson' Spatial_W1
spatial_top_genes = corr_dict[method_name]['ensemble_id'][:n_gene_pathway_thr]

# Calculate average expression of the top genes across all samples
average_expression = {}
for sample_id, adata_sample in adata_by_sample.items():
    expr_matrix = adata_sample.X.toarray() if sp.issparse(adata_sample.X) else adata_sample.X
    top_gene_indices = [adata_sample.var_names.get_loc(gene) for gene in spatial_top_genes]
    avg_expr = expr_matrix[:, top_gene_indices].mean(axis=0)
    average_expression[sample_id] = avg_expr
average_expression_df = pd.DataFrame(average_expression,
                                        index=spatial_top_genes).T
average_expression_df = average_expression_df.T
print(average_expression_df.head())
gene_ensemble_map = hlps.map_ensembl_to_symbol(average_expression_df.index.tolist())
average_expression_df['symbol'] = average_expression_df.index.map(gene_ensemble_map)
### show as heatmap

# Set up the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(average_expression_df.drop(columns='symbol'),
            ## fix color map to be more readable: 0-4
            vmin=0, vmax=4,  # Adjust these limits based on your data
            cmap='viridis', annot=True, fmt=".02f",
            cbar_kws={'label': 'Average Expression'},
            xticklabels=average_expression_df.columns,
            ### make font size smaller
            yticklabels=average_expression_df['symbol'].values,
            linewidths=.5, linecolor='black')
plt.title("Average Expression of Top Spatial " + method_name +
          " Genes Across Samples")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.tight_layout()
plt.show()


# Create a Venn diagram
n_gene_venn = 50
genes_pearson = set(corr_dict['Pearson']["gene"][:n_gene_venn])
genes_w1 = set(corr_dict["Spatial_W1"]["gene"][:n_gene_venn])
genes_w2_geary = set(corr_dict["Spatial_W2_Geary"]["gene"][:n_gene_venn])
genes_w2_moran = set(corr_dict["Spatial_W2_Moran"]["gene"][:n_gene_venn])
genes_regression = set(corr_dict["Regression"]["gene"][:n_gene_venn])

# Plotting two Venn diagrams due to 3-set limitation
venn3([genes_pearson, genes_w1, genes_w2_geary],
    set_labels=("Pearson", "Spatial_W1", "Spatial_W2_Geary"))
venn3([genes_regression, genes_w1, genes_w2_moran],
    set_labels=("regression", "Spatial_W1", "Spatial_W2_Moran"))



#############################################
######### Enrichment analysis using g:Profiler
from gprofiler import GProfiler

method = 'Spatial_W1'#'Regression'  # or 'Spatial_W2_Geary', 'Spatial_W2_Moran', 'Pearson', 'Spatial_W1'
cor_df = corr_dict[method]  # or any other method you want to use
num_pathway_genes = 300
top_genes = cor_df.dropna(subset=['symbol']).sort_values("correlation", ascending=True).head(num_pathway_genes)["symbol"].tolist()
# Run enrichment
gp = GProfiler(return_dataframe=True)
enrich_results = gp.profile(organism='hsapiens', query=top_genes)
# View top results
enrich_results[['source', 'name', 'p_value', 'intersection_size']].head(10)
# Filter GO:BP terms only
go_bp_results = enrich_results[enrich_results['source'] == 'GO:BP']
# Show top 10 by p-value
print(go_bp_results[['name', 'p_value', 'intersection_size']].sort_values('p_value').head(20))
#############################################

categories = go_bp_results['name'].head(10).tolist()
values = -np.log10(go_bp_results['p_value'].head(10).tolist())

plt.bar(categories, values, color='skyblue')
plt.xlabel('top enriched GO:BP categories')
plt.ylabel('-log10(p_value)')
plt.xticks(rotation=45, ha='right') 
plt.title(method+ '-based Enrichment Analysis - Top 10 GO:BP Categories')
plt.show()