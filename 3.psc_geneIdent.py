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
adata.obs['p_hat_comp'] = 1 - adata.obs['p_hat']

# Copy adata per sample for plotting
sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}

print(f"Factor {result['factor_index'] + 1}:")
### sanity check
plot_grid(adata_by_sample, sample_ids, key="p_hat", 
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

spatial_corr_1 = gid.spatial_weighted_correlation_matrix(expr_matrix, residual_vector, 
                                                        W1, an_adata_sample.var_names)
spatial_corr_2 = gid.spatial_weighted_correlation_matrix(expr_matrix, residual_vector, 
                                                        W2_geary, an_adata_sample.var_names)
spatial_corr_3 = gid.spatial_weighted_correlation_matrix(expr_matrix, residual_vector, 
                                                        W2_moran, an_adata_sample.var_names)
pearson_corr = gid.pearson_correlation_with_residuals(expr_matrix, residual_vector, 
                                                     gene_names=an_adata_sample.var_names)

regression_res = gid.regression_with_residuals(expr_matrix, residual_vector,
                                               gene_names=an_adata_sample.var_names)

regression_res = gid.regression_with_residuals(expr_matrix, residual_vector,
                                               gene_names=an_adata_sample.var_names, scale=True)
regression_corr = pd.DataFrame({
        "gene": regression_res["gene"],
        "correlation": regression_res["slope"]})

regression_corr.sort_values("correlation", ascending=True, inplace=True)
raw_reg_thr = regression_corr['correlation'].head(200).tail(1).values[0]

### make a histogram of the regression coefficients
plt.figure(figsize=(10, 6))
sns.histplot(regression_corr['correlation'], bins=30, 
             kde=False, color='blue', stat='density')
plt.title("Distribution of Regression Coefficients")
plt.xlabel("Regression Coefficient Value")
plt.ylabel("Density")
plt.axvline(x=0, color='red', linestyle='--', label='Zero Line')
plt.legend()
plt.show()


# Example usage on real data:
import scipy.sparse
coords = an_adata_sample.obsm["spatial"]  # shape (n_cells, 2)
gene_names = an_adata_sample.var_names.to_numpy()

gp_results = gid.fit_gp_similarity_scores(expr_matrix, residual_vector, coords, gene_names)

import time
import multiprocessing
n_cores = multiprocessing.cpu_count()
n_jobs = max(1, int(n_cores * 0.5))  # Use ~50% of logical cores
 #(O(n³) time and O(n²) memory per job)

start_time = time.time() ### TODO: add printing progress to the function later
gp_results = gid.fit_gp_similarity_scores_fastmode(expr_matrix, 
                                                  residual_vector, coords, gene_names, 
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
    "GP_Regression": gp_regression_corr
}

for key, cor_df in corr_dict.items():
    # Ensure 'gene' column is present
    if 'gene' not in cor_df.columns:
        cor_df['gene'] = an_adata_sample.var_names
    
    cor_df = cor_df.sort_values("correlation", ascending=True)
    id_to_symbol = fn.map_ensembl_to_symbol(cor_df['gene'].tolist())
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
    top_genes[key] = cor_df.sort_values("correlation", ascending=True).head(num_genes_viz)[['ensemble_id','symbol', 'correlation']]
top_genes


for key, df in top_genes.items():
    for i in range(num_genes_viz):
        gene_symbol = df['symbol'].values[i]
        print(f"Top {i+1} gene for {key}: {gene_symbol}")
        # Plot the spatial distribution of the top genes
        fn.plot_gene_spatial(an_adata_sample, df['ensemble_id'].values[i], 
                             title=f"{key} - {gene_symbol}", cmap="viridis")

# Calculate mean and standard deviation of correlations for each method
for name, df in corr_dict.items():
    cor_vals = df["correlation"].dropna()
    mean_val = cor_vals.mean()
    std_val = cor_vals.std()
    print(f"{name}: mean = {mean_val:.4f}, std = {std_val:.4f}")
    print(f"{name}: max = {cor_vals.max():.4f}, min = {cor_vals.min():.4f}")


##### plot scatter plots of Regression vs GP_regression Correlations
regression_series = corr_dict["Regression"]["correlation"]

pearson_series = corr_dict["Pearson"]["correlation"]
gp_series = corr_dict["GP_Regression"]["correlation"]

### check if correlation values of the two methods are equivalent
if not regression_series.index.equals(gp_series.index):
    print("Warning: Regression and GP Regression indices do not match. Aligning on gene index.")

# Align on gene index just in case
x = regression_series #
y = gp_series
# Set up the plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.4, s=10)
plt.xlabel("regression Correlation")
plt.ylabel("GP Regression Correlation")
plt.tight_layout()
plt.legend()
plt.show()



##### plot scatter plots of Pearson vs Spatially Weighted Correlations
pearson_series = corr_dict["Pearson"]["correlation"]
n_gene_pathway_thr = 200
r_pathway_thr_pearson = pearson_series[:n_gene_pathway_thr].tail(1)

# Spatial_W2_Geary Spatial_W2_Moran Spatial_W1
##### plot scatter plots of Pearson vs Spatially Weighted Correlations
spatial_series = corr_dict["Spatial_W2_Geary"]["correlation"]
n_gene_pathway_thr = 200
r_pathway_thr_spatial = spatial_series[:n_gene_pathway_thr].tail(1)

# Set up the plot
fig, axs = plt.subplots(1, 3, figsize=(22, 7), sharex=True, sharey=True)
spatial_methods = ["Spatial_W1","Spatial_W2_Geary", "Spatial_W2_Moran"]
for i, method in enumerate(spatial_methods):

    spatial_series = corr_dict[method]["correlation"]
    
    # Align on gene index just in case
    common_genes = pearson_series.index.intersection(spatial_series.index)
    x = pearson_series.loc[common_genes]
    y = spatial_series.loc[common_genes]

    ### add a vertical line at the threshold
    axs[i].axvline(r_pathway_thr_pearson.values[0], color='red', 
                   linestyle='--', label='Pearson Pathway Threshold')
    #### add a horizontal line at the spatial threshold
    axs[i].axhline(r_pathway_thr_spatial.values[0], color='blue', 
                   linestyle='--', label='Spatial Pathway Threshold')
    # Focus on genes with negative Pearson correlation
    mask = x < 0
    x = x[mask]
    y = y[mask]
    
    axs[i].scatter(x, y, alpha=0.4, s=10)
    axs[i].set_title(f"{method} vs Pearson")
    axs[i].set_xlabel("Pearson Correlation")
    axs[i].set_ylabel("Spatial Correlation")

plt.suptitle("Pearson vs Spatially Weighted Correlations", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

### scatter plot of Pearson vs Regression Correlation
pearson_series = corr_dict["Pearson"]["correlation"]
regression_series = regression_res["slope"]
## sort by value of reg and find the 200 value 
n_gene_pathway_thr = 200
r_pathway_thr_pearson = pearson_series[:n_gene_pathway_thr].tail(1)
r_pathway_thr_regression = raw_reg_thr

# Align on gene index just in case
common_genes = pearson_series.index.intersection(regression_series.index)
x = pearson_series.loc[common_genes]
y = regression_series.loc[common_genes] 
# Set up the plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.4, s=10)
plt.axvline(r_pathway_thr_pearson.values[0], color='red',
            linestyle='--', label='Pearson')
plt.axhline(r_pathway_thr_regression, color='blue',
            linestyle='--', label='Regression')
plt.title("Pearson vs Regression Correlation")
plt.xlabel("Pearson Correlation")
plt.ylabel("Regression Correlation")
plt.legend()
plt.tight_layout()
plt.show()


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
gene_ensemble_map = fn.map_ensembl_to_symbol(average_expression_df.index.tolist())
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
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

# First Venn Diagram: Pearson, W1, W2_Geary
venn3([genes_pearson, genes_w1, genes_w2_geary],
    set_labels=("Pearson", "Spatial_W1", "Spatial_W2_Geary"),
    ax=axes[0])
axes[0].set_title("Pearson vs W1 vs W2_Geary")

# Second Venn Diagram: Pearson, W2_Geary, W2_Moran
venn3([genes_pearson, genes_w2_geary, genes_w2_moran],
    set_labels=("Pearson", "Spatial_W2_Geary", "Spatial_W2_Moran"),
    ax=axes[1])
axes[1].set_title("Pearson vs W2_Geary vs W2_Moran")

# Third Venn Diagram: Pearson, Regression, W2_Moran
venn3([genes_pearson, genes_regression, genes_w2_moran],
    set_labels=("Pearson", "Regression", "Spatial_W2_Moran"),
    ax=axes[2])
axes[2].set_title("Pearson vs Regression vs W2_Moran")
plt.suptitle("Venn Diagrams of Top Genes Across Methods", fontsize=16)
plt.tight_layout()
plt.show()

# Step 1: Identify Pearson-high genes
pearson_high = corr_dict["Pearson"].query("correlation > -0.5")["gene"]
# Step 2: Collect genes with strong negative spatial correlation
interesting_genes = set()
for method in ["Spatial_W1", "Spatial_W2_Geary", "Spatial_W2_Moran"]:
    spatial_low = corr_dict[method].query("correlation < -1")["gene"]
    overlap = set(pearson_high).intersection(spatial_low)
    interesting_genes.update(overlap)

symbol_map = corr_dict["Pearson"].set_index("gene")["symbol"]
symbol_df = symbol_map.loc[list(interesting_genes)].to_frame()

# Step 3: Merge correlation values and gene symbols
results = []
for method, df in corr_dict.items():
    filtered = df[df["gene"].isin(interesting_genes)].copy()
    filtered = filtered.set_index("gene")[["correlation"]]
    filtered.columns = [method]
    results.append(filtered)

# Add symbol (assuming it's consistent across all dict entries)
symbol_map = corr_dict["Pearson"].set_index("gene")["symbol"]
symbol_df = symbol_map.loc[list(interesting_genes)].to_frame()

# Step 4: Combine all into a single DataFrame
from functools import reduce
merged = reduce(lambda left, right: left.join(right, how="outer"), results)
merged = symbol_df.join(merged)

# Step 5: Display all columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
merged.reset_index()

merged['dif'] = merged['Spatial_W1'] - merged['Pearson']
### sort by the difference
merged_sorted = merged.sort_values(by='dif', ascending=True)
merged_sorted

### visualize the top 20 genes with the largest difference in a loop
for i in range(20):
    gene_symbol = merged_sorted['symbol'].values[i]
    gene_ensemble_id = merged_sorted.index[i]
    print(f"Top {i+1} gene: {gene_symbol} (Ensembl ID: {gene_ensemble_id})")
    # Plot the spatial distribution of the top genes
    fn.plot_gene_spatial(an_adata_sample, gene_ensemble_id, 
                         title=f"{gene_symbol} (Ensembl ID: {gene_ensemble_id})", cmap="viridis")


############### PLOT HEATMAP OF CORRELATIONS BETWEEN METHODS ######################
# Extract correlation values from each dataframe
s1 = spatial_corr_1[["gene", "correlation"]].set_index("gene").rename(columns={"correlation": "Spatial_W1"})
s2 = spatial_corr_2[["gene", "correlation"]].set_index("gene").rename(columns={"correlation": "Spatial_W2_Geary"})
s3 = spatial_corr_3[["gene", "correlation"]].set_index("gene").rename(columns={"correlation": "Spatial_W2_Moran"})
s4 = pearson_corr[["gene", "correlation"]].set_index("gene").rename(columns={"correlation": "Pearson"})

# Combine into a single DataFrame
corr_df = pd.concat([s1, s2, s3, s4], axis=1)

# Drop rows with any missing values
corr_df_clean = corr_df.dropna()

# Compute correlation matrix
correlation_matrix = corr_df_clean.corr()

# Plot heatmap
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
plt.title("Pairwise Correlation Between Gene-Residual Association Methods")
plt.tight_layout()
plt.show()


#############################################
######### Enrichment analysis using g:Profiler
from gprofiler import GProfiler
# Get top N gene symbols (drop NaNs first)
cor_df = corr_dict['Spatial_W1']  # or any other method you want to use
cor_df = corr_dict['Spatial_W2_Geary']  # or any other method you want to use
cor_df = corr_dict['Spatial_W2_Moran']  # or any other method you want to use
cor_df = corr_dict['Pearson']  # or any other method you want to use

num_pathway_genes = 800
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


NMF_loadings = adata_merged.uns["nmf_components"]
NMF_loadings_df = pd.DataFrame(NMF_loadings,
                               index=[f'NMF{i+1}' for i in range(NMF_loadings.shape[0])],
                               columns=adata_merged.var_names)

NMF_loadings_df = NMF_loadings_df.T  # Transpose to have genes as rows
gene_ensemble__map = fn.map_ensembl_to_symbol(NMF_loadings_df.index)
NMF_loadings_df['symbol'] = NMF_loadings_df.index.map(gene_ensemble__map)
factor_idx = 12
NMF_loadings_df_sorted = NMF_loadings_df.sort_values(by=f'NMF{factor_idx+1}', ascending=False)
print(f"Top 20 genes for NMF factor {factor_idx+1}:")
print(NMF_loadings_df_sorted[['symbol', f'NMF{factor_idx+1}']].head(20).to_string(index=False))
top_genes_nmf = NMF_loadings_df_sorted['symbol'].head(200).tolist()

gp = GProfiler(return_dataframe=True)
enrich_results_nmf = gp.profile(organism='hsapiens', query=top_genes_nmf)
enrich_results_nmf[['source', 'name', 'p_value', 'intersection_size']].head(10)
# Filter GO:BP terms only
go_bp_results_nmf = enrich_results_nmf[enrich_results_nmf['source'] == 'GO:BP']
print(go_bp_results_nmf[['name', 'p_value', 'intersection_size']].sort_values('p_value').head(20))
######################################################################


### visaulize the top NMF loadings genes
# Plot the top NMF loadings genes spatially
df = NMF_loadings_df_sorted.head(10)
for i in range(10):
    gene_ensemble_id = df.index[i]
    gene_symbol = df['symbol'].values[i]
    fn.plot_gene_spatial(an_adata_sample, gene_ensemble_id, 
                         title=gene_symbol, cmap="viridis")


## TODO: 
# run HiDDEN on the residuals from 30NMF regressed out - is there any data leakage?

