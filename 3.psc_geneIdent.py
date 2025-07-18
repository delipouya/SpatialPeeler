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


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()

adata = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad')

with open('/home/delaram/SpatialPeeler/Data/PSC_liver/results.pkl', 'rb') as f:
    results = pickle.load(f)


########################################################################
######################## Gene-Based analysis
########################################################################
factor_idx = 22#8 #12

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
    title_prefix="HiDDEN predictions", counter=factor_idx+1)

plot.plot_grid(adata_by_sample, sample_ids, key="1_p_hat", 
    title_prefix="HiDDEN predictions", counter=factor_idx+1)


#4:7 are PSC samples, 0:3 are normal samples - initial analysis on sample #5, 4
sample_id_to_check = 1 #5
an_adata_sample = adata_by_sample[sample_ids[sample_id_to_check]]


PATTERN_COND = 'LOF'#'GOF'
expr_matrix = an_adata_sample.X.toarray() if issparse(an_adata_sample.X) else an_adata_sample.X  # shape: (n_spots, n_genes)
p_hat_vector = an_adata_sample.obs['p_hat']  # shape: (n_spots,)
neg_p_hat_vector = an_adata_sample.obs['1_p_hat']  # shape: (n_spots,)
pattern_vector = p_hat_vector if PATTERN_COND == 'GOF' else neg_p_hat_vector


pearson_corr = gid.pearson_correlation_with_pattern(expr_matrix, pattern_vector, 
                                                     gene_names=an_adata_sample.var_names)

NMF_idx_values = an_adata_sample.obsm["X_nmf"][:,factor_idx]
weighted_pearson_corr = gid.weighted_pearson_correlation_with_pattern(expr_matrix, pattern_vector, 
                                                                      W_vector=NMF_idx_values, 
                                                                      gene_names=an_adata_sample.var_names, 
                                                                      scale=True)

symbols= weighted_pearson_corr['gene'].map(hlps.map_ensembl_to_symbol(weighted_pearson_corr['gene'].tolist()))
weighted_pearson_corr['symbols'] = symbols

regression_res = gid.regression_with_pattern(expr_matrix, pattern_vector,
                                               gene_names=an_adata_sample.var_names, 
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

for key, cor_df in corr_dict.items():
    # Ensure 'gene' column is present
    if 'gene' not in cor_df.columns:
        cor_df['gene'] = an_adata_sample.var_names
    
    cor_df = cor_df.sort_values("correlation", ascending=False)
    id_to_symbol = hlps.map_ensembl_to_symbol(cor_df['gene'].tolist())
    cor_df['symbol'] = cor_df['gene'].map(id_to_symbol)
    cor_df['ensemble_id'] = cor_df['gene']  # Keep the original gene ID
    print(f"Top 10 genes for {key}:")
    print(cor_df[['symbol', 'correlation']].head(20).to_string(index=False))
    corr_dict[key] = cor_df


num_genes_viz = 4
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


#4:7 are PSC samples, 0:3 are normal samples - initial analysis on sample #5, 4
for sample_id_to_check in range(0, 4):
    an_adata_sample_2 = adata_by_sample[sample_ids[sample_id_to_check]]
    print(f"Sample {sample_ids[sample_id_to_check]}:")
    df = top_genes["Regression"]
    i = 1
    gene_symbol = df['symbol'].values[i]
    print(f"Top {i+1} gene for {key}: {gene_symbol}")
    plot.plot_gene_spatial(an_adata_sample_2, df['ensemble_id'].values[i], 
                            title=f"{key} - {gene_symbol}", cmap="viridis")

x_axis = 'Regression'
y_axis = 'Pearson' 
df_vis = pd.merge(corr_dict[x_axis], corr_dict[y_axis], on='symbol', how='inner')
plt.figure(figsize=(5, 5))
plt.scatter(df_vis["correlation_x"], 
            df_vis["correlation_y"], alpha=0.4, s=10)
plt.xlabel(x_axis + " Cor")
plt.ylabel(y_axis + " Cor")
plt.legend()
plt.show()


n_gene_pathway_thr = 300
method_name = 'Pearson'  
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


plt.figure(figsize=(20, 10))
sns.heatmap(average_expression_df.drop(columns='symbol'),
            vmin=0, vmax=4,  # Adjust these limits based on your data
            cmap='viridis', annot=True, fmt=".02f",
            cbar_kws={'label': 'Average Expression'},
            xticklabels=average_expression_df.columns,
            yticklabels=average_expression_df['symbol'].values,
            linewidths=.5, linecolor='black')
plt.title("Average Expression of Top Spatial " + method_name +
          " Genes Across Samples")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.tight_layout()
plt.show()


n_gene_venn = 50
genes_pearson = set(corr_dict['Pearson']["gene"][:n_gene_venn])
genes_regression = set(corr_dict["Regression"]["gene"][:n_gene_venn])

# Plotting two Venn diagrams due to 3-set limitation
venn3([genes_pearson, genes_w1, genes_regression],
    set_labels=("Pearson", "Spatial_W1", "regression"))



#############################################
######### Enrichment analysis using g:Profiler

method = 'Regression'#'Regression' 
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