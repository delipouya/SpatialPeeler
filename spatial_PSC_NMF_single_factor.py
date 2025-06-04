import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import functools
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import hiddensc
from hiddensc import utils, files, vis
import scanpy as sc
import scvi
import anndata
from sklearn.decomposition import NMF
import functools
from scipy.linalg import svd
from numpy.linalg import LinAlgError
from hiddensc import models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit
#from scipy.stats import mannwhitneyu
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
from hiddensc import models
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit
from scipy.sparse import issparse
import functions as fn

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()


# --------- Import data and preprocess ---------
# Files to load
file_names = [
    "normal_A", "normal_B", "normal_C", "normal_D",
    "PSC_A", "PSC_B", "PSC_C", "PSC_D"
]

    
adata_dict = {}
cell_type_sets = {}
# Load and extract unique cell types
for fname in file_names:
    
    PREFIX = fname.split('.')[0]  # Use the first file name as prefix
    at_data_dir = functools.partial(os.path.join, root_path,'SpatialPeeler','data_PSC')
    adata = sc.read(at_data_dir(f'{PREFIX}.h5ad'))
    adata_dict[fname] = adata

    adata.obs['binary_label'] = adata.obs['disease']!='normal' #'primary sclerosing cholangitis'

    hiddensc.datasets.preprocess_data(adata)
    hiddensc.datasets.normalize_and_log(adata)
    cell_types = adata.obs['cell_type'].unique()
    cell_type_sets[fname] = set(cell_types)

# Add batch info before merging
for fname, ad in adata_dict.items():
    ad.obs["batch"] = fname  # This will let HiDDEN know the origin

# Concatenate all datasets
adata_merged = anndata.concat(adata_dict.values(), 
                              label="sample_id", keys=adata_dict.keys(), merge="same")
adata_merged.obs['batch'] = adata_merged.obs['batch'].astype(str)
adata_merged.obs['sample_id'] = adata_merged.obs['sample_id'].astype(str)

### Apply NMF
# Step 1: Normalize and log-transform - I DIDN'T USE THIS
#sc.pp.normalize_total(adata_merged, target_sum=1e4)
#sc.pp.log1p(adata_merged)
#adata_merged.layers["lognorm"] = adata_merged.X.copy()
# ------------------

n_factors = 30  # or choose based on elbow plot, coherence, etc.
nmf_model = NMF(n_components=n_factors, init='nndsvda', 
                random_state=RAND_SEED, max_iter=1000)
# X must be dense; convert if sparse
X = adata_merged.X
if sp.issparse(X): 
    X = X.toarray()
W = nmf_model.fit_transform(X)  # cell × factor matrix
H = nmf_model.components_        # factor × gene matrix

adata_merged.obsm["X_nmf"] = W
adata_merged.uns["nmf_components"] = H
#adata_merged.write_h5ad(os.path.join(root_path, 'SpatialPeeler', 
#                                     'data_PSC', 'PSC_NMF_30.h5ad'))
# ---------------------------------------------


adata_merged = sc.read_h5ad(os.path.join(root_path, 'SpatialPeeler','data_PSC', 'PSC_NMF_30.h5ad'))

sample_ids = adata_merged.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in adata_merged.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())

# Plot for each sample and NMF factor 1–4
#for sid in sample_ids:
#    print(f"Plotting NMF for sample {sid}")
#    for k in range(n_factors-1):
#        fn.plot_spatial_nmf(adata_by_sample[sid], k, sample_id=sid)

total_factors = adata_merged.obsm["X_nmf"].shape[1]
adata = adata_merged

nmf_factors = adata.obsm['X_nmf'][:, :8]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])
nmf_df['sample_id'] = adata.obs['sample_id'].values
nmf_long = nmf_df.melt(id_vars='sample_id', 
                       var_name='Factor', 
                       value_name='Score')

plt.figure(figsize=(8, 35))
sns.violinplot(y="Factor", x="Score", hue="sample_id", data=nmf_long, inner="box", palette="Set2")
plt.title("Distribution of NMF Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


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


'''
## run the model.py script within the hiddensc package
num_pcs, ks, ks_pval = determine_pcs_heuristic_ks_nmf(adata=adata, 
                                                                  orig_label="binary_label", 
                                                                  max_pcs=60)
ptimal_num_pcs_ks = num_pcs[np.argmax(ks)]
plt.figure(figsize=(4, 2))
plt.scatter(num_pcs, ks, s=10, c='k', alpha=0.5)
plt.axvline(x = optimal_num_pcs_ks, color = 'r', alpha=0.2)
plt.scatter(optimal_num_pcs_ks, ks[np.argmax(ks)], s=20, c='r', marker='*', edgecolors='k')
plt.xticks(np.append(np.arange(0, 60, 15), optimal_num_pcs_ks), fontsize=18)
plt.xlabel('Number of PCs')
plt.ylabel('KS')
plt.show()
print(f"Optimal number of PCs: {optimal_num_pcs_ks}")
'''

optimal_num_pcs_ks = total_factors
# Set up HiDDEN input
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['status'] = adata.obs['binary_label'].astype(int).values

# Run factor-wise HiDDEN-like analysis (logistic regression on single factors)
results = fn.single_factor_logistic_evaluation(
    adata, factor_key="X_pca", max_factors=optimal_num_pcs_ks
)

########################  VISUALIZATION  ########################
for i in range(14,optimal_num_pcs_ks): #optimal_num_pcs_ks
    fn.plot_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i)
    fn.plot_logit_p_hat_vs_nmf_by_sample(adata, results, sample_ids, factor_idx=i)

########################
# Store output for the best-performing factor (e.g., first one, or pick based on AUC)
counter = 29
for result in results:
    print(f"Factor {result['factor_index'] + 1}:")
    print(f"  p_hat mean: {result['p_hat'].mean():.4f}")
    print(f"  KMeans labels: {np.unique(result['kmeans_labels'])}")
    print(f"  Status distribution: {np.bincount(result['status'])}")

    adata.obs['p_hat'] = result['p_hat']
    adata.obs['new_label'] = result['kmeans_labels']
    adata.obs['new_label'] = adata.obs['new_label'].astype('category')
    adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
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
    fn.plot_grid(adata_by_sample, sample_ids, key="X_nmf", 
    title_prefix="NMF Factor", counter=counter, from_obsm=True, factor_idx=3)
    # For p_hat
    fn.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix="HiDDEN predictions", counter=counter)
    # For raw residuals
    fn.plot_grid(adata_by_sample, sample_ids, key="raw_residual", 
    title_prefix="Raw Residual", counter=counter)
    # For Pearson residuals
    fn.plot_grid(adata_by_sample, sample_ids, key="pearson_residual", 
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



########################################################################
######################## Gene-Based analysis
########################################################################
factor_idx = 12

result = results[factor_idx] 
adata.obs['p_hat'] = result['p_hat']
adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
adata.obs['raw_residual'] = result['raw_residual']
adata.obs['pearson_residual'] = result['pearson_residual']
adata.obs['deviance_residual'] = result['deviance_residual']

print(f"Factor {result['factor_index'] + 1}:")
### sanity check
fn.plot_grid(adata_by_sample, sample_ids, key="p_hat", 
    title_prefix="HiDDEN predictions", counter=factor_idx+1)

# Copy adata per sample for plotting
sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}

an_adata_sample = adata_by_sample[sample_ids[5]]
# Compute spatial weights using squidpy
# Get expression matrix and coordinates

#### calculating spatial weights using squidpy
import squidpy as sq
sq.gr.spatial_neighbors(an_adata_sample, coord_type="generic", n_neighs=10) #### how many neighbors to use?
W1 = an_adata_sample.obsp["spatial_connectivities"]
W1_array = np.array(W1.todense())  # if W1 is sparse, convert to dense matrix


##### calculating spatial weights using custom function
coords = an_adata_sample.obsm['spatial']  # shape: (n_spots, 2)
W2_geary = compute_spatial_weights(coords, k=10, mode="geary")
W2_moran = compute_spatial_weights(coords, k=10, mode="moran")

print('squidpy-W mean: ', W1_array.mean(),
       'squidpy-W sd: ', W1_array.std())
print('Geary-W mean: ', W2_geary.mean(),
         'Geary-W sd: ', W2_geary.std())
print('Moran-W mean: ', W2_moran.mean(),
            'Moran-W sd: ', W2_moran.std())

expr_matrix = an_adata_sample.X.toarray() if issparse(an_adata_sample.X) else an_adata_sample.X  # shape: (n_spots, n_genes)
residual_vector = an_adata_sample.obs['pearson_residual']  # shape: (n_spots,)
spatial_corr_1 = spatial_weighted_correlation_matrix(expr_matrix, residual_vector, W1, an_adata_sample.var_names)
spatial_corr_2 = spatial_weighted_correlation_matrix(expr_matrix, residual_vector, W2_geary, an_adata_sample.var_names)
spatial_corr_3 = spatial_weighted_correlation_matrix(expr_matrix, residual_vector, W2_moran, an_adata_sample.var_names)
pearson_corr = pearson_correlation_with_residuals(expr_matrix, residual_vector, gene_names=an_adata_sample.var_names)

corr_dict = {
    "Spatial_W1": spatial_corr_1,
    "Spatial_W2_Geary": spatial_corr_2,
    "Spatial_W2_Moran": spatial_corr_3,
    "Pearson": pearson_corr
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


top_genes = {}
for key, cor_df in corr_dict.items():
    top_genes[key] = cor_df.sort_values("correlation", ascending=True).head(1)[['ensemble_id','symbol', 'correlation']]
top_genes

def plot_gene_spatial(adata, gene_id, title, cmap="viridis"):
    """Plot the expression of a single gene over spatial coordinates."""
    if gene_id not in adata.var_names:
        print(f"Gene {gene_id} not found in adata.var_names.")
        return
    gene_idx = adata.var_names.get_loc(gene_id)
    expr = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, "toarray") else adata.X[:, gene_idx]
    coords = adata.obsm["spatial"]
    plt.figure(figsize=(5, 4))
    plt.scatter(coords[:, 0], coords[:, 1], c=expr, cmap=cmap, s=10)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(label="Expression")
    plt.tight_layout()
    plt.show()


for key, df in top_genes.items():
    print(f"Top 10 genes for {key}:")    # Plot the top gene spatially
    gene_ensemble_id = df['ensemble_id'].values[0]
    plot_gene_spatial(an_adata_sample, gene_ensemble_id, title=f"Top Gene ({key}): {df['symbol'].values[0]}", cmap="viridis")



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

top_genes = cor_df.dropna(subset=['symbol']).sort_values("correlation", ascending=True).head(200)["symbol"].tolist()
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



import squidpy as sq
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=8)
W = adata.obsp["spatial_connectivities"]

# Identifying genes assoiciated with the residuals
for result in results:
    expr_matrix = adata.X.toarray() if issparse(adata.X) else adata.X  # shape: (n_spots, n_genes)
    pearson_residuals = result['pearson_residual']    # shape: (n_spots,)
    coords = adata.obsm['spatial']   # shape: (n_spots, 2)
    # Run the correlation
    spatial_corr = fn.spatial_weighted_correlation_matrix(expr_matrix, pearson_residuals, 
                                                          coords, mode="geary")
    top_genes = spatial_corr.sort_values(ascending=False).head(30)


## TODO: 
# check the time it takes to run the correlation function
# make a standard correlation function - spatially unaware
# how to interpret the scatter plot of p_hat vs nmf scores
# run HiDDEN on the residuals from 30NMF regressed out - is there any data leakage?


from scipy.sparse import issparse

if not issparse(expr_matrix):
    from scipy.sparse import csr_matrix
    expr_matrix = csr_matrix(expr_matrix)







######################################################################
############## Combining p_hat values across factors ##############
######################################################################

### Gives an average prediction confidence over factors.
# If your goal is to highlight subtle, distributed signals across factors in a robust way:
combined_p_hat_1 = np.mean([r['p_hat'] for r in results], axis=0)


#### Weight by a proxy for factor importance, e.g., variance, AUC, or KS score.
# Example: weight by factor variance
variances = [np.var(r['p_hat']) for r in results]
weights = np.array(variances) / np.sum(variances)
combined_p_hat_2 = np.average([r['p_hat'] for r in results], axis=0, weights=weights)


#### Fit a meta-model (e.g., logistic regression) on the matrix of all p_hat values.
# Stack all p_hat values into a feature matrix (n_cells x n_factors)
X_stack = np.column_stack([r['p_hat'] for r in results])
y = adata.obs['status'].values
# Fit logistic regression on top
clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X_stack, y)
combined_p_hat_3 = clf.predict_proba(X_stack)[:, 1]


# If instead you want to identify cells where any one factor is strongly perturbed (more conservative):
combined_p_hat_4 = np.max([r['p_hat'] for r in results], axis=0)

# Store the combined p_hat in adata
phat_combined = np.column_stack([
    combined_p_hat_1,  # or combined_p_hat_2, 3, or 4
    combined_p_hat_2,
    combined_p_hat_3,
    combined_p_hat_4
]) 

adata.obs['p_hat_combined_1'] = phat_combined[:, 0]
adata.obs['p_hat_combined_2'] = phat_combined[:, 1]
adata.obs['p_hat_combined_3'] = phat_combined[:, 2]
adata.obs['p_hat_combined_4'] = phat_combined[:, 3]

# Stack factor-wise p_hat values
p_hat_matrix = np.column_stack([r["p_hat"] for r in results])  # shape: (n_cells, n_factors)
W = adata.obsm["X_nmf"][:, :p_hat_matrix.shape[1]]  # shape: (n_cells, n_factors)
# Normalize weights row-wise so they sum to 1
W_normalized = W / W.sum(axis=1, keepdims=True)
# Compute combined p_hat per cell as weighted sum
combined_p_hat_spatial = np.sum(W_normalized * p_hat_matrix, axis=1)
adata.obs["p_hat_combined_5"] = combined_p_hat_spatial.astype(np.float32)

### calculate combined p_hat spatially - based on NMF weights

def plot_combined_p_hat(a, sample_id, p_hat_col):
        spatial = a.obsm["spatial"]
        p_hat = a.obs[p_hat_col].values

        upper = np.quantile(p_hat, 0.99)
        p_hat = np.minimum(p_hat, upper)

        plt.figure(figsize=(5, 5))
        sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10)
        plt.axis("equal")
        plt.title(f"Combined p_hat {p_hat_col} – {sample_id}", fontsize=14)
        plt.colorbar(sc, label="p_hat")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()

# Plot the combined p_hat spatially
for i in range(5):
    ############# plotting all samples in a grid
    
    sample_ids = adata.obs['sample_id'].unique().tolist()
    adata_by_sample = {
        sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
        for sample_id in sample_ids
    }
    # Plot 2x4 grid for each combined p_hat
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    for j, sid in enumerate(sample_ids):
        a = adata_by_sample[sid]
        p_hat_col = f'p_hat_combined_{i+1}'
        spatial = a.obsm["spatial"]
        p_hat = a.obs[p_hat_col].clip(upper=np.quantile(adata.obs[p_hat_col], 0.99))

        axs[j].scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10)
        axs[j].set_title(sid, fontsize=10)
        axs[j].axis("off")
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="p_hat")
    plt.suptitle(f"Combined p_hat {p_hat_col} across spatial coordinates", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


# Save the updated AnnData object with combined p_hat
### make violin plots for all combined p_hat across samples
df_violin = adata.obs[["sample_id", 
                       "p_hat_combined_1", 
                        "p_hat_combined_2",
                        "p_hat_combined_3",
                        "p_hat_combined_4",
                        "p_hat_combined_5"
                       ]].copy()

for i in range(1, 6):
    df_violin = adata.obs[["sample_id", 
                           "p_hat_combined_"+str(i)]].copy()
    df_violin.columns = ["sample_id", "p_hat_combined_"+str(i)]
    
    # Plot single violin plot for each combined p_hat
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        x="sample_id",
        y="p_hat_combined_"+str(i),  # Change this to 2, 3, etc. for other combined p_hat
        hue="sample_id",
        data=df_violin,
        palette="Set2",
        density_norm="width",
        inner=None,
        legend=False
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribution of p_hat_combined_{i} per sample")
    plt.tight_layout()
    plt.show()


# Plot multi-violin plots
df_violin_long = df_violin.melt(
    id_vars="sample_id",
    var_name="Combination",
    value_name="p_hat"
)
plt.figure(figsize=(20, 8))
sns.violinplot(
    x="sample_id", y="p_hat", hue="Combination", data=df_violin_long,
    split=False, inner="box", palette="Set2", dodge=True
)
plt.xticks(rotation=45, ha="right")
plt.title("Comparison of Combined p_hat Scores Across Samples")
plt.xlabel("Sample ID")
plt.ylabel("p_hat")
plt.legend(title="Combination", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()





psc_mask = adata.obs["disease"] != "normal"
psc_adata = adata[psc_mask].copy()
psc_sample_ids = psc_adata.obs["sample_id"].unique()
from sklearn.cluster import KMeans

cluster_labels = pd.Series(index=psc_adata.obs_names, dtype="object")

for sid in psc_sample_ids:
    sample_mask = psc_adata.obs["sample_id"] == sid
    p_hat_vals = psc_adata.obs.loc[sample_mask, "p_hat_combined_1"].values.reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
    labels = kmeans.fit_predict(p_hat_vals)

    # Optional: force cluster 1 to be higher-mean cluster
    mean0, mean1 = p_hat_vals[labels == 0].mean(), p_hat_vals[labels == 1].mean()
    # If mean0 is greater, flip labels to ensure cluster 1 has higher mean
    if mean0 > mean1:
        labels = 1 - labels
        temp = mean0
        mean0 = mean1
        mean1 = temp
    # Print means for debugging
    print(f"Sample {sid}: mean0={mean0:.4f}, mean1={mean1:.4f}")

    cluster_labels.loc[sample_mask] = labels

psc_adata.obs["kmeans_p_hat_combined_1"] = cluster_labels.astype("category")


def plot_clusters_per_sample(adata_dict, col="kmeans_p_hat_combined_1", cmap="tab10"):
    n = len(adata_dict)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, (sid, a) in enumerate(adata_dict.items()):
        spatial = a.obsm["spatial"]
        clusters = a.obs[col].astype(int).values

        axs[i].scatter(spatial[:, 0], spatial[:, 1], c=clusters, cmap=cmap, s=10)
        axs[i].set_title(sid)
        axs[i].axis("off")

    for j in range(i+1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("KMeans Clustering of p_hat_combined_1 in PSC samples", fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage:
# Split adata by sample
psc_by_sample = {
    sid: psc_adata[psc_adata.obs["sample_id"] == sid].copy()
    for sid in psc_sample_ids
}

plot_clusters_per_sample(psc_by_sample, cmap="Paired")  # Or try "Paired", "Accent", "Dark2"

###
