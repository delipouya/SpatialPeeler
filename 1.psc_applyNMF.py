import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import functools
import numpy as np
import scipy.sparse as sp
import pandas as pd
import hiddensc
from hiddensc import utils, vis
import scanpy as sc
import scvi
import anndata
from sklearn.decomposition import NMF
import functools
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
import matplotlib.pyplot as plt

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
vis.visual_settings()

# Files to load
file_names = [
    "normal_A", "normal_B", "normal_C", "normal_D",
    "PSC_A", "PSC_B", "PSC_C", "PSC_D"
]
    
adata_dict = {}
cell_type_sets = {}
old_preprecess = False
scale_features = True
print_status = True
reverse_log = False

# Load and extract unique cell types
for fname in file_names:

    print(f"Loading data from {fname}...")
    
    PREFIX = fname.split('.')[0]  # Use the first file name as prefix
    at_data_dir = functools.partial(os.path.join, root_path,'SpatialPeeler','Data/PSC_liver/')
    adata = sc.read(at_data_dir(f'{PREFIX}.h5ad'))
    adata_dict[fname] = adata

    adata.obs['binary_label'] = adata.obs['disease']!='normal' #'primary sclerosing cholangitis'
    
    # ----------------------------
    # (remove extremely low-depth spots if needed.
    # Paper used UMI threshold for cells
    ### check how many spots have UMI total less than 200
    min_counts = 100
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    print("Number of spots with total counts < "+str(min_counts)+":", np.sum(adata.obs['total_counts'] < min_counts))
    ### filter out spots with less than 1000 UMI counts
    sc.pp.filter_cells(adata, min_counts=min_counts)  # adjust if you want a hard cutoff
    print("After filtering low-count spots, new shape:", adata.shape)
    # ----------------------------

    if print_status:
        ### print data statistics
        print("BEFORE expm1")
        X = adata.X
        xmin = X.min() if sp.issparse(X) else np.min(X)
        xmax = X.max() if sp.issparse(X) else np.max(X)
        print("min:", xmin, "max:", xmax)
        cell_sums = np.array(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
        print("cell sum min:", cell_sums.min())
        print("cell sum max:", cell_sums.max())
        print("CV:", cell_sums.std() / cell_sums.mean())
        print('--------------------------------------')

    # ----------------------------
    if reverse_log:
        print("Reversing log1p transformation...")
        # Reverse log1p transformation
        if sp.issparse(adata.X):
            adata.X = adata.X.copy()
            adata.X.data = np.expm1(adata.X.data)
        else:
            adata.X = np.expm1(adata.X)
    
    if old_preprecess:
        hiddensc.datasets.preprocess_data(adata)
        hiddensc.datasets.normalize_and_log(adata)

    elif scale_features:
        # unit-variance scaling WITHOUT centering 
        # Keeps non-negativity (important for NMF) while matching the "variance normalize" idea.
        sc.pp.scale(adata, zero_center=False)

    # ----------------------------
    if print_status:
        X = adata.X
        xmin = X.min() if sp.issparse(X) else np.min(X)
        xmax = X.max() if sp.issparse(X) else np.max(X)
        cell_sums = np.array(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)

        if reverse_log:
            print("AFTER reverse log1p")
        else:
            print("AFTER scaling")
        
        print("min:", xmin, "max:", xmax)
        print("cell sum min:", cell_sums.min())
        print("cell sum max:", cell_sums.max())
        print("CV:", cell_sums.std() / cell_sums.mean())
        print("--------------------------------------")

    
cell_types = adata.obs['cell_type'].unique()
cell_type_sets[fname] = set(cell_types)

# Add batch info before merging
for fname, ad in adata_dict.items():
    ad.obs["batch"] = fname  # This will let HiDDEN know the origin

### convert adata_dict to a list
adata_list = list(adata_dict.values())


for adata in adata_list:
    print(adata.obs['batch'][0], adata.shape)
print(adata_dict.items())

var_names_list = [set(adata.var_names) for adata in adata_list]
### check if the var_names are the same across all adatas
common_var_names = set.intersection(*var_names_list)
print("Number of common genes across all datasets:", len(common_var_names))


adata_merged = anndata.concat(
        adata_list, 
        label='batch', 
        axis ='obs',
        join='outer',
        keys=[adata.obs['batch'][0] for adata in adata_list]
    )
print("Merged adata shape:", adata_merged.shape)

adata_merged.obs['batch'] = adata_merged.obs['batch'].astype(str)
adata_merged.obs['sample_id'] = adata_merged.obs['batch'].astype(str)

print(adata_merged.shape)

##################
# 2) basic gene filtering 
# Paper: remove genes detected in fewer than ~1/500 cells: min_cells = max(1, n_cells//500)
min_cells = max(1, adata_merged.n_obs // 500)
### print teh number of genes ot be removed
num_genes_before = adata_merged.n_vars
print(f"Number of genes before filtering: {num_genes_before}")
sc.pp.filter_genes(adata_merged, min_cells=min_cells)
num_genes_after = adata_merged.n_vars
print(f"Number of genes after filtering (min_cells={min_cells}): {num_genes_after}, removed {num_genes_before - num_genes_after} genes.")
##################

print(adata_merged.shape)


################## Adding HVG selection - it was not included in the original code ----------------
USE_ALL_GENES = False
if USE_ALL_GENES:
    pass  # keep all genes after min_cells filtering
else:
    # 3) HVG selection across pooled data, but batch-aware to avoid puck dominance ----
    sc.pp.highly_variable_genes(
        adata_merged,
        n_top_genes=2000,
        batch_key="batch",     # use batch as the "sample" grouping
        flavor="seurat_v3",
        subset=True
    )
print(f"HVG selection: kept {adata_merged.n_vars} genes.")
# ---------------------------------------------

### Apply NMF
n_factors = 10 #30  # or choose based on elbow plot, coherence, etc.
nmf_model = NMF(n_components=n_factors, init='nndsvda', 
                random_state=RAND_SEED, max_iter=1000)
# X must be dense; convert if sparse
X = adata_merged.X
if sp.issparse(X): 
    X = X.toarray()

W = nmf_model.fit_transform(X)      # (cells/spots x factors)
H = nmf_model.components_           # (factors x genes)

adata_merged.obsm["X_nmf"] = W
adata_merged.uns["nmf"] = {
    "H": H,
    "genes": adata_merged.var_names.to_list(),
    "n_factors": n_factors,
    "model_params": nmf_model.get_params(),
}

#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG_NMF10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_revLog_varScale_2000HVG_NMF10.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_10_varScale_2000HVG_filtered.h5ad'
adata_merged.write_h5ad(file_name)


# ---------------------------------------------
##### Identifying the optimal number of factors (K) using MSE
adata_merged = sc.read_h5ad(file_name)
# Data - H.W = residuals -> split based on the case-control label
# Calculate MSE and then plot it over various values of K

### identify the min and max and median of the original data
data_min = adata_merged.X.min()
data_max = adata_merged.X.max()
data_median = np.median(adata_merged.X)
print(f"Original data - min: {data_min}, max: {data_max}, median: {data_median}")





case_mse_list = []
control_mse_list = []
for k in range(5, 31):
    
    ############## Re-run NMF with different K ##############
    nmf_model = NMF(n_components=k, init='nndsvda', 
                    random_state=RAND_SEED, max_iter=1000)
    X = adata_merged.X
    if sp.issparse(X): 
        X = X.toarray()
    W = nmf_model.fit_transform(X)  # cell × factor matrix
    H = nmf_model.components_        # factor × gene matrix
    adata_merged.obsm["X_nmf"] = W
    adata_merged.uns["nmf_components"] = H
    #################################################

    W = adata_merged.obsm["X_nmf"][:,0:k] # spots x factors
    H = adata_merged.uns["nmf_components"][0:k,:] # factors x genes

    print(f"K: {k}, W shape: {W.shape}, H shape: {H.shape}")
    # Reconstruct the data    
    V_reconstructed = np.dot(W, H)
    residuals = adata_merged.X - V_reconstructed

    residuals_case = residuals[adata_merged.obs['binary_label'] == True]
    residuals_control = residuals[adata_merged.obs['binary_label'] == False]

    case_mse = np.sum(np.square(residuals_case))/residuals_case.shape[0]
    control_mse = np.sum(np.square(residuals_control))/residuals_control.shape[0]

    case_mse_list.append(case_mse)
    control_mse_list.append(control_mse)


# save the lists as a dataframe
mse_df = pd.DataFrame({
    'K': range(5, 31),
    'Case MSE': case_mse_list,
    'Control MSE': control_mse_list
})
mse_df.to_csv(os.path.join(root_path, 'SpatialPeeler', 'Data/PSC_liver', 
                           'mse_nmf_factors.csv'), index=False)

#X-axis: K, y-axis: MSE (case/control). Do the two lines become closer as the value of K gets bigger? 
plt.figure(figsize=(10, 6))
plt.plot(range(5, 31), case_mse_list, label='Case MSE', marker='o')
plt.plot(range(5, 31), control_mse_list, label='Control MSE', marker='o')
plt.xlabel('Number of NMF Factors (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE for Case and Control Groups vs Number of NMF Factors')
plt.legend()
plt.grid()
plt.show()

### plot the difference between the two lines
plt.figure(figsize=(10, 7))
plt.plot(range(5, 31), np.array(control_mse_list) -  np.array(case_mse_list) , 
         label='Control - Case MSE', marker='o')
plt.xlabel('Number of NMF Factors (K)')
plt.ylabel('Difference in MSE (Control - Case)')
plt.ylim([0, 200])
plt.title('Difference in MSE for Control and Case Groups vs Number of NMF Factors')
plt.legend()
plt.grid()
plt.show()

