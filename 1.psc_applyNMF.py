import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import functools
import numpy as np
import scipy.sparse as sp
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
                              label="sample_id", keys=adata_dict.keys(), 
                              merge="same")

adata_merged.obs['batch'] = adata_merged.obs['batch'].astype(str)
adata_merged.obs['sample_id'] = adata_merged.obs['sample_id'].astype(str)

### Apply NMF
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

##### Identifying the optimal number of factors (K) using MSE
adata_merged = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad')
# Data - H.W = residuals -> split based on the case-control label
# Calculate MSE and then plot it over various values of K

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

import matplotlib.pyplot as plt
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

