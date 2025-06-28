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


from SpatialPeeler import helpers 


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
helpers.set_random_seed(helpers.RANDOM_SEED)
helpers.print_module_versions([sc, anndata, scvi, hiddensc])
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
