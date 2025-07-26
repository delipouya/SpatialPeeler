import os
import pandas as pd
import anndata
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
from sklearn.decomposition import NMF
import functools

from SpatialPeeler import importing as imp
from SpatialPeeler import helpers as hlps
#from scipy.stats import mannwhitneyu
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)


import scanpy as sc
import pandas as pd
from scipy import io
from scipy.sparse import csr_matrix

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()
root_dir = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq'

#######################################
#### importing the unprocessed data  ####
#subdir = '2023-08-25_Puck_230117_01/'
#puck_dir = os.path.join(root_dir, subdir)
#puck_id = subdir.split('_')[2]  # Extract puck ID from subdir name
#adata = imp.load_slide_seq_puck(puck_dir=puck_dir, puck_id=subdir)
#adata_merged = imp.load_all_slide_seq_data(root_dir)
########################################




base = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline"
#scaled_counts = io.mmread(f"{base}/all_final_cropped_pucks_standardpipeline_scaled_counts.mtx").tocsr()

merged_counts = io.mmread(f"{base}/all_final_cropped_pucks_standardpipeline_counts_merged.mtx").tocsr()
obs_names = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_barcodes.tsv", header=None)[0].values
var_names = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_features.tsv", header=None)[0].values
obs = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_metadata.csv", index_col=0)

all(obs.index == obs_names) ### Check if obs.index matches obs_names
adata_merged = sc.AnnData(X=merged_counts.T, ### X should be(cells x genes)
                   obs= obs)
adata_merged.obs_names = obs_names
adata_merged.var_names = var_names

### Apply NMF

# Step 1: Normalize and log-transform - I DIDN'T USE THIS
sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)
adata_merged.layers["lognorm"] = adata_merged.X.copy()
sc.pp.highly_variable_genes(adata_merged, n_top_genes=2000, flavor='seurat_v3', subset=True, layer='lognorm')
sc.pp.scale(adata_merged, max_value=10, zero_center=True, layer='lognorm')


# Step 4: Run NMF
# Note: NMF requires dense input, so we convert the sparse matrix to dense.
# You can also use the `init='nndsvda'` option for better initialization
# and set `max_iter` to a higher value for convergence.
# Choose the number of factors (components) based on prior knowledge or elbow plot.
# For example, you can use 30 factors as a starting point.

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


### save adata object with NMF results  
adata_merged.write_h5ad('/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad')