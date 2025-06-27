import os
import pandas as pd
import anndata
from scipy import io
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
from functools import reduce
from matplotlib_venn import venn3


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()





root_dir = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq'
########################
#### checking if the function works with a single puck
subdir = '2023-08-25_Puck_230117_01/'
puck_folder = os.path.join(root_dir, subdir)
adata = load_slide_seq_puck(puck_dir=puck_folder, puck_id=subdir)
########################

adata_merged = load_all_slide_seq_data(root_dir)


### Apply NMF
# Step 1: Normalize and log-transform - I DIDN'T USE THIS
sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)
adata_merged.layers["lognorm"] = adata_merged.X.copy()

# Step 2: Filter genes to keep top 2000 highly variable genes
sc.pp.highly_variable_genes(adata_merged, n_top_genes=2000
    , flavor='seurat_v3', subset=True, layer='lognorm')
# Step 3: Scale the data
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