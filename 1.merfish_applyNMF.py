import scanpy as sc
import pandas as pd
from scipy import io
import os
import sys
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
import anndata as ad

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
import glob

normalize_log1p = False
adata_dict = {}
for folder_name in os.listdir("/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/"):
    if not os.path.isdir(os.path.join("/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/", folder_name)):
        continue

    # Load expression matrix and metadata files
    exp_file = glob.glob(f"/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/{folder_name}/*_exp.mtx.gz")[0]
    gene_file = glob.glob(f"/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/{folder_name}/*_feats.txt.gz")[0]
    meta_file = glob.glob(f"/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/{folder_name}/*_meta.tsv.gz")[0]
    obs_extra_file = glob.glob(f"/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/{folder_name}/*_obs.txt.gz")[0]

    X = io.mmread(exp_file).T.tocsr()  # transpose 
    genes = pd.read_csv(gene_file, header=None)[0].values
    meta = pd.read_csv(meta_file, sep="\t")
    obs_extra = pd.read_csv(obs_extra_file, sep="\t")
    
    # Build AnnData object
    adata = sc.AnnData(X.T, obs=meta, var=pd.DataFrame(index=genes))
    adata.obs = adata.obs.join(obs_extra)
    adata.obs['sample_id'] = folder_name

    if folder_name in ['s11r1_sham_2', 's5r1_sham_1']:
        adata.obs['Condition'] = 'sham'
    elif folder_name in ['s11r0_infected_2', 's5r0_infected_1']:
        adata.obs['Condition'] = 'infected'

    adata.obsm["spatial"] = adata.obs[["spot_x", "spot_y"]].values


    # Optional per-sample normalization and log1p
    if normalize_log1p:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers['lognorm'] = adata.X.copy()  # preserve transformed data
    
    adata_dict[folder_name] = adata
    

merged_adata = ad.concat(
    adata_dict,
    axis=0,  # Concatenate along observations (samples)
    join='outer', # Keep all variables present in any AnnData object
    label='batch', # Add a column 'batch' to .obs indicating the origin of each observation
    merge='same', # Strategy for merging elements aligned to the alternative axis (e.g., .var)
    uns_merge='same' # Strategy for merging uns (unstructured annotation)
)

sc.pp.normalize_total(merged_adata, target_sum=1e4)
sc.pp.log1p(merged_adata)

### Apply NMF
n_factors = 30  # or choose based on elbow plot, coherence, etc.
nmf_model = NMF(n_components=n_factors, init='nndsvda', 
                random_state=RAND_SEED, max_iter=1000)
# X must be dense; convert if sparse
X = merged_adata.X
if sp.issparse(X): 
    X = X.toarray()

print(X.shape)


W = nmf_model.fit_transform(X)  # cell × factor matrix
H = nmf_model.components_        # factor × gene matrix

merged_adata.obsm["X_nmf"] = W
merged_adata.uns["nmf_components"] = H

merged_adata.obs_names = merged_adata.obs_names.astype(str)
merged_adata.var_names = merged_adata.var_names.astype(str)

### (3880933, 140) - around 4 million spots
outp = "/home/delaram/SpatialPeeler/Data/MERFISH_polyomavirus/merged_adata_infected_sham_allnorm_NMF30.h5ad"
merged_adata.write_h5ad(outp, compression="gzip")
